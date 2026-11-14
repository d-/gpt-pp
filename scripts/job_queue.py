#!/usr/bin/env python
"""A small GPU job queue: run a list of jobs, each on its own card or cards, until all are done.

    python scripts/job_queue.py experiments/ladder.jsonl --gpus 0,1,2,3 [--watch] [--poll 5] [--state FILE]
    tmux new-session -d -s queue "python scripts/job_queue.py experiments/ladder.jsonl --gpus 0,1,2,3"

The jobs file has one JSON object a line (blank lines and lines starting with # are skipped):

    {"name": "ladder-gelu", "cmd": "$PY -m gptpp.train --preset gptpp-ladder --run ladder-gelu --set act=gelu"}
    {"name": "base", "gpus": 4, "cmd": "$PY -m torch.distributed.run --nproc_per_node=4 -m gptpp.train ..."}

    name      unique; also the name of the log file. Name a training job after its run (--run), so that the
              default `summary` below is the run's
    cmd       a shell command, run from the repository root with CUDA_VISIBLE_DEVICES set to the job's cards
              and $PY set to the Python that runs the queue
    gpus      number of cards the job needs (default 1)
    summary   the file whose existence means the job is done (default $GPTPP_WORK/ckpt/<name>/summary.json,
              which the trainer writes when a run completes); null means the job is never skipped

What it does:
  - A card runs one job at a time. Jobs start in file order; a job that needs more cards than are free waits,
    and the jobs behind it wait too, so a four-card job is not starved by one-card jobs.
  - A job whose summary file exists is skipped. A job that fails is recorded and the queue goes on; it is
    tried again only by a later invocation of the queue.
  - The file is read again at every poll, so jobs appended while the queue runs are picked up. With --watch
    the queue keeps waiting for new jobs when it has nothing left to do; without it, it exits.
  - Each job's output goes to $GPTPP_WORK/logs/<name>.log. Every event (started, finished, failed, skipped)
    is appended as a JSON line to the state file (default $GPTPP_WORK/logs/queue-<jobs file stem>.state.jsonl).
  - A card that another process is using (more than --busy-mib of memory in use) is left alone until it is
    free; --no-check-free turns the check off.
  - Ctrl-C once: no new jobs are started and the running ones are left to finish. Ctrl-C again (or SIGTERM):
    the running jobs are terminated. Closing the terminal (SIGHUP) is ignored, so the queue survives in tmux.

The exit status is 0 if no job failed.
"""

from __future__ import annotations

import os
import sys

# Python puts a script's directory first on the module path: take it off, so that nothing in this directory
# can shadow a library module (this file was once named queue.py, like the standard library's).
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("GPTPP_WORK", "/home/ubuntu/d-work/gpt"))


@dataclass
class Running:
    process: subprocess.Popen
    cards: list[int]
    started: float
    log: Path


@dataclass
class Queue:
    jobs_file: Path
    cards: list[int]
    state_file: Path
    check_free: bool = True
    busy_mib: int = 2000
    running: dict[str, Running] = field(default_factory=dict)
    settled: dict[str, str] = field(default_factory=dict)  # name -> finished | failed | skipped
    reported: set[str] = field(default_factory=set)  # messages already printed once
    last_busy_report: float = 0.0
    stopping: bool = False

    # ---- reading the jobs

    def read_jobs(self) -> list[dict]:
        """The jobs in file order. A line that is not yet complete JSON (being appended) is retried at the
        next poll; an unusable job is reported once and ignored."""
        jobs, names = [], set()
        try:
            lines = self.jobs_file.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            self.once(f"cannot read {self.jobs_file}: {error}")
            return []
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError:
                if number < len(lines):  # the last line may still be in the middle of being written
                    self.once(f"line {number} of {self.jobs_file.name} is not JSON; ignored")
                continue
            if not isinstance(job, dict) or not isinstance(job.get("name"), str) or not isinstance(job.get("cmd"), str):
                self.once(f"line {number} of {self.jobs_file.name} needs a \"name\" and a \"cmd\"; ignored")
                continue
            if job["name"] in names:
                self.once(f"line {number}: a second job named {job['name']!r}; ignored")
                continue
            names.add(job["name"])
            jobs.append(job)
        return jobs

    @staticmethod
    def summary_path(job: dict) -> Optional[Path]:
        if "summary" not in job:
            return WORK / "ckpt" / job["name"] / "summary.json"
        if job["summary"] is None:
            return None
        path = Path(os.path.expandvars(job["summary"]))
        return path if path.is_absolute() else REPO / path

    # ---- cards

    def free_cards(self) -> list[int]:
        taken = {card for run in self.running.values() for card in run.cards}
        free = [card for card in self.cards if card not in taken]
        if self.check_free and free:
            in_use = self.memory_in_use()
            foreign = [card for card in free if in_use.get(card, 0) > self.busy_mib]
            if foreign and time.time() - self.last_busy_report > 600:
                self.last_busy_report = time.time()
                self.say(f"card(s) {foreign} are in use by another process; leaving them alone until they are free")
            free = [card for card in free if card not in foreign]
        return free

    @staticmethod
    def memory_in_use() -> dict[int, int]:
        """MiB of memory in use on each card, from nvidia-smi; empty if it cannot be asked."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30)
            return {int(index): int(used) for index, used in
                    (line.split(",") for line in out.stdout.strip().splitlines())}
        except (OSError, ValueError, subprocess.SubprocessError):
            return {}

    # ---- events

    def once(self, message: str) -> None:
        if message not in self.reported:
            self.reported.add(message)
            self.say(message)

    @staticmethod
    def say(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def record(self, name: str, event: str, **fields: object) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "a", encoding="utf-8") as f:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(json.dumps({"time": stamp, "name": name, "event": event, **fields}) + "\n")

    # ---- running

    def start(self, job: dict, cards: list[int]) -> None:
        log = WORK / "logs" / f"{job['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, cards)), PY=sys.executable)
        try:
            with open(log, "w", encoding="utf-8") as out:
                # Its own session: the job does not receive the terminal's Ctrl-C, and can be stopped as a group.
                process = subprocess.Popen(job["cmd"], shell=True, cwd=REPO, env=env, stdout=out,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            self.settled[job["name"]] = "failed"
            self.record(job["name"], "failed", reason=f"could not be started: {error}")
            self.say(f"failed {job['name']}: could not be started: {error}")
            return
        self.running[job["name"]] = Running(process, cards, time.time(), log)
        self.record(job["name"], "started", gpus=cards, pid=process.pid, log=str(log))
        self.say(f"started {job['name']} on card(s) {cards} (pid {process.pid}, log {log})")

    def reap(self) -> None:
        """Record the jobs that have ended and free their cards."""
        for name, run in list(self.running.items()):
            code = run.process.poll()
            if code is None:
                continue
            del self.running[name]
            seconds = round(time.time() - run.started, 1)
            event = "finished" if code == 0 else "failed"
            self.settled[name] = event
            self.record(name, event, returncode=code, seconds=seconds, gpus=run.cards)
            detail = f" (exit status {code}, see {run.log})" if code else ""
            self.say(f"{event} {name} after {seconds / 60:.1f} min{detail}")

    def launch(self, jobs: list[dict]) -> None:
        """Start the pending jobs for which cards are free, in file order."""
        free = self.free_cards()
        for job in jobs:
            name = job["name"]
            if name in self.settled or name in self.running:
                continue
            summary = self.summary_path(job)
            if summary is not None and summary.exists():
                self.settled[name] = "skipped"
                self.record(name, "skipped", summary=str(summary))
                self.say(f"skipped {name}: {summary} exists")
                continue
            need = job.get("gpus", 1)
            if not isinstance(need, int) or need < 1 or need > len(self.cards):
                self.settled[name] = "failed"
                self.record(name, "failed", reason=f"needs {need!r} cards; the queue has {len(self.cards)}")
                self.say(f"failed {name}: needs {need!r} cards; the queue has {len(self.cards)}")
                continue
            if need > len(free):
                break  # keep the order: later jobs wait behind this one
            self.start(job, free[:need])
            free = free[need:]

    def stop_jobs(self) -> None:
        """Terminate the running jobs: SIGTERM to each job's process group, SIGKILL after 20 seconds."""
        for name, run in self.running.items():
            self.say(f"terminating {name}")
            try:
                os.killpg(run.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 20
        while time.time() < deadline and any(run.process.poll() is None for run in self.running.values()):
            time.sleep(0.2)
        for run in self.running.values():
            if run.process.poll() is None:
                try:
                    os.killpg(run.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                run.process.wait()
        for name, run in list(self.running.items()):
            self.settled[name] = "failed"
            self.record(name, "failed", reason="terminated with the queue", returncode=run.process.returncode,
                        seconds=round(time.time() - run.started, 1), gpus=run.cards)
        self.running.clear()

    def loop(self, poll: float, watch: bool) -> int:
        interrupts = 0

        def on_interrupt(signum: int, frame: object) -> None:
            nonlocal interrupts
            interrupts += 1 if signum == signal.SIGINT else 2

        signal.signal(signal.SIGINT, on_interrupt)
        signal.signal(signal.SIGTERM, on_interrupt)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self.say(f"queue on card(s) {self.cards}: jobs {self.jobs_file}, state {self.state_file}")
        while True:
            try:
                self.reap()
                if interrupts >= 2:
                    self.stop_jobs()
                    break
                if interrupts == 1 and not self.stopping:
                    self.stopping = True
                    self.say("no new jobs will be started; waiting for the running ones "
                             "(interrupt again to terminate them)")
                jobs = self.read_jobs()
                if not self.stopping:
                    self.launch(jobs)
                pending = [job["name"] for job in jobs
                           if job["name"] not in self.settled and job["name"] not in self.running]
                if not self.running and (self.stopping or (not pending and not watch)):
                    break
            except Exception as error:  # a full disk, a vanished file: the queue says so and keeps going
                self.once(f"queue error, will keep trying: {error!r}")
            time.sleep(poll)
        failed = sorted(name for name, event in self.settled.items() if event == "failed")
        counts = {event: sum(1 for e in self.settled.values() if e == event)
                  for event in ("finished", "skipped", "failed")}
        self.say(f"queue done: {counts['finished']} finished, {counts['skipped']} skipped, {counts['failed']} failed"
                 + (f" ({', '.join(failed)})" if failed else ""))
        return 1 if failed else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a JSON-lines list of jobs, one job per GPU.")
    parser.add_argument("jobs", help="JSON-lines file of jobs: {\"name\": ..., \"cmd\": ..., \"gpus\": 1}")
    parser.add_argument("--gpus", default="0", help="the cards to use, e.g. 0,1,2,3")
    parser.add_argument("--state", default=None,
                        help="state file (default $GPTPP_WORK/logs/queue-<jobs stem>.state.jsonl)")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between looks at the jobs file and the jobs")
    parser.add_argument("--watch", action="store_true", help="keep waiting for new jobs instead of exiting when idle")
    parser.add_argument("--no-check-free", action="store_true", help="do not ask nvidia-smi whether a card is in use")
    parser.add_argument("--busy-mib", type=int, default=2000, help="memory in use above which a card counts as taken")
    args = parser.parse_args(argv)

    jobs_file = Path(args.jobs)
    if not jobs_file.exists() and not args.watch:
        parser.error(f"{jobs_file} does not exist (with --watch the queue waits for it)")
    state = Path(args.state) if args.state else WORK / "logs" / f"queue-{jobs_file.stem}.state.jsonl"
    cards = [int(card) for card in args.gpus.split(",") if card.strip()]
    queue = Queue(jobs_file, cards, state, check_free=not args.no_check_free, busy_mib=args.busy_mib)
    return queue.loop(args.poll, args.watch)


if __name__ == "__main__":
    sys.exit(main())
