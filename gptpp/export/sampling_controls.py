"""Record what the figure `gpt-sampling-controls` (fig-07-06) shows before pico's weights are loaded in the page:
`assets/data/gpt/sampling-controls.json` and `sampling-controls.bin`.

    python -m gptpp.export.sampling_controls [--out <dir>]

Nothing is computed in Python. The texts are written in Node by the page's own model (`gpt-transformer-forward-pass.js`
on `pico.json` and `pico.bin`, the 8-bit weights the page runs) through the figure module's `record`, so that the page
continues them with the same numbers; this script only stores what Node returns. Run it again whenever pico's export
(`gptpp.export.browser_model --name pico`) or the figure's presets change; then write the block again
(`node /home/ubuntu/d-work/gpt/blocks/f7-write-blocks.js --only fig-07-06`).

`sampling-controls.json`, field by field:

    model, tokenizer   "pico", "bpe4k": the files the page loads
    seed               the seed of every recorded text: the uniform drawn at position i is the same under every rule
    limit              the most tokens a text is written to (a story that ends earlier ends with the start token)
    repeat             the length of a run that counts as repeated (4 tokens)
    vocabulary, params, weights    pico's vocabulary, parameter count and the bytes of its weights file
    presets[k]         {key, rule: {temperature (0: greedy), topK (0: off), topP (1: off), minP (0: off)}}
    openings[j]        {key, text, given (tokens in the opening), tokens (the opening's token ids),
                        runs[k]: {key (the preset), tokens (the ids written after the opening), pm (the probability
                        pico gave each of them at temperature 1, four significant digits), stopped ("stop": the story
                        ended by itself; "steps": the limit was reached)}}

`sampling-controls.bin` holds one array, `sampling-controls.logits` (float32, openings x vocabulary): pico's logits for
the token after each opening, from which the page draws the distribution and applies any rule before the weights are
there. Its manifest entry is in `sampling-controls.entries.json`; the manifest is merged at the end.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import DATA, SITE, Blob, write_json

NAME = "sampling-controls"
RECORD_JS = r"""
const [site, out] = process.argv.slice(1), path = require('path');   // after `node -e`, the arguments begin at 1
const M = require(path.join(site, 'assets/js/gpt/gpt-transformer-forward-pass.js')), T = require(path.join(site, 'assets/js/gpt/gpt-byte-pair-tokenizer.js'));
const F = require(path.join(site, 'assets/js/gpt/gpt-sampling-controls-figure.js')), DEF = F['gpt-sampling-controls.defaults'];
const r = F['gpt-sampling-controls.record'](M.fromDisk(DEF.model, out), T.fromDisk(DEF.tokenizer, out));
process.stdout.write(JSON.stringify({ data: r.data, logits: Buffer.from(r.logits.buffer, r.logits.byteOffset, r.logits.byteLength).toString('base64') }));
"""


def record(out: Path) -> dict:
    """Run the figure module's `record` in Node on the exported pico; returns {"data": ..., "logits": float32 array}."""
    node = shutil.which("node") or str(Path.home() / ".local" / "bin" / "node")
    figure = SITE / "assets" / "js" / "gpt" / "gpt-sampling-controls-figure.js"
    if not Path(node).exists() or not figure.exists():
        raise SystemExit("Node or the figure module is not here: nothing recorded")
    done = subprocess.run([node, "-e", RECORD_JS, str(SITE), str(out)], capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise SystemExit(f"recording failed: {done.stderr.strip()[-2000:]}")
    raw = json.loads(done.stdout)
    data = raw["data"]
    logits = np.frombuffer(base64.b64decode(raw["logits"]), dtype="<f4").reshape(len(data["openings"]), data["vocabulary"])
    return {"data": data, "logits": logits}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DATA), help="the site's data folder (pico.json, pico.bin and bpe4k.json are read from it)")
    out = Path(parser.parse_args(argv).out)
    rec = record(out)
    data, logits = rec["data"], rec["logits"]
    blob = Blob(NAME, out)
    blob.put(f"{NAME}.logits", np.ascontiguousarray(logits, dtype=np.float32),
             "pico's logits for the token after each of the figure's openings (openings x vocabulary), float32")
    size = blob.save()
    # pm keeps four significant digits; ids are integers: nothing is lost in the short floats of write_json
    wrote = write_json(out / f"{NAME}.json", data)
    for op in data["openings"]:
        print(f"{op['text']!r}: " + ", ".join(f"{run['key']} {len(run['tokens'])} ({run['stopped']})" for run in op["runs"]))
    print(f"{NAME}.json {wrote} bytes, {NAME}.bin {size} bytes")
    merged = subprocess.run([sys.executable, "-m", "gptpp.export.manifest", "--out", str(out)], capture_output=True, text=True)
    print(merged.stdout.strip().splitlines()[-1] if merged.stdout.strip() else merged.stderr.strip())


if __name__ == "__main__":
    main()
