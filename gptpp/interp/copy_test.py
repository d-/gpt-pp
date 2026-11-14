"""Does a trained model copy? The loss on the second copy of random tokens, for any list of finished runs.

    python -m gptpp.interp.copy_test --runs pico grid-w128-l4-200M gptpp-base [--half 50] [--sequences 256]
                                     [--device cpu] [--out results/interp/copy-test.json]

Each sequence is `<|bos|>`, then `half` tokens drawn uniformly from the tokens of the run's own vocabulary that
occur in its training set, then the same tokens again (as in gptpp.export.loss_by_position, whose functions this
uses). The first copy cannot be predicted: a model that looks back and copies predicts the second. Reported, in
bits a token: the mean loss on each copy (the second without its first token, which nothing announces), the loss of
a uniform guess over the vocabulary, and the run's size, data and tokens seen. One model a run, its final weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from gptpp.data import TokenData
from gptpp.export.loss_by_position import drawable_tokens, load, loss_by_position, repeated_sequences

WORK = Path("/home/ubuntu/d-work/gpt")
REPO = Path(__file__).resolve().parents[2]


def measure(run: str, half: int, sequences: int, device: str, seed: int) -> dict:
    directory = WORK / "ckpt" / run
    config = json.loads((directory / "config.json").read_text())
    model_config, train_config = config["model_config"], config["train_config"]
    data = TokenData(train_config["data"])
    model, step, tokens = load(directory / "model.pt", device)
    pool = drawable_tokens(data)
    rows = repeated_sequences(pool, data.meta["bos_id"], half, sequences, seed).to(device)
    loss = loss_by_position(model, rows, 64) / math.log(2)          # entry t - 1: the token at position t
    first, second = loss[:half], loss[half + 1:]                     # the second copy without its first token
    return {
        "run": run, "data": train_config["data"], "layers": model_config["n_layer"], "width": model_config["d_model"],
        "position": model_config["pos"], "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_seen": tokens, "step": step, "half": half, "sequences": sequences,
        "first_copy_bits": round(first.mean().item(), 3), "second_copy_bits": round(second.mean().item(), 3),
        "second_copy_first_token_bits": round(loss[half].item(), 3),
        "uniform_guess_bits": round(math.log2(len(pool)), 3),
    }


def repeats(windows: int = 2048) -> dict:
    """How often held-out text continues as it did the last time (gptpp.interp.circuits.repeat_statistics), for the
    two corpora in their own vocabularies, in windows of 256 and of 1,024 tokens cut from the held-out stream."""
    import numpy as np

    from gptpp.interp import circuits

    out = {}
    for name in ("tinystories/bpe4k", "fineweb-edu/bpe32k"):
        data = TokenData(name)
        stream = np.fromfile(data.dir / "val.bin", dtype=np.uint16)
        for length in (256, 1024):
            n = min(windows, len(stream) // length)
            rows = torch.from_numpy(stream[: n * length].astype(np.int64)).view(n, length)
            r = circuits.repeat_statistics(rows, exclude=[data.meta["bos_id"]])
            out[f"{name}, windows of {length}"] = {
                "windows": n, "seen": round(r["seen"], 4), "continues": round(r["continues"], 4),
                "share_of_all_positions_induction_predicts": round(r["seen"] * r["continues"], 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--half", type=int, default=50)
    ap.add_argument("--sequences", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results" / "interp" / "copy-test.json"))
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        row = measure(run, args.half, args.sequences, args.device, args.seed)
        rows.append(row)
        print(f"{run:28s} {row['data']:22s} {row['layers']:2d} layers, {row['width']:3d} wide, {row['tokens_seen'] / 1e6:6.0f}M tokens: "
              f"first copy {row['first_copy_bits']:6.2f}, second copy {row['second_copy_bits']:6.2f} bits a token "
              f"(a uniform guess {row['uniform_guess_bits']:.2f})", flush=True)
    out = {"what": __doc__.split("\n\n")[2].replace("\n", " "), "written_by": "python -m gptpp.interp.copy_test", "runs": rows,
           "repeats_in_held_out_text": dict(
               what="seen: the share of positions whose token occurred earlier in the window; continues: among those, the "
                    "share whose next token is the one that followed the latest earlier occurrence", **repeats())}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
