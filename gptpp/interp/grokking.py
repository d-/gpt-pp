"""Grokking on modular addition: the run, and the measures that show a circuit forming before the test loss moves.

The task of Power et al. (2022, arXiv:2201.02177) in the form that Nanda et al. took apart (2023,
arXiv:2301.05217): the model reads the three tokens `a b =` and must put (a + b) mod p on the last one. A random
30% of the p * p pairs is the training set and the rest is held out. Trained with weight decay, the model first
memorizes its training pairs (the held-out loss rises), and thousands of steps later it is right on every pair.
What it has learned by then is trigonometry: the embedding holds cos and sin of 2 pi k a / p for a handful of
frequencies k, attention and the MLP multiply them into cos and sin of 2 pi k (a + b) / p, and the unembedding
scores each c by cos(2 pi k (a + b - c) / p), which is largest at c = a + b. Addition has become rotation.

    python -m gptpp.interp.grokking --out $GPTPP_WORK/ckpt/grokking           # the recipe: p = 113, 40,000 steps
    python -m gptpp.interp.grokking --out <dir> --p 59 --weight-decay 2       # a documented shorter variant

The model is the lab's GPT with its switches set to the paper's architecture: one layer, 128 wide, 4 heads, a
ReLU MLP of 512 units, learned positions, no normalization, embedding and unembedding untied. The vocabulary is
the p residues and `=`; the logit of `=` is never read. Every matrix is redrawn from N(0, 1 / d_model): the
delay between fitting and generalizing grows with the scale of the initialization (Liu et al., 2023,
arXiv:2210.01117), and the lab's own 0.02 would shorten it. The optimizer is the paper's: AdamW on every
parameter, rate 1e-3, weight decay 1, betas (0.9, 0.98), the whole training set in every step, no schedule. The
loss is taken in float64, since a float32 softmax cannot represent a loss below about 1e-7.

What is written to --out: `config.json`, `history.json` (at each of a few hundred log-spaced steps: train and
test loss and accuracy, the norm of the weights, the power of the embedding at each frequency, restricted and
excluded loss) and `states.pt` (the model at each of those steps, from which an exporter derives the rest). On a
card shared with other jobs, `--budget 540` stops the run after nine minutes and the same command continues it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig
from ..model import GPT


# ----------------------------------------------------------------------------------------------------------------
# The task


def all_pairs(p: int) -> tuple[Tensor, Tensor]:
    """Every pair as a sequence `a b =` (p * p, 3), in the order a * p + b, and its answer (a + b) mod p."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return torch.stack((a, b, torch.full_like(a, p)), dim=1), (a + b) % p


def train_mask(p: int, fraction: float, seed: int) -> Tensor:
    """Which of the p * p pairs are trained on: floor(fraction * p * p) of them, chosen by the seed."""
    n = p * p
    chosen = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[: int(fraction * n)]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[chosen] = True
    return mask


def build_model(p: int, seed: int, d_model: int = 128, n_head: int = 4, init: str = "recipe") -> GPT:
    """The one-layer model of the paper as a configuration of the lab's GPT, initialized at the paper's scale."""
    cfg = ModelConfig(vocab_size=p + 1, n_layer=1, n_head=n_head, d_model=d_model, block_size=3, pos="learned",
                      norm="none", embed_norm=False, act="relu", mlp_mult=4.0, qk_norm=False, bias=True,
                      tie_embeddings=False, logit_softcap=0.0, init="gpt2")
    torch.manual_seed(seed)
    model = GPT(cfg)
    if init == "recipe":
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.dim() >= 2:
                    nn.init.normal_(parameter, std=1 / math.sqrt(d_model))
    return model


def answer_logits(model: GPT, tokens: Tensor, p: int) -> Tensor:
    """The logits of the p residues at the `=` position, in float64: (B, p)."""
    return model(tokens)[:, -1, :p].double()


# ----------------------------------------------------------------------------------------------------------------
# Fourier analysis over the residues


def fourier_basis(p: int) -> Tensor:
    """(p, p) float64 with orthonormal rows: the constant, then cos and sin of 2 pi k a / p for k = 1 .. (p - 1) / 2.

    Row 2k - 1 is the cosine of frequency k and row 2k its sine. For an odd p these p functions of a = 0 .. p - 1
    are an orthonormal basis: the discrete Fourier transform written with real numbers."""
    if p % 2 == 0:
        raise ValueError("the basis is written for an odd modulus")
    a = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * a / p), torch.sin(2 * math.pi * k * a / p)]
    basis = torch.stack(rows)
    return basis / basis.norm(dim=1, keepdim=True)


def embedding_power(w_e: Tensor, basis: Tensor) -> Tensor:
    """The squared norm of the embedding's component at each frequency: (1 + (p - 1) / 2,), entry 0 the constant.

    w_e (p, d) holds one row a residue. Transforming along the residues gives one d-vector per basis function;
    a frequency's power is the squared length of its cosine vector plus that of its sine vector. The entries add
    up to the squared Frobenius norm of w_e (Parseval)."""
    square = (basis.to(w_e.device) @ w_e.double()).pow(2).sum(dim=1)
    return torch.cat((square[:1], square[1::2] + square[2::2]))


def key_frequencies(power: Tensor, share: float = 0.1) -> list[int]:
    """The frequencies the trained embedding is made of: those holding at least `share` of the strongest one's
    power, strongest first. In a grokked model these are a handful out of (p - 1) / 2; which ones depends on the seed."""
    waves = power[1:]
    order = torch.argsort(waves, descending=True)
    return [int(k) + 1 for k in order if waves[k] >= share * waves.max()]


def key_directions(p: int, frequencies: list[int], basis: Tensor, products: bool = True) -> Tensor:
    """Orthonormal rows over the pairs (a, b), (4 * len(frequencies), p * p): the key-frequency terms of a
    two-dimensional Fourier transform over a and b.

    For each key frequency k these are the four products of cos or sin of 2 pi k a / p with cos or sin of
    2 pi k b / p. They are the terms that cos(w (a + b)) = cos(wa) cos(wb) - sin(wa) sin(wb) and
    sin(w (a + b)) = sin(wa) cos(wb) + cos(wa) sin(wb) are made of: five key frequencies give the 20 terms that
    Nanda et al. keep (Section 5.1). With `products=False` the rows are those two combinations only, cos and sin of
    2 pi k (a + b) / p, two a frequency: the narrower reading, kept for comparison."""
    rows = []
    for k in frequencies:
        c, s = basis[2 * k - 1], basis[2 * k]
        if products:
            rows += [torch.outer(c, c).reshape(-1), torch.outer(s, s).reshape(-1),
                     torch.outer(s, c).reshape(-1), torch.outer(c, s).reshape(-1)]
        else:
            rows.append((torch.outer(c, c) - torch.outer(s, s)).reshape(-1) / math.sqrt(2))
            rows.append((torch.outer(s, c) + torch.outer(c, s)).reshape(-1) / math.sqrt(2))
    return torch.stack(rows) if rows else torch.zeros(0, p * p, dtype=torch.float64)


MEASURES = ("restricted_test", "restricted_all", "excluded_train")


def progress_measures(logits: Tensor, answers: Tensor, train: Tensor, test: Tensor, directions: Tensor) -> Tensor:
    """The two hidden progress measures of Nanda et al. (Section 5.1) from the logits of all p * p pairs (p * p, p),
    as a (3,) tensor in the order of MEASURES.

    `train` and `test` are the indices of the training and the held-out pairs; `directions` are the key-frequency
    terms of the logits' Fourier transform over (a, b), from `key_directions`. The part of the logits that lies
    along them is what the trigonometric algorithm computes.

        restricted loss   keep only that part and the constant term (each class's mean over the pairs): the loss
                          the generalizing circuit would have on its own. The paper does not say on which pairs;
                          it is reported here on the held-out pairs and on all of them.
        excluded loss     remove exactly that part and keep the rest: on the training pairs, the loss of whatever
                          fits them without the circuit, that is, of memorization. It rises as the circuit takes over.
    """
    along = directions.T @ (directions @ logits)
    restricted = along + (logits - along).mean(dim=0, keepdim=True)
    excluded = logits - along
    return torch.stack((F.cross_entropy(restricted[test], answers[test]), F.cross_entropy(restricted, answers),
                        F.cross_entropy(excluded[train], answers[train])))


# ----------------------------------------------------------------------------------------------------------------
# The run


def log_steps(steps: int, n: int = 400) -> list[int]:
    """Step 0 and about n log-spaced steps up to the last, without repeats (the first few dozen are consecutive)."""
    spaced = np.unique(np.round(np.logspace(0, math.log10(steps), n)).astype(int))
    return [0] + [int(s) for s in spaced]


def weight_norm(model: nn.Module) -> Tensor:
    """The Euclidean norm of all the parameters together (a scalar tensor)."""
    return torch.sqrt(sum(parameter.detach().double().pow(2).sum() for parameter in model.parameters()))


SCALARS = ("train_loss", "test_loss", "train_acc", "test_acc", "weight_norm")


def run(p: int = 113, fraction: float = 0.3, steps: int = 40000, lr: float = 1e-3, weight_decay: float = 1.0,
        seed: int = 0, data_seed: int = 0, init: str = "recipe", device: str = "cuda", every: int = 2000,
        budget: float | None = None, resume: dict | None = None) -> dict:
    """Train, and measure at the log-spaced steps. Returns the configuration, the history and the saved states.

    A measurement leaves its numbers and the model's state on the device and reads nothing back, so that a card
    shared with other jobs is not waited for a few hundred times; everything is collected at the end. With a
    `budget` in seconds the run stops when it is spent and returns {"partial": ...}, which a later call takes as
    `resume` and continues from, step for step as if it had never stopped."""
    tokens, answers = all_pairs(p)
    train = train_mask(p, fraction, data_seed)
    model = build_model(p, seed, init=init).to(device)
    tokens, answers = tokens.to(device), answers.to(device)
    train_idx, test_idx = torch.where(train)[0].to(device), torch.where(~train)[0].to(device)
    x_train, y_train = tokens[train_idx], answers[train_idx]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),
                                  fused=str(device).startswith("cuda"))
    basis = fourier_basis(p).to(device)
    wanted = set(log_steps(steps))
    settings = {"p": p, "fraction": fraction, "steps": steps, "lr": lr, "weight_decay": weight_decay, "seed": seed,
                "data_seed": data_seed, "init": init}
    logged: list[int] = []
    scalars: list[Tensor] = []      # one (5,) tensor a measurement, in the order of SCALARS
    powers: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    first, before = 0, 0.0
    if resume is not None:
        if resume["settings"] != settings:
            raise ValueError(f"the partial run was made with {resume['settings']}, not {settings}")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        first, before, logged = resume["step"], resume["seconds"], list(resume["logged"])
        scalars, powers = list(resume["scalars"].to(device)), list(resume["powers"].to(device))
        states = [{name: value.to(device) for name, value in state.items()} for state in resume["states"]]
    started = time.time()
    for step in range(first, steps + 1):
        if step in wanted and (not logged or logged[-1] < step):
            with torch.no_grad():
                logits = answer_logits(model, tokens, p)
                right = (logits.argmax(dim=-1) == answers).double()
                scalars.append(torch.stack((F.cross_entropy(logits[train_idx], answers[train_idx]),
                                            F.cross_entropy(logits[test_idx], answers[test_idx]),
                                            right[train_idx].mean(), right[test_idx].mean(), weight_norm(model))))
                powers.append(embedding_power(model.embed.weight[:p], basis))
                states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
                logged.append(step)
        if step % every == 0 or step == steps:           # the latest measurement, read back from the card
            row = scalars[-1].tolist()
            print(f"step {step:6d}  at step {logged[-1]}: train {row[0]:.3e} ({row[2]:.3f})  test {row[1]:.3e} "
                  f"({row[3]:.3f})  norm {row[4]:.1f}  {before + time.time() - started:.0f} s", flush=True)
        if step == steps:
            break
        if budget is not None and step % 50 == 0 and step > first and time.time() - started > budget:
            partial = {"settings": settings, "step": step, "seconds": before + time.time() - started, "logged": logged,
                       "model": {name: value.cpu() for name, value in model.state_dict().items()},
                       "optimizer": optimizer.state_dict(), "scalars": torch.stack(scalars).cpu(),
                       "powers": torch.stack(powers).cpu(),
                       "states": [{name: value.cpu() for name, value in state.items()} for state in states]}
            return {"partial": partial}
        loss = F.cross_entropy(answer_logits(model, x_train, p), y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    table = torch.stack(scalars).cpu()
    seconds = before + time.time() - started
    history: dict[str, list] = {"step": logged}
    history.update({name: table[:, i].tolist() for i, name in enumerate(SCALARS)})
    history["embedding_power"] = torch.stack(powers).cpu().tolist()

    # The key frequencies are read off the trained embedding; the two progress measures are then taken at every
    # saved step with those frequencies, as the paper does.
    key = key_frequencies(powers[-1].cpu())
    history.update(measure_states(model, states, key, tokens, answers, train_idx, test_idx, basis))
    states = [{name: value.cpu() for name, value in state.items()} for state in states]
    config = dict(settings, n_train=int(train.sum()), n_test=int((~train).sum()), betas=[0.9, 0.98],
                  model_config=model.cfg.to_dict(), params=model.num_params(), key_frequencies=key,
                  seconds=round(seconds, 1), device=str(device))
    return {"config": config, "history": history, "states": states, "train_mask": train}


@torch.no_grad()
def measure_states(model: GPT, states: list[dict[str, Tensor]], key: list[int], tokens: Tensor, answers: Tensor,
                   train_idx: Tensor, test_idx: Tensor, basis: Tensor) -> dict[str, list]:
    """The progress measures at every saved state, with the key frequencies of the trained model: the three of
    MEASURES on the 4 terms a frequency (the paper's), and the same three with the suffix `_sum` on the two waves of
    a + b alone."""
    p, device = basis.size(0), tokens.device
    out: dict[str, list] = {}
    for suffix, products in (("", True), ("_sum", False)):
        directions = key_directions(p, key, basis.cpu(), products).to(device)
        rows = []
        for state in states:
            model.load_state_dict(state)
            rows.append(progress_measures(answer_logits(model, tokens, p), answers, train_idx, test_idx, directions))
        table = torch.stack(rows).cpu()
        out.update({name + suffix: table[:, i].tolist() for i, name in enumerate(MEASURES)})
    return out


def remeasure(out: Path, device: str) -> None:
    """Take the progress measures again from a finished run's saved states and rewrite its history.json."""
    config = json.loads((out / "config.json").read_text())
    history = json.loads((out / "history.json").read_text())
    saved = torch.load(out / "states.pt", map_location="cpu", weights_only=True)
    p = config["p"]
    tokens, answers = all_pairs(p)
    train = saved["train_mask"]
    model = build_model(p, config["seed"], init=config["init"]).to(device)
    states = [{name: value.to(device) for name, value in state.items()} for state in saved["states"]]
    history.update(measure_states(model, states, config["key_frequencies"], tokens.to(device), answers.to(device),
                                  torch.where(train)[0].to(device), torch.where(~train)[0].to(device), fourier_basis(p).to(device)))
    (out / "history.json").write_text(json.dumps(history) + "\n")
    print(f"measured {len(states)} states again with the key frequencies {config['key_frequencies']}; wrote {out / 'history.json'}")


def first_step_at(history: dict, name: str, level: float) -> int | None:
    """The first logged step from which `name` stays at or above `level` to the end of the run."""
    values, steps = history[name], history["step"]
    for i in range(len(values)):
        if all(v >= level for v in values[i:]):
            return steps[i]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the run's directory")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--init", choices=("recipe", "lab"), default="recipe",
                    help="recipe: every matrix N(0, 1/d_model); lab: the GPT-2 initialization of the lab's model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds after which the run stops and writes partial.pt; the same command continues it")
    ap.add_argument("--remeasure", action="store_true",
                    help="no training: take the progress measures again from the saved states of a finished run")
    args = ap.parse_args()
    torch.set_num_threads(4)                              # the work is on the card; a busy machine has no cores to spare
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.remeasure:
        remeasure(out, args.device)
        return
    partial = out / "partial.pt"
    resume = torch.load(partial, map_location="cpu", weights_only=True) if partial.exists() else None
    if resume is not None:
        print(f"continuing from step {resume['step']} ({resume['seconds']:.0f} s so far)")
    result = run(args.p, args.fraction, args.steps, args.lr, args.weight_decay, args.seed, args.data_seed, args.init,
                 args.device, budget=args.budget, resume=resume)
    if "partial" in result:
        torch.save(result["partial"], partial)
        print(f"stopped at step {result['partial']['step']} of {args.steps}: the budget of {args.budget:.0f} s is spent; "
              f"run the same command again to continue")
        raise SystemExit(3)
    history, config = result["history"], result["config"]
    config["train_acc_99_step"] = first_step_at(history, "train_acc", 0.99)
    config["test_acc_99_step"] = first_step_at(history, "test_acc", 0.99)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history) + "\n")
    torch.save({"states": result["states"], "steps": history["step"], "train_mask": result["train_mask"],
                "model_config": config["model_config"]}, out / "states.pt")
    partial.unlink(missing_ok=True)
    print(f"key frequencies {config['key_frequencies']}; train accuracy 99% from step {config['train_acc_99_step']}, "
          f"test accuracy 99% from step {config['test_acc_99_step']}; {config['seconds']} s; wrote {out}")


if __name__ == "__main__":
    main()
