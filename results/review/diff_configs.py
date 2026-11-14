import json, sys, os, glob
R = "/home/ubuntu/gpt-pp/results/runs"
def load(run):
    p = f"{R}/{run}/config.json"
    if not os.path.exists(p): return None
    c = json.load(open(p))
    flat = {}
    for k, v in c["model_config"].items(): flat["m." + k] = v
    for k, v in c["train_config"].items(): flat["t." + k] = v
    d = c["derived"]
    for k in ("seq_len", "n_gpu", "tokens_per_step", "steps", "tokens", "eval_every_steps", "val_windows"):
        flat["d." + k] = d.get(k)
    for g, s in d.get("groups", {}).items():
        flat["g." + g] = f"{s['optimizer']} lr={s['lr']:g} wd={s['weight_decay']:g}"
    flat["preset"] = c["preset"]; flat["train_preset"] = c["train_preset"]
    flat["git"] = (c.get("git_commit") or "")[:7]
    flat["data"] = c["data_meta"].get("dataset", "?") + "/" + str(c["data_meta"].get("tokenizer", "?"))
    flat["params.total"] = c["params"]["total"]
    s = f"{R}/{run}/summary.json"
    flat["_val"] = json.load(open(s))["val_loss"] if os.path.exists(s) else None
    return flat
def diff(a, b, na, nb, skip=("_val", "preset", "train_preset", "d.steps", "d.tokens", "d.eval_every_steps", "params.total")):
    A, B = load(a), load(b)
    if A is None or B is None:
        print(f"## {na} -> {nb}: missing config ({a if A is None else b})"); return
    keys = sorted(set(A) | set(B))
    ds = [f"{k}: {A.get(k)!r} -> {B.get(k)!r}" for k in keys if A.get(k) != B.get(k) and k not in skip]
    va = "%.4f" % A["_val"] if A["_val"] else "n/a"; vb = "%.4f" % B["_val"] if B["_val"] else "n/a"
    print(f"## {na} ({va}) -> {nb} ({vb})  [{A['preset']}/{A['train_preset']} -> {B['preset']}/{B['train_preset']}]")
    for d in ds: print("     ", d)
runs = sorted(os.listdir(R))
what = sys.argv[1]
if what == "ladder":
    for s in ("s0", "s1"):
        rungs = sorted([r for r in runs if r.startswith("ladder-") and r.endswith(s)], key=lambda r: int(r.split("-")[1]))
        print("=====", s, rungs)
        for a, b in zip(rungs, rungs[1:]): diff(a, b, a, b)
    for r in sorted(set(x[:-3] for x in runs if x.startswith("ladder-"))):
        if f"{r}-s0" in runs and f"{r}-s1" in runs: diff(f"{r}-s0", f"{r}-s1", f"{r}-s0", f"{r}-s1")
elif what == "full":
    for r in sys.argv[2:]:
        f = load(r)
        print("=====", r); [print("   ", k, "=", v) for k, v in sorted(f.items())]
else:
    ref = sys.argv[2]
    for r in runs:
        if r.startswith(what) and r != ref: diff(ref, r, ref, r)
