"""Scratch wrapper of the review: the trainer, with the learned position table moved out of the 'embedding' group
into a group of its own at the rate given as the first argument. Nothing in the repository is edited."""
import sys
import gptpp.train as T
import gptpp.optim as O
POS_LR = float(sys.argv[1])
_orig = O.build_optimizers
def patched(model, cfg, tokens_per_step=None):
    opts = _orig(model, cfg, tokens_per_step)
    if model.pos_embed is not None:
        w = model.pos_embed.weight
        for opt in opts:
            for g in opt.param_groups:
                if g["name"] == "embedding" and any(p is w for p in g["params"]):
                    g["params"] = [p for p in g["params"] if p is not w]
                    new = {k: v for k, v in g.items() if k != "params"}
                    new.update(params=[w], name="position", base_lr=POS_LR, lr=POS_LR)
                    opt.add_param_group(new)
                    return opts
    return opts
T.build_optimizers = patched
T.main(sys.argv[2:])
