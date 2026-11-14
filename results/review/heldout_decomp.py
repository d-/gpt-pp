"""Scratch check of the review: recompute the trainer's held-out loss of the main model from its checkpoint and say
where the difference from the document protocol comes from."""
import json, math, numpy as np, torch, torch.nn.functional as F
from gptpp.model import GPT
from gptpp.data import TokenData
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
model, cfg = GPT.from_checkpoint("/home/ubuntu/d-work/gpt/ckpt/gptpp-base/model.pt", device=dev)
data = TokenData("fineweb-edu/bpe32k"); val = np.asarray(data.val_tokens()); bos = data.meta["bos_id"]
T = 1024; n = 1048576 // T
tok = torch.from_numpy(val[:n * T + 1].astype(np.int64)).to(dev)
x, y = tok[:-1].view(n, T), tok[1:].view(n, T)
def run(autocast):
    out = []
    with torch.no_grad():
        for i in range(0, n, 4):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                logits = model(x[i:i + 4])
            out.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y[i:i + 4].reshape(-1), reduction="none").view(-1, T).double().cpu())
    return torch.cat(out).numpy()
nll = run(True); yb = y.cpu().numpy(); isbos = yb == bos
print(f"trainer's windows (1,024 x 1,024, bf16): mean {nll.mean():.4f} nats (summary.json: 3.0249)")
nll32 = run(False); print(f"the same in tf32: {nll32.mean():.4f}")
print(f"BOS as a target: {isbos.sum()} tokens, {nll[isbos].mean():.3f} nats each; mean without them {nll[~isbos].mean():.4f}")
pos = np.arange(T)
for lo, hi in ((0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 1024)):
    print(f"   targets at window positions {lo}..{hi-1}: {nll[:, lo:hi].mean():.3f} nats")
late = nll[:, 64:][~isbos[:, 64:]].mean(); print(f"   mean over positions >= 64, BOS targets left out: {late:.4f}")
seen = np.load("/home/ubuntu/d-work/gpt/bench/lab-review/val_blocks_in_train.npy")
mask = np.zeros(n * T, dtype=bool)
for j in seen: mask[max(j * 64 - 1, 0): j * 64 + 63] = True    # target index t is token t + 1
flat = nll.reshape(-1)
print(f"64-token blocks seen verbatim in the 3B training tokens: {mask.mean():.2%} of targets, {flat[mask].mean():.3f} nats; the rest {flat[~mask].mean():.4f}; all {flat.mean():.4f}")
nbytes = 4860438
print(f"bits a byte of these tokens: total bits {flat.sum()/math.log(2):.0f} / {nbytes} bytes = {flat.sum()/math.log(2)/nbytes:.4f}; without BOS targets {flat[~isbos.reshape(-1)].sum()/math.log(2)/nbytes:.4f}")
