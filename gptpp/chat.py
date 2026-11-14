"""The chat stage: a model that continues text is taught to answer.

    python -m gptpp.chat train --base gptpp-base --run gptpp-chat [--data chat/bpe32k] [--lr-scale 0.2] \
        [--epochs 1 | --tokens 5e6] [--batch 32] [--grad-accum 1] [--eval-tokens 5e6] [--seed 0] [--overwrite]
    python -m gptpp.chat --run gptpp-chat                                  # talk to it
    python -m gptpp.chat --run gptpp-chat --prompt "What is a prime number?" [--temperature 0.7] [--seed 0]

A pretrained model has one skill: given a text, continue it. Asked "What is a prime number?", it continues the
page the question might have stood on: more questions, a heading, an essay. The chat stage changes what follows
a question, not the model. A conversation is written as one token sequence with five tokens that the tokenizer
reserved and that never occur in the pretraining text,

    <|bos|> <|user_start|> What is a prime number? <|user_end|>
            <|assistant_start|> A whole number greater than one whose only divisors ... <|assistant_end|>
            <|user_start|> ...

and the pretrained model goes on training on such sequences with the same next-token loss, counted only on the
tokens the assistant writes: its text and its <|assistant_end|>. The user's text, <|bos|> and the three other
markers are context; the model reads them and is never asked to predict them, so it does not learn to write the
user's side. <|assistant_end|> carries the loss because that is how the model learns to stop. This is supervised
fine-tuning on instructions (Wei et al., "Finetuned language models are zero-shot learners", 2021,
arXiv:2109.01652; Ouyang et al., 2022, arXiv:2203.02155, where it is the first of three stages); the format and
the masked loss follow nanochat's, the data is a subset of SmolTalk (scripts/prepare_chat.py).

What this module holds:

    render, prompt_ids, parse     the format: messages to tokens with the loss mask, and back
    ChatData, pack, ChatLoader    conversations on disk; whole conversations packed into rows of the context length
    evaluate                      held-out loss on the tokens that carry it, the text and the end token apart
    finetune                      the loop: the lab's optimizer groups at reduced rates, bfloat16, masked loss
    reply, answer                 generation with the KV cache until <|assistant_end|>

Rows. A row has seq_len + 1 tokens and holds whole conversations, each begun by <|bos|>; the room left over is
padding that carries no loss. No conversation is cut (the data holds none longer than the context), and every
conversation is read from its first token, as it will be when the model is used. Conversations are placed by
best fit from a buffer of a thousand, so little of a row is padding. A row's later conversations can see its
earlier ones, as documents could in pretraining.

Rates. The optimizers and parameter groups are those of pretraining (Muon on the hidden matrices, AdamW on the
embedding and the unembedding; gptpp/optim.py), every peak rate multiplied by `--lr-scale`, with a short warmup
(the optimizer state is new: a checkpoint holds only the weights) and a linear decay to `--final-lr-frac` of the
peak. The embeddings and unembedding rows of the five special tokens start untrained and learn at the same rates.

What a run writes in $GPTPP_WORK/ckpt/<run>/ (and small copies in <repo>/results/runs/<run>/):

    config.json    the base checkpoint, the model and stage configuration, the data's meta, the groups' rates
    log.jsonl      {"kind": "train", "step", "tokens", "time", "loss", "lr", "lr_mult", "tok_s", "loss_tokens"}
                   {"kind": "val", "step", "tokens", "time", "val_loss", "val_loss_text", "val_loss_end",
                    "plain_loss"}     held-out conversations (all tokens that carry the loss; the assistant's
                                      text; its end token) and held-out plain text from the pretraining data
                   {"kind": "sample", "step", "tokens", "prompt", "text", "stopped"}   a greedy answer
    model.pt       {"model": state_dict, "model_config": dict, "step", "tokens"}, as the trainer writes it
    summary.json   the losses before and after, tokens, seconds, throughput, memory

`tokens` counts the tokens the model read (rows times seq_len, padding included); `loss_tokens` those that
carried the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import signal
import sys
import time
from bisect import bisect_right, insort
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import repo_dir, work_dir
from .config import TrainConfig
from .data import TOKEN_DTYPE, TokenData
from .generate import load_tokenizer, sample
from .model import GPT, KVCache
from .optim import build_optimizers, lr_multiplier, set_lr

BOS, USER_START, USER_END = "<|bos|>", "<|user_start|>", "<|user_end|>"
ASSISTANT_START, ASSISTANT_END = "<|assistant_start|>", "<|assistant_end|>"

Message = dict  # {"role": "user" | "assistant" | "system", "content": str}


# ----------------------------------------------------------------------------------------------------------------
# The format


def turns(messages: Sequence[Message]) -> list[Message]:
    """The messages as alternating turns, the user first. The format has no marker for a system message: a
    leading one is put in front of the first user message, a blank line between them."""
    messages = list(messages)
    if messages and messages[0]["role"] == "system":
        if len(messages) < 2 or messages[1]["role"] != "user":
            raise ValueError("a system message must be followed by a user message")
        merged = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = [{"role": "user", "content": merged}, *messages[2:]]
    for i, message in enumerate(messages):
        if message["role"] != ("user", "assistant")[i % 2]:
            raise ValueError(f"message {i} is by {message['role']!r}: turns alternate, the user first")
    return messages


def assemble(roles: Sequence[str], pieces: Sequence[Sequence[int]], special: dict[str, int]) -> tuple[list[int], list[int]]:
    """Token ids and loss mask of a conversation whose messages are already encoded (`pieces`, one a message).

    The mask is 1 on the tokens that carry the loss: the assistant's text and its <|assistant_end|>. It is 0 on
    <|bos|>, on the user's turn and on <|assistant_start|>, which the model is given, never asked to produce.
    """
    ids, mask = [special[BOS]], [0]
    for i, (role, piece) in enumerate(zip(roles, pieces)):
        if role != ("user", "assistant")[i % 2]:
            raise ValueError(f"message {i} is by {role!r}: turns alternate, the user first")
        if role == "user":
            ids += [special[USER_START], *piece, special[USER_END]]
            mask += [0] * (len(piece) + 2)
        else:
            ids += [special[ASSISTANT_START], *piece, special[ASSISTANT_END]]
            mask += [0] + [1] * (len(piece) + 1)
    return ids, mask


def render(tokenizer, messages: Sequence[Message]) -> tuple[list[int], list[int]]:
    """A conversation as the model sees it: (token ids, loss mask), see `assemble`."""
    messages = turns(messages)
    pieces = [tokenizer.encode(message["content"]) for message in messages]
    return assemble([message["role"] for message in messages], pieces, tokenizer.special)


def prompt_ids(tokenizer, messages: Sequence[Message]) -> list[int]:
    """The tokens after which the assistant answers: the conversation, which ends with the user's message,
    then <|assistant_start|>."""
    messages = turns(messages)
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("the conversation must end with the user's message")
    return render(tokenizer, messages)[0] + [tokenizer.special[ASSISTANT_START]]


def parse(tokenizer, ids: Sequence[int]) -> list[Message]:
    """The inverse of `render`: the messages of a token sequence. An unfinished last turn is returned as far as
    it goes; an empty one (a prompt that ends with <|assistant_start|>) is left out."""
    name = {i: marker for marker, i in tokenizer.special.items()}
    ids = list(ids)
    if not ids or name.get(ids[0]) != BOS:
        raise ValueError("a conversation begins with <|bos|>")
    messages: list[Message] = []
    role: Optional[str] = None
    piece: list[int] = []
    for i in ids[1:]:
        marker = name.get(i)
        if marker is None and role is not None:
            piece.append(i)
        elif marker in (USER_START, ASSISTANT_START) and role is None:
            role, piece = ("user" if marker == USER_START else "assistant"), []
        elif (marker, role) in ((USER_END, "user"), (ASSISTANT_END, "assistant")):
            messages.append({"role": role, "content": tokenizer.decode(piece)})
            role = None
        else:
            raise ValueError(f"{marker or 'text'} out of place after {len(messages)} messages")
    if role is not None and piece:
        messages.append({"role": role, "content": tokenizer.decode(piece)})
    return messages


# ----------------------------------------------------------------------------------------------------------------
# Conversations on disk, and rows of them


class ChatData(TokenData):
    """Conversations on disk: the token shards of Contract 1 (data.py), each conversation begun by <|bos|>, and
    beside every token file a mask file, one byte a token, 1 where the token carries the loss."""

    @staticmethod
    def mask_file(path: Path) -> Path:
        return path.with_name(path.name.replace(".bin", ".mask.bin"))

    @property
    def bos_id(self) -> int:
        return int(self.meta["bos_id"])

    def shard_mask(self, i: int) -> np.memmap:
        return np.memmap(self.mask_file(self.train_files[i]), dtype=np.uint8, mode="r")

    def val_mask(self) -> np.memmap:
        return np.memmap(self.mask_file(self.val_file), dtype=np.uint8, mode="r")


def conversations(tokens: np.ndarray, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Where each conversation of a token stream starts, and how long it is: every conversation begins with
    <|bos|>, which occurs nowhere else."""
    starts = np.flatnonzero(np.asarray(tokens) == bos_id)
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError("the stream does not begin with <|bos|>")
    return starts, np.diff(np.append(starts, len(tokens)))


def pack(lengths: Sequence[int], capacity: int, order: Optional[Sequence[int]] = None,
         buffer: int = 1000) -> list[list[int]]:
    """Pack conversations into rows of at most `capacity` tokens; returns the conversations of each row.

    Conversations enter a buffer in the order given. A row is filled by best fit: again and again the longest
    buffered conversation that still fits, until none fits. Every conversation is placed whole and exactly once;
    one longer than a row gets a row to itself (and is cut there by `rows_of`). This is the row-packing rule of
    nanochat's pretraining loader with padding in place of cropping.
    """
    order = range(len(lengths)) if order is None else order
    waiting: list[tuple[int, int]] = []  # (tokens, conversation), sorted
    rows: list[list[int]] = []
    cursor, n = 0, len(order)
    while cursor < n or waiting:
        row: list[int] = []
        space = capacity
        while True:
            while len(waiting) < buffer and cursor < n:
                c = int(order[cursor])
                insort(waiting, (min(int(lengths[c]), capacity), c))
                cursor += 1
            k = bisect_right(waiting, (space, math.inf)) - 1  # the longest conversation of at most `space` tokens
            if k < 0:
                break
            length, c = waiting.pop(k)
            row.append(c)
            space -= length
        rows.append(row)
    return rows


def rows_of(tokens: np.ndarray, mask: np.ndarray, starts: np.ndarray, lengths: np.ndarray, rows: list[list[int]],
            capacity: int, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows as arrays (len(rows), capacity): token ids, and the loss mask. The room after a row's last
    conversation is filled with `pad_id` and carries no loss."""
    ids = np.full((len(rows), capacity), pad_id, dtype=TOKEN_DTYPE)
    carries = np.zeros((len(rows), capacity), dtype=np.uint8)
    for r, row in enumerate(rows):
        at = 0
        for c in row:
            start, n = int(starts[c]), min(int(lengths[c]), capacity - at)
            ids[r, at:at + n] = tokens[start:start + n]
            carries[r, at:at + n] = mask[start:start + n]
            at += n
    return ids, carries


def inputs_and_targets(ids: np.ndarray, carries: np.ndarray, device: Union[str, torch.device]) -> tuple[Tensor, Tensor]:
    """Rows of T + 1 tokens as inputs x (B, T) and targets y (B, T): y is the next token where that token
    carries the loss and -1, which the loss ignores, everywhere else."""
    block = torch.from_numpy(ids.astype(np.int64))
    counted = torch.from_numpy(carries[:, 1:].astype(bool))
    x, y = block[:, :-1], torch.where(counted, block[:, 1:], -1)
    return x.contiguous().to(device), y.contiguous().to(device)


class ChatLoader:
    """Training batches (batch, seq_len) of inputs and masked targets.

    Each shard is read once a pass: its conversations are put in an order drawn from the seed, the pass and the
    shard, and packed into rows; the rows are served in that order, the incomplete last batch dropped.
    """

    def __init__(self, data: ChatData, seq_len: int, batch: int, seed: int = 0,
                 device: Union[str, torch.device] = "cpu", buffer: int = 1000):
        self.data, self.seq_len, self.batch, self.seed = data, seq_len, batch, seed
        self.device, self.buffer = torch.device(device), buffer
        self.epoch, self.shard_index, self.cursor = 0, -1, 0
        self.ids = self.carries = np.empty((0, seq_len + 1))
        self._next_shard()

    def _plan(self, shard_index: int, epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
        tokens, mask = self.data.shard(shard_index), self.data.shard_mask(shard_index)
        starts, lengths = conversations(tokens, self.data.bos_id)
        order = np.random.default_rng([self.seed, epoch, shard_index]).permutation(len(starts))
        return tokens, mask, starts, lengths, pack(lengths, self.seq_len + 1, order, self.buffer)

    def _next_shard(self) -> None:
        for _ in range(len(self.data.train_files) + 1):
            self.shard_index += 1
            if self.shard_index == len(self.data.train_files):
                self.shard_index, self.epoch = 0, self.epoch + 1
            tokens, mask, starts, lengths, rows = self._plan(self.shard_index, self.epoch)
            if len(rows) >= self.batch:
                self.ids, self.carries = rows_of(tokens, mask, starts, lengths, rows, self.seq_len + 1, self.data.bos_id)
                self.cursor = 0
                return
        raise ValueError(f"{self.data.dir}: no shard holds {self.batch} rows of {self.seq_len} tokens")

    def rows_per_pass(self) -> int:
        """Rows served in one pass over the data (the first pass; later ones differ by a few rows)."""
        total = 0
        for i in range(len(self.data.train_files)):
            n = len(self.ids) if i == self.shard_index and self.epoch == 0 else len(self._plan(i, 0)[-1])
            total += n // self.batch * self.batch
        return total

    def next_batch(self) -> tuple[Tensor, Tensor]:
        if self.cursor + self.batch > len(self.ids):
            self._next_shard()
        rows = slice(self.cursor, self.cursor + self.batch)
        self.cursor += self.batch
        return inputs_and_targets(self.ids[rows], self.carries[rows], self.device)


def val_batches(data: ChatData, seq_len: int, batch: int, device: Union[str, torch.device] = "cpu",
                keep: Optional[Sequence[int]] = None) -> list[tuple[Tensor, Tensor]]:
    """The held-out conversations (all, or those listed in `keep`) packed into rows in their order on disk, as
    batches of inputs and masked targets on `device`. The same rows whatever the batch size."""
    tokens, mask = data.val_tokens(), data.val_mask()
    starts, lengths = conversations(tokens, data.bos_id)
    order = range(len(starts)) if keep is None else list(keep)
    rows = pack(lengths, seq_len + 1, order)
    ids, carries = rows_of(tokens, mask, starts, lengths, rows, seq_len + 1, data.bos_id)
    return [inputs_and_targets(ids[i:i + batch], carries[i:i + batch], device) for i in range(0, len(ids), batch)]


def plain_batches(name: Union[str, Path], seq_len: int, batch: int, n_windows: int,
                  device: Union[str, torch.device] = "cpu") -> list[tuple[Tensor, Tensor]]:
    """The first windows of a plain-text dataset's held-out tokens (the trainer's own held-out windows), every
    token a target: what the stage does to the model's first skill."""
    tokens = TokenData(name).val_tokens()
    n_windows = min(n_windows, (len(tokens) - 1) // seq_len)
    block = torch.from_numpy(np.stack([tokens[i * seq_len:(i + 1) * seq_len + 1] for i in range(n_windows)]).astype(np.int64))
    return [(block[i:i + batch, :-1].contiguous().to(device), block[i:i + batch, 1:].contiguous().to(device))
            for i in range(0, n_windows, batch)]


# ----------------------------------------------------------------------------------------------------------------
# Held-out loss


@torch.no_grad()
def evaluate(model: GPT, batches: Sequence[tuple[Tensor, Tensor]], end_id: Optional[int] = None) -> dict:
    """Mean cross-entropy in nats over the targets that count (y >= 0), in bfloat16 autocast on a card.

    With `end_id`, the mean is also given apart for the assistant's text and for its end token: before the stage
    the model has never been asked for the end token, and what it pays there says so.
    """
    was_training = model.training
    model.eval()
    device = model.embed.weight.device
    total = torch.zeros(4, dtype=torch.float64, device=device)  # loss and count: all targets, end tokens
    for x, y in batches:
        counted = y >= 0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        losses = F.cross_entropy(logits[counted].float(), y[counted], reduction="none").double()
        total[0] += losses.sum()
        total[1] += losses.numel()
        if end_id is not None:
            is_end = y[counted] == end_id
            total[2] += losses[is_end].sum()
            total[3] += is_end.sum()
    model.train(was_training)
    loss, n, loss_end, n_end = total.tolist()
    result = {"loss": loss / n, "tokens": int(n)}
    if end_id is not None:
        result.update(loss_text=(loss - loss_end) / max(n - n_end, 1), loss_end=loss_end / max(n_end, 1),
                      end_tokens=int(n_end))
    return result


# ----------------------------------------------------------------------------------------------------------------
# Generation


@torch.no_grad()
def reply(model: GPT, prompt: Sequence[int], stop: dict[int, str], max_new_tokens: int = 256,
          temperature: float = 0.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
          min_p: Optional[float] = None, seed: Optional[int] = None,
          on_token: Optional[Callable[[int], None]] = None) -> tuple[list[int], str]:
    """The tokens the model writes after `prompt`, and why it stopped.

    The prompt is run once and fills the KV cache; each further token costs one pass over one token. Generation
    ends when the model writes a token of `stop` (ids to names; the reason returned is that token's name), or
    after `max_new_tokens`, or when the context is full (the reason is then "length"). Temperature 0 takes the
    most likely token; otherwise `seed` fixes the draw. `on_token` is called with each token as it is written.
    """
    window = model.cfg.block_size
    if len(prompt) >= window:
        raise ValueError(f"the prompt has {len(prompt)} tokens and the context {window}: nothing can be written")
    was_training = model.training
    model.eval()
    device = model.embed.weight.device
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    kv = KVCache(model.cfg.n_layer)
    written: list[int] = []
    stopped = "length"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model.forward_explicit(torch.tensor([list(prompt)], dtype=torch.long, device=device), kv=kv)[:, -1]
        budget = min(max_new_tokens, window - len(prompt))
        for n in range(budget):
            token = sample(logits, temperature, top_k, top_p, min_p, generator)
            if int(token) in stop:
                stopped = stop[int(token)]
                break
            written.append(int(token))
            if on_token is not None:
                on_token(int(token))
            if n + 1 < budget:
                logits = model.forward_explicit(token, kv=kv)[:, -1]
    model.train(was_training)
    return written, stopped


def answer(model: GPT, tokenizer, messages: Sequence[Message], **sampling) -> tuple[str, dict]:
    """The assistant's answer to a conversation that ends with the user's message: the text, and
    {"tokens": how many, "stopped": "end" (it wrote <|assistant_end|>), "length", or another marker's name}."""
    stop = {i: name for name, i in tokenizer.special.items()}
    written, stopped = reply(model, prompt_ids(tokenizer, messages), stop, **sampling)
    return tokenizer.decode(written), {"tokens": len(written), "stopped": "end" if stopped == ASSISTANT_END else stopped}


# ----------------------------------------------------------------------------------------------------------------
# The stage


def checkpoint_path(spec: str, ckpt: str = "model.pt") -> Path:
    """A checkpoint by run name (under $GPTPP_WORK/ckpt), by run directory, or by file."""
    path = Path(spec)
    if path.is_file():
        return path
    return (path if path.is_dir() else work_dir() / "ckpt" / spec) / ckpt


def stage_config(base: TrainConfig, args: argparse.Namespace, seq_len: int, tokens: int) -> TrainConfig:
    """The stage's training configuration: the base's optimizer and groups, every peak rate times lr_scale, a
    short warmup and a linear decay."""
    def scaled(rate: Optional[float]) -> Optional[float]:
        return None if rate is None else rate * args.lr_scale

    return base.replace(
        data=str(args.data), tokens=float(tokens), seq_len=seq_len, device_batch=args.batch, grad_accum=args.grad_accum,
        lr=base.lr * args.lr_scale, lr_embedding=scaled(base.lr_embedding), lr_unembedding=scaled(base.lr_unembedding),
        lr_gains=scaled(base.lr_gains), schedule="linear", warmup_tokens=float(args.warmup_tokens),
        final_lr_frac=args.final_lr_frac, eval_tokens=float(args.eval_tokens), seed=args.seed,
        compile=not args.no_compile)


def finetune(args: argparse.Namespace) -> dict:
    """Run the stage described by the arguments; returns the summary."""
    from .train import RunLog, _interrupt, copy_results, git_commit, save_checkpoint

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    on_gpu = device.type == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    # ---- the base model, the data, the budget
    base_file = checkpoint_path(args.base, args.ckpt)
    saved = torch.load(base_file, map_location="cpu", weights_only=True)
    base = {"checkpoint": f"{base_file.parent.name}/{base_file.name}", "step": saved.get("step"),
            "tokens": saved.get("tokens")}
    del saved
    model, model_cfg = GPT.from_checkpoint(base_file, device)
    model.train()
    base_config = base_file.parent / "config.json"
    base_train = (TrainConfig.from_dict(json.loads(base_config.read_text())["train_config"]) if base_config.exists()
                  else TrainConfig())  # a checkpoint without its run's config: today's defaults

    data = ChatData(args.data)
    if data.vocab_size != model_cfg.vocab_size:
        raise SystemExit(f"{args.data} has a vocabulary of {data.vocab_size}, the model of {model_cfg.vocab_size}")
    tokenizer = load_tokenizer(data.meta["tokenizer"])
    seq_len = model_cfg.block_size
    loader = ChatLoader(data, seq_len, args.batch, args.seed, device)
    tokens_per_step = args.batch * seq_len * args.grad_accum
    if args.tokens is not None:
        n_steps = max(1, round(args.tokens / tokens_per_step))
    else:
        n_steps = max(1, int(args.epochs * loader.rows_per_pass()) // (args.batch * args.grad_accum))
    total_tokens = n_steps * tokens_per_step
    train_cfg = stage_config(base_train, args, seq_len, total_tokens)
    eval_every = max(1, round(train_cfg.eval_tokens / tokens_per_step))
    eval_steps = set(range(0, n_steps, eval_every)) | {n_steps}

    held_out = val_batches(data, seq_len, args.eval_batch, device)
    plain = (plain_batches(args.plain_data, seq_len, args.eval_batch, args.plain_windows, device)
             if args.plain_data != "none" else [])
    end_id = tokenizer.special[ASSISTANT_END] if tokenizer is not None else None

    run_dir = Path(args.out) / args.run if args.out else work_dir() / "ckpt" / args.run
    if (run_dir / "summary.json").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} holds a completed run; pass --overwrite to replace it")
    run_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("summary.json", "model.pt", "interrupted_*.pt"):  # of an earlier attempt
        for stale in run_dir.glob(pattern):
            stale.unlink()

    optimizers = build_optimizers(model, train_cfg, tokens_per_step)
    compiled = torch.compile(model) if train_cfg.compile else model
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu)
    params = model.num_params()
    groups = {group["name"]: {"optimizer": type(optimizer).__name__, "lr": group["base_lr"],
                              "weight_decay": group["weight_decay"]}
              for optimizer in optimizers for group in optimizer.param_groups}
    config = {
        "run": args.run, "stage": "chat", "base": base, "argv": sys.argv, "lr_scale": args.lr_scale,
        "model_config": model_cfg.to_dict(), "train_config": train_cfg.to_dict(), "data_meta": data.meta,
        "params": params,
        "derived": {"seq_len": seq_len, "tokens_per_step": tokens_per_step, "steps": n_steps, "tokens": total_tokens,
                    "rows_per_pass": loader.rows_per_pass(), "passes": total_tokens / (loader.rows_per_pass() * seq_len),
                    "eval_every_steps": eval_every, "held_out_rows": sum(len(x) for x, _ in held_out),
                    "plain_windows": sum(len(x) for x, _ in plain), "plain_data": args.plain_data, "groups": groups},
        "git_commit": git_commit(),
        "hardware": {"gpu": torch.cuda.get_device_name(device) if on_gpu else "cpu", "torch": torch.__version__,
                     "cuda": torch.version.cuda, "python": platform.python_version()},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = RunLog(run_dir / "log.jsonl")
    print(f"{args.run}: {base['checkpoint']} (step {base['step']}), {params['total'] / 1e6:.1f}M parameters; "
          f"{n_steps} steps of {tokens_per_step:,} tokens ({args.batch} x {seq_len}"
          f"{' x ' + str(args.grad_accum) if args.grad_accum > 1 else ''}) = {total_tokens / 1e6:.1f}M tokens of "
          f"{args.data}, {config['derived']['passes']:.2f} passes; rates x{args.lr_scale}: "
          + ", ".join(f"{name} {group['lr']:.3g}" for name, group in groups.items())
          + f"; evaluation every {eval_every} steps on {config['derived']['held_out_rows']} rows", flush=True)

    # ---- the loop
    previous_handler = signal.signal(signal.SIGTERM, _interrupt)  # a kill is handled like Ctrl-C
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    step = 0
    start = time.perf_counter()
    interval_start, interval_steps = start, 0
    loss_sum = torch.zeros((), device=device)
    loss_tokens = torch.zeros((), dtype=torch.int64, device=device)
    train_seconds, first_interval_seconds, first_interval_steps = 0.0, 0.0, 0
    eval_durations: list[float] = []
    evaluations: list[dict] = []
    train_loss = float("nan")
    try:
        for step in range(n_steps + 1):
            tokens = step * tokens_per_step

            if step in eval_steps:
                if on_gpu:
                    torch.cuda.synchronize(device)
                pause = time.perf_counter()
                chat = evaluate(model, held_out, end_id)
                record = {"step": step, "tokens": tokens, "time": round(pause - start, 3), "val_loss": chat["loss"],
                          "val_loss_text": chat.get("loss_text"), "val_loss_end": chat.get("loss_end"),
                          "plain_loss": evaluate(model, plain)["loss"] if plain else None}
                evaluations.append(record)
                log.write("val", **record)
                line = (f"eval  step {step:>6} | tokens {tokens / 1e6:8.2f}M | val_loss {chat['loss']:.4f}"
                        + (f" (text {chat['loss_text']:.4f}, end {chat['loss_end']:.4f})" if end_id is not None else "")
                        + (f" | plain {record['plain_loss']:.4f}" if plain else ""))
                if tokenizer is not None and args.sample_prompt:
                    text, info = answer(model, tokenizer, [{"role": "user", "content": args.sample_prompt}],
                                        max_new_tokens=args.sample_tokens)
                    log.write("sample", step=step, tokens=tokens, prompt=args.sample_prompt, text=text,
                              stopped=info["stopped"])
                eval_durations.append(time.perf_counter() - pause)
                print(line + f" | {eval_durations[-1]:.1f} s", flush=True)
                interval_start += time.perf_counter() - pause

            if step == n_steps:
                break

            multiplier = lr_multiplier((step + 0.5) * tokens_per_step, total_tokens, train_cfg)
            set_lr(optimizers, multiplier)
            for _ in range(args.grad_accum):
                x, y = loader.next_batch()
                with autocast:
                    loss = compiled(x, y)
                (loss / args.grad_accum).backward()
                loss_sum += loss.detach() / args.grad_accum
                loss_tokens += (y >= 0).sum()
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            interval_steps += 1

            if (step + 1) % args.log_every == 0 or step + 1 == n_steps:
                train_loss = loss_sum.item() / interval_steps  # .item() waits for the card
                now = time.perf_counter()
                seconds, interval_start = now - interval_start, now
                train_seconds += seconds
                if first_interval_steps == 0:  # the first interval pays for compilation
                    first_interval_seconds, first_interval_steps = seconds, interval_steps
                tok_s = interval_steps * tokens_per_step / seconds
                done = (step + 1) * tokens_per_step
                matrix_lr = next(g["lr"] for o in optimizers for g in o.param_groups if g["name"] == "matrices")
                log.write("train", step=step + 1, tokens=done, time=round(now - start, 3), loss=train_loss,
                          lr=matrix_lr, lr_mult=multiplier, tok_s=round(tok_s), loss_tokens=int(loss_tokens.item()))
                print(f"step {step + 1:>6}/{n_steps} | tokens {done / 1e6:8.2f}M | loss {train_loss:.4f} | "
                      f"lr x{multiplier:.3f} | {tok_s / 1e3:6.1f}k tok/s | "
                      f"{(total_tokens - done) / tok_s / 60:5.1f} min left", flush=True)
                if not math.isfinite(train_loss):
                    raise FloatingPointError(f"the training loss is {train_loss} at step {step + 1}")
                loss_sum.zero_()
                interval_steps = 0
    except BaseException as error:
        log.write("error", step=step, tokens=step * tokens_per_step, error=f"{type(error).__name__}: {error}")
        log.close()
        if isinstance(error, KeyboardInterrupt) and step > 0 and not args.no_model:
            save_checkpoint(run_dir / f"interrupted_{step}.pt", model, step, step * tokens_per_step)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    # ---- the end
    if not args.no_model:
        save_checkpoint(run_dir / "model.pt", model, n_steps, total_tokens)
    seconds = time.perf_counter() - start
    steady_steps = n_steps - first_interval_steps
    step_seconds = ((train_seconds - first_interval_seconds) / steady_steps if steady_steps > 0
                    else train_seconds / n_steps)
    eval_seconds = float(np.median(eval_durations[1:])) if len(eval_durations) > 1 else eval_durations[0]
    before, after = evaluations[0], evaluations[-1]
    summary = {
        "run": args.run, "status": "completed", "stage": "chat", "base": base, "lr_scale": args.lr_scale,
        "val_loss": after["val_loss"], "val_loss_text": after["val_loss_text"], "val_loss_end": after["val_loss_end"],
        "plain_loss": after["plain_loss"],
        "before": {key: before[key] for key in ("val_loss", "val_loss_text", "val_loss_end", "plain_loss")},
        "train_loss": train_loss, "steps": n_steps, "tokens": total_tokens, "loss_tokens": int(loss_tokens.item()),
        "passes": config["derived"]["passes"], "seconds": round(seconds, 1), "train_seconds": round(train_seconds, 1),
        "tokens_per_second": round(tokens_per_step / step_seconds), "step_seconds": round(step_seconds, 4),
        "eval_seconds": round(eval_seconds, 2),
        "compile_seconds": round(max(0.0, first_interval_seconds - first_interval_steps * step_seconds), 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if on_gpu else None,
        "params": params,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.close()
    if args.results != "none":
        copy_results(run_dir, (Path(args.results) if args.results else repo_dir() / "results" / "runs") / args.run)
    print(f"done: held-out loss {before['val_loss']:.4f} -> {after['val_loss']:.4f}"
          + (f", plain text {before['plain_loss']:.4f} -> {after['plain_loss']:.4f}" if plain else "")
          + f", {summary['tokens_per_second']:,} tok/s, {seconds / 60:.1f} min"
          + (f", peak memory {summary['peak_memory_gb']} GB" if on_gpu else "") + f", results in {run_dir}", flush=True)
    return summary


# ----------------------------------------------------------------------------------------------------------------
# The command line


def load_run(run: str, ckpt: str = "model.pt", device: Optional[str] = None, tokenizer: Optional[str] = None):
    """The model of a run and the tokenizer of its data: (model, tokenizer)."""
    from .generate import run_tokenizer_name

    path = checkpoint_path(run, ckpt)
    model, _ = GPT.from_checkpoint(path, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = tokenizer or run_tokenizer_name(path.parent)
    loaded = load_tokenizer(name) if name else None
    if loaded is None:
        raise SystemExit(f"no tokenizer could be loaded for {path} ({name!r}): pass --tokenizer")
    return model, loaded


def talk(args: argparse.Namespace) -> None:
    """One answer to --prompt, or a conversation at the terminal."""
    model, tokenizer = load_run(args.run, args.ckpt, args.device, args.tokenizer)
    sampling = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, min_p=args.min_p, seed=args.seed)
    if args.prompt is not None:
        text, info = answer(model, tokenizer, [{"role": "user", "content": args.prompt}], **sampling)
        print(text + ("" if info["stopped"] == "end" else f"  [{info['stopped']}]"))
        return

    print("An empty line or /quit ends the conversation; /new starts another.")
    stop = {i: name for name, i in tokenizer.special.items()}
    messages: list[Message] = []
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if line in ("", "/quit"):
            break
        if line == "/new":
            messages = []
            continue
        messages.append({"role": "user", "content": line})
        while len(prompt_ids(tokenizer, messages)) + args.max_new_tokens > model.cfg.block_size and len(messages) > 2:
            messages = messages[2:]  # the oldest exchange leaves the context
        written: list[int] = []
        shown = 0

        def show(token: int) -> None:  # print the text as it is written, whole characters only
            nonlocal shown
            written.append(token)
            text = tokenizer.decode(written)
            if not text.endswith("�"):
                print(text[shown:], end="", flush=True)
                shown = len(text)

        print("gpt-pp> ", end="", flush=True)
        _, stopped = reply(model, prompt_ids(tokenizer, messages), stop, on_token=show, **sampling)
        print("" if stopped == ASSISTANT_END else f"  [{stopped}]")
        messages.append({"role": "assistant", "content": tokenizer.decode(written)})
        if sampling["seed"] is not None:
            sampling["seed"] += 1  # a new draw for every turn, the same conversation for the same seed


def main(argv: Optional[list[str]] = None) -> Optional[dict]:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        parser = argparse.ArgumentParser(prog="python -m gptpp.chat train", description="Run the chat stage.")
        parser.add_argument("--base", required=True, help="the pretrained run (name under $GPTPP_WORK/ckpt, "
                                                          "directory) or a checkpoint file")
        parser.add_argument("--ckpt", default="model.pt", help="checkpoint file inside the base run")
        parser.add_argument("--run", required=True, help="name of the run: its directory under --out")
        parser.add_argument("--data", default="chat/bpe32k", help="<dataset>/<tokenizer> under $GPTPP_WORK/data")
        parser.add_argument("--epochs", type=float, default=1.0, help="passes over the conversations")
        parser.add_argument("--tokens", type=float, default=None, help="budget in tokens read, instead of --epochs")
        parser.add_argument("--batch", type=int, default=32, help="rows per forward pass")
        parser.add_argument("--grad-accum", type=int, default=1, help="forward passes per optimizer step")
        parser.add_argument("--lr-scale", type=float, default=0.2,
                            help="every peak rate of pretraining is multiplied by this")
        parser.add_argument("--warmup-tokens", type=float, default=1e6)
        parser.add_argument("--final-lr-frac", type=float, default=0.05)
        parser.add_argument("--eval-tokens", type=float, default=5e6, help="evaluate every this many tokens read")
        parser.add_argument("--eval-batch", type=int, default=16, help="rows per forward pass of an evaluation")
        parser.add_argument("--plain-data", default="fineweb-edu/bpe32k",
                            help="plain-text dataset whose held-out windows are scored at each evaluation; 'none' to skip")
        parser.add_argument("--plain-windows", type=int, default=128)
        parser.add_argument("--sample-prompt", default="What is the capital of France?",
                            help="a question answered greedily at each evaluation; '' to skip")
        parser.add_argument("--sample-tokens", type=int, default=48)
        parser.add_argument("--log-every", type=int, default=10)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--no-compile", action="store_true", help="do not torch.compile the model")
        parser.add_argument("--no-model", action="store_true", help="do not write model.pt (a probe)")
        parser.add_argument("--out", default=None, help="where run directories live (default $GPTPP_WORK/ckpt)")
        parser.add_argument("--results", default=None, help="directory that receives <run>/ with the small result "
                                                            "files (default <repo>/results/runs; 'none' to skip)")
        parser.add_argument("--overwrite", action="store_true", help="replace a completed run of the same name")
        return finetune(parser.parse_args(argv[1:]))

    parser = argparse.ArgumentParser(prog="python -m gptpp.chat", description="Talk to a model after the chat stage "
                                     "('python -m gptpp.chat train --help' for the stage itself).")
    parser.add_argument("--run", required=True, help="run name under $GPTPP_WORK/ckpt, or a run directory")
    parser.add_argument("--ckpt", default="model.pt")
    parser.add_argument("--prompt", default=None, help="one user message to answer; without it, a conversation")
    parser.add_argument("--tokenizer", default=None, help="tokenizer name (default: the one of the run's data)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 takes the most likely token")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    talk(parser.parse_args(argv))
    return None


if __name__ == "__main__":
    main()
