"""What every exporter shares: where the site's data folder is, and a writer of packed arrays.

The site's figure kit reads `manifest.json`, a map from a key to `{file, offset, bytes, dtype, shape, lo, hi, note}`
(`dtype` is `u8`, `f16` or `f32`; a `u8` entry with `lo` and `hi` decodes to `lo + (hi - lo) * byte / 255`). Several
exporters may run at once, so none of them writes the manifest: each writes its own `<name>.entries.json`, and
`gptpp.export.manifest` merges them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SITE = Path(os.environ.get("GPTPP_SITE", "/home/ubuntu/d-.github.io"))
DATA = SITE / "assets" / "data" / "gpt"
WORK = Path(os.environ.get("GPTPP_WORK", "/home/ubuntu/d-work/gpt"))
CKPT = WORK / "ckpt"

_DTYPES = {"uint8": "u8", "float16": "f16", "float32": "f32"}


def quantize(a: np.ndarray, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, float, float]:
    """An array as bytes over [lo, hi] (its own range by default): what a figure only draws need not be a float."""
    a = np.asarray(a, dtype=np.float64)
    lo = float(a.min()) if lo is None else float(lo)
    hi = float(a.max()) if hi is None else float(hi)
    if hi <= lo:
        hi = lo + 1.0
    q = np.clip(np.rint((a - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
    return q, lo, hi


class Blob:
    """One `<name>.bin` holding several arrays, with the manifest entries that describe them.

        blob = Blob("attention-patterns")
        blob.put("attention-patterns.layer0", patterns_u8, "T x T patterns of layer 0, one per head", lo=0.0, hi=1.0)
        blob.save()            # writes attention-patterns.bin and attention-patterns.entries.json
    """

    def __init__(self, name: str, out: Path | str | None = None):
        self.name, self.out = name, Path(out) if out else DATA
        self.buf, self.entries = bytearray(), {}

    def put(self, key: str, array: np.ndarray, note: str, lo: float | None = None, hi: float | None = None) -> dict:
        if not key.startswith(self.name):
            raise ValueError(f"key {key!r} must begin with the file's name {self.name!r}: keys are unique in the folder")
        a = np.ascontiguousarray(array)
        if str(a.dtype) not in _DTYPES:
            raise TypeError(f"{key}: dtype {a.dtype} is not one of uint8, float16, float32")
        self.buf += b"\0" * ((-len(self.buf)) % a.itemsize)          # floats stay aligned to their size
        entry = {"dtype": _DTYPES[str(a.dtype)], "file": f"{self.name}.bin", "offset": len(self.buf),
                 "bytes": int(a.nbytes), "shape": list(a.shape), "note": note}
        if entry["dtype"] == "u8" and lo is not None and hi is not None:
            entry.update(lo=float(lo), hi=float(hi))
        self.entries[key] = entry
        self.buf += a.astype(a.dtype.newbyteorder("<")).tobytes()
        return entry

    def put_u8(self, key: str, array: np.ndarray, note: str, lo: float | None = None, hi: float | None = None) -> dict:
        """Quantize to bytes over [lo, hi] and store."""
        q, lo, hi = quantize(array, lo, hi)
        return self.put(key, q, note, lo, hi)

    def save(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        write_atomic(self.out / f"{self.name}.bin", bytes(self.buf))
        write_json(self.out / f"{self.name}.entries.json", self.entries)
        return len(self.buf)


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path | str, obj, compact: bool = True) -> int:
    """JSON with short floats; returns the bytes written."""
    text = json.dumps(_round(obj), separators=(",", ":") if compact else None, ensure_ascii=False, indent=None if compact else 1)
    write_atomic(Path(path), text.encode("utf-8"))
    return len(text.encode("utf-8"))


def _round(o, digits: int = 6):
    """Floats to six significant digits: five made a loss printed to three decimals round twice (2.649451 was
    stored as 2.6495 and printed as 2.650)."""
    if isinstance(o, float):
        return float(f"{o:.{digits}g}")
    if isinstance(o, dict):
        return {k: _round(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, digits) for v in o]
    if isinstance(o, np.generic):
        return _round(o.item(), digits)
    if isinstance(o, np.ndarray):
        return _round(o.tolist(), digits)
    return o
