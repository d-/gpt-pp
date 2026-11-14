"""Merge every `<name>.entries.json` of the site's data folder into `manifest.json`.

    python -m gptpp.export.manifest [--out <dir>]

Safe to run at any time and by several exporters at once: it only reads the entries files and replaces the manifest
in one step. It checks that keys are unique, that every entry lies inside an existing file, and prints the folder's
size by file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import DATA, write_json

_SIZE = {"u8": 1, "f16": 2, "f32": 4}


def merge(out: Path) -> dict:
    manifest, owner, problems = {}, {}, []
    for f in sorted(out.glob("*.entries.json")):
        for key, e in json.loads(f.read_text()).items():
            if key in manifest:
                problems.append(f"key {key} is in {owner[key]} and in {f.name}")
            manifest[key], owner[key] = e, f.name
            target = out / e["file"]
            n = 1
            for d in e["shape"]:
                n *= d
            if not target.exists():
                problems.append(f"{key}: {e['file']} does not exist")
            elif e["offset"] + e["bytes"] > target.stat().st_size:
                problems.append(f"{key}: lies outside {e['file']}")
            if n * _SIZE[e["dtype"]] != e["bytes"]:
                problems.append(f"{key}: bytes {e['bytes']} is not shape x dtype size")
    manifest["_generated"] = {"by": "gptpp.export.manifest", "entries": len(manifest)}
    write_json(out / "manifest.json", manifest)
    return {"entries": len(manifest) - 1, "problems": problems}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    out = Path(ap.parse_args().out)
    res = merge(out)
    total = 0
    for f in sorted(out.iterdir()):
        if f.is_file():
            total += f.stat().st_size
            print(f"{f.stat().st_size / 1e6:8.3f} MB  {f.name}")
    print(f"{total / 1e6:8.3f} MB  in all; {res['entries']} manifest entries")
    for p in res["problems"]:
        print("PROBLEM:", p)
    sys.exit(1 if res["problems"] else 0)


if __name__ == "__main__":
    main()
