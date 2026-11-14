"""Exporters: scripts that turn the lab's checkpoints and logs into the small files the site's figures draw.

Each exporter is run as `python -m gptpp.export.<name> --ckpt <path> [--out <dir>]`, can be run again when a newer
checkpoint lands, and writes `<name>.json` (structured, small) and, for arrays, `<name>.bin` with
`<name>.entries.json`; `python -m gptpp.export.manifest` then merges every `*.entries.json` into `manifest.json`.
"""
