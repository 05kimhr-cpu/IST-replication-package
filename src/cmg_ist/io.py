"""Minimal IO for the shared CMG raw data.

DATA_ROOT is read from the environment variable CMG_IST_DATA_ROOT if
set, otherwise falls back to a sibling `data/raw` directory next to
the package.  A reader reproducing the study should set the
environment variable to point at their MCMD clone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_ROOT = Path(os.environ.get("CMG_IST_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))


def iter_samples(language: str) -> Iterator[dict]:
    path = DATA_ROOT / f"{language}.jsonl"
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_samples(language: str, limit: int | None = None) -> list[dict]:
    out = []
    for i, s in enumerate(iter_samples(language)):
        if limit is not None and i >= limit:
            break
        out.append(s)
    return out


def clean_msg(raw_msg: str) -> str:
    """MCMD-style messages end with trailing newlines and may have odd spacing.

    We strip the outer whitespace only. Inner formatting is what we study.
    """
    return raw_msg.strip()
