# -*- coding: utf-8 -*-
"""
I/O utilities (torch save/load, jsonl, text files).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def torch_save(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)


def torch_load(path: str, map_location: str = "cpu") -> Any:
    return torch.load(path, map_location=map_location)


def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def read_jsonl(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], encoding: str = "utf-8") -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding=encoding) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

