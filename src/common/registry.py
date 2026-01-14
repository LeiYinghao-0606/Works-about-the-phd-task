# -*- coding: utf-8 -*-
"""
Simple registry (optional, for pluggable builders/verifiers/models).
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[Any], Any]:
        def _wrap(obj: Any) -> Any:
            if key in self._items:
                raise KeyError(f"[{self.name}] duplicate key: {key}")
            self._items[key] = obj
            return obj
        return _wrap

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"[{self.name}] unknown key: {key}. Available: {list(self._items.keys())}")
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def keys(self):
        return list(self._items.keys())

