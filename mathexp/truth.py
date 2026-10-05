"""Brute-force truth with a JSON cache keyed by problem_id and n.

A timeout is cached as ``value: null`` (unknown) together with the time limit it
ran under; a later call with a larger limit retries it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import REPO
from .problems.base import Family

DEFAULT_CACHE = REPO / "cache" / "truth.json"
DEFAULT_TIME_LIMIT = 20.0


class TruthCache:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else Path(os.environ.get("MATHEXP_CACHE", DEFAULT_CACHE))
        self.data: dict[str, dict] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, family: Family, spec: dict, n: int,
            time_limit: float = DEFAULT_TIME_LIMIT) -> tuple[int | None, float, bool]:
        """Return (value, seconds, cached). value None means unknown (timeout)."""
        key = f"{family.problem_id(spec)}:{n}"
        hit = self.data.get(key)
        if hit is not None and (hit["value"] is not None or hit["time_limit"] >= time_limit):
            return hit["value"], hit["seconds"], True
        t0 = time.perf_counter()
        value = family.truth(spec, n, time_limit)
        secs = time.perf_counter() - t0
        self.data[key] = {"value": value, "seconds": round(secs, 4), "time_limit": time_limit,
                          "family": family.name}
        self._save()
        return value, secs, False
