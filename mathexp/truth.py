"""Brute-force truth with a JSON cache keyed by problem_id and n, plus a witness cache.

A timeout is cached as ``value: null`` (unknown) together with the time limit it
ran under; a later call with a larger limit retries it.

Run 3 keeps an extremal witness (the family or set achieving f(n)) for every value
computed: ``truth.json`` stays values-only, and the witnesses go to an append-only
sidecar ``<cache stem>.witness.jsonl`` next to it, so a large witness never makes
the values file slow to rewrite. A value cached before witnesses existed is
recomputed once by :meth:`TruthCache.get_witness`.

``save_every`` (seconds) throttles rewrites of ``truth.json`` for bulk builds
(``python -m mathexp clusters``); call :meth:`flush` at the end. The default 0
saves after every computation, as before.
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
    def __init__(self, path: Path | str | None = None, save_every: float = 0.0):
        self.path = Path(path) if path else Path(os.environ.get("MATHEXP_CACHE", DEFAULT_CACHE))
        self.witness_path = self.path.with_name(self.path.stem + ".witness.jsonl")
        self.data: dict[str, dict] = {}
        self.save_every = save_every
        self._last_save = 0.0
        self._dirty = False
        self._witnesses: dict[str, list] | None = None
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _save(self, force: bool = False) -> None:
        self._dirty = True
        if not force and self.save_every and time.monotonic() - self._last_save < self.save_every:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        self._last_save = time.monotonic()
        self._dirty = False

    def flush(self) -> None:
        if self._dirty:
            self._save(force=True)

    # ---- witnesses ----------------------------------------------------------
    def _load_witnesses(self) -> dict[str, list]:
        if self._witnesses is None:
            self._witnesses = {}
            if self.witness_path.exists():
                for line in self.witness_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        self._witnesses[r["key"]] = r["witness"]
        return self._witnesses

    def _put_witness(self, key: str, witness) -> None:
        if witness is None:
            return
        self._load_witnesses()[key] = witness
        self.witness_path.parent.mkdir(parents=True, exist_ok=True)
        with self.witness_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "witness": witness}, separators=(",", ":")) + "\n")

    def _compute(self, family: Family, spec: dict, n: int, key: str, time_limit: float):
        t0 = time.perf_counter()
        value, witness = family.truth_witness(spec, n, time_limit)
        secs = time.perf_counter() - t0
        self.data[key] = {"value": value, "seconds": round(secs, 4), "time_limit": time_limit,
                          "family": family.name}
        self._put_witness(key, witness)
        self._save()
        return value, witness, secs

    # ---- public -------------------------------------------------------------
    def get(self, family: Family, spec: dict, n: int,
            time_limit: float = DEFAULT_TIME_LIMIT) -> tuple[int | None, float, bool]:
        """Return (value, seconds, cached). value None means unknown (timeout)."""
        key = f"{family.problem_id(spec)}:{n}"
        hit = self.data.get(key)
        if hit is not None and (hit["value"] is not None or hit["time_limit"] >= time_limit):
            return hit["value"], hit["seconds"], True
        value, _, secs = self._compute(family, spec, n, key, time_limit)
        return value, secs, False

    def get_witness(self, family: Family, spec: dict, n: int,
                    time_limit: float = DEFAULT_TIME_LIMIT) -> tuple[int | None, list | None]:
        """(value, witness) at n; recomputes once if the value was cached without one."""
        key = f"{family.problem_id(spec)}:{n}"
        w = self._load_witnesses().get(key)
        hit = self.data.get(key)
        if w is not None and hit is not None and hit["value"] is not None:
            return hit["value"], w
        if hit is not None and hit["value"] is None and hit["time_limit"] >= time_limit:
            return None, None
        value, witness, _ = self._compute(family, spec, n, key, time_limit)
        return value, witness
