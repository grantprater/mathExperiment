"""The prediction store: append-only JSONL ledgers (design borrowed from ~/reticle).

Files, all in one store directory:

- ``predictions.jsonl`` -- one row per binary claim with a numeric confidence,
  written *before* the outcome is known (``predicted_first: true``,
  ``outcome: null``); outcomes are separate rows
  ``{kind: "outcome", ref, outcome, observed, ...}`` joined by id.
- ``problems.jsonl`` -- problem lifecycle rows (queued / done / rejected / duplicate).
- ``weaves.jsonl`` -- one row per checked bound.
- ``calls.jsonl`` -- every model call: full prompt, raw response, parse status, meta.

Rules enforced in code (PROTOCOL.md: "every convention that was merely written
down got violated again"):

- (a) :meth:`Store.require_interval_predictions` refuses truth computation for
  a (problem, n) until its interval prediction is on disk;
- (b) a store opened for the ``mock`` backend refuses the repo's ``store/``.
"""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from . import REPO

FILES = ("predictions", "problems", "weaves", "calls")
REAL_STORE = (REPO / "store").resolve()
MOCK_DEFAULT = REPO / "scratch" / "store-mock"


class Refusal(RuntimeError):
    """A protocol rule refused an action."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{int(time.time())}-{secrets.token_hex(3)}"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


class Store:
    def __init__(self, path: Path | str, backend: str):
        self.dir = Path(path)
        if backend == "mock" and _inside(self.dir, REAL_STORE):
            raise Refusal(f"the mock backend may not write into the real store {REAL_STORE}; "
                          f"use e.g. {MOCK_DEFAULT}")
        self.backend = backend
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        assert name in FILES, name
        return self.dir / f"{name}.jsonl"

    def append(self, name: str, row: dict) -> dict:
        with self.path(name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        return row

    def rows(self, name: str) -> list[dict]:
        p = self.path(name)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ---- predictions --------------------------------------------------------
    def predict(self, *, run: str, by: str, domain: str, kind: str, level: str, claim: str,
                confidence: float, problem: str, call: str | None, prompt_version: str | None,
                **extra) -> dict:
        row = {"id": new_id("pred"), "when": now(), "run": run, "by": by, "domain": domain,
               "kind": kind, "level": level, "claim": claim, "confidence": float(confidence),
               "predicted_first": True, "problem": problem, "call": call,
               "prompt_version": prompt_version, **extra, "outcome": None}
        return self.append("predictions", row)

    def outcome(self, ref: str, outcome: str, observed, **extra) -> dict:
        assert outcome in ("right", "wrong", "couldnt-tell"), outcome
        row = {"id": new_id("out"), "when": now(), "kind": "outcome", "ref": ref,
               "outcome": outcome, "observed": observed, **extra}
        return self.append("predictions", row)

    def require_interval_predictions(self, problem: str, n: int) -> dict:
        """Refusal (a): truth for (problem, n) needs its interval prediction on disk."""
        for r in self.rows("predictions"):
            if r.get("kind") == "interval" and r.get("problem") == problem and r.get("n") == n:
                return r
        raise Refusal(f"refusing to compute f({n}) for problem {problem}: "
                      f"no interval prediction has been written for it")

    def joined_predictions(self) -> list[dict]:
        """Prediction rows with their latest outcome folded in, in file (time) order."""
        rows = self.rows("predictions")
        outs = {}
        for r in rows:
            if r.get("kind") == "outcome":
                outs[r["ref"]] = r
        joined = []
        for r in rows:
            if r.get("kind") == "outcome":
                continue
            o = outs.get(r["id"])
            if o:
                r = {**r, "outcome": o["outcome"], "observed": o.get("observed"),
                     **{k: v for k, v in o.items() if k in ("abs_log2_err", "log2_err")}}
            joined.append(r)
        return joined
