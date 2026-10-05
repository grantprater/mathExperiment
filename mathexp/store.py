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

- (a) :meth:`Store.require_value_predictions` refuses truth computation for
  a (problem, n) until its value-level prediction (a distribution; an interval in
  run 1) is on disk;
- (b) a store opened for the ``mock`` backend refuses the repo's ``store/``;
- (run 3) :meth:`Store.require_method_prediction` refuses the A4b probe until the
  problem's ``method_band`` row is on disk.
"""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from . import REPO

FILES = ("predictions", "problems", "weaves", "calls")
#: Prediction kinds that count as "f(n) was predicted" for refusal (a): run 2's
#: distribution, and run 1's interval (kept so run-1 stores still read).
VALUE_KINDS = ("distribution", "interval")
#: Outcome fields folded into the joined prediction rows.
OUTCOME_FIELDS = ("abs_log2_err", "log2_err", "bits", "p_true")
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
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and not text.endswith("\n"):
            # a process killed mid-append leaves an unterminated last line: skip it if
            # it does not parse, so a crashed run's store still reads
            try:
                json.loads(lines[-1])
            except json.JSONDecodeError:
                lines = lines[:-1]
        return [json.loads(line) for line in lines if line.strip()]

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

    def require_value_predictions(self, problem: str, n: int) -> dict:
        """Refusal (a): truth for (problem, n) needs its value prediction on disk."""
        for r in self.rows("predictions"):
            if r.get("kind") in VALUE_KINDS and r.get("problem") == problem and r.get("n") == n:
                return r
        raise Refusal(f"refusing to compute f({n}) for problem {problem}: "
                      f"no value prediction has been written for it")

    def require_method_prediction(self, problem: str, run: str | None = None) -> dict:
        """Run 3 refusal: the probe weave for a problem needs its method_band row on disk."""
        for r in self.rows("predictions"):
            if r.get("kind") == "method_band" and r.get("problem") == problem and (run is None or r.get("run") == run):
                return r
        raise Refusal(f"refusing to run the probe for problem {problem}: "
                      f"no method_band prediction has been written for it")

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
                     **{k: v for k, v in o.items() if k in OUTCOME_FIELDS}}
            joined.append(r)
        return joined
