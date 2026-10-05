"""The problem-family interface.

A *spec* is a JSON-able dict describing one problem; the varying size parameter
is always called ``n`` and its computed values are listed in ``spec["ns"]``.
Everything except ``ns`` identifies the problem.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from abc import ABC, abstractmethod


class Family(ABC):
    #: Short family name; also the calibration "domain".
    name: str = ""
    #: Human label used in reports.
    label: str = ""

    # ---- required -------------------------------------------------------
    @abstractmethod
    def validate(self, spec: dict) -> list[str]:
        """Return a list of errors; empty means the spec is valid."""

    @abstractmethod
    def render(self, spec: dict) -> str:
        """Neutral natural-language statement defining f(n). No names, no story."""

    @abstractmethod
    def truth(self, spec: dict, n: int, time_limit: float | None) -> int | None:
        """Exact f(n) by brute force, or None on timeout."""

    @abstractmethod
    def trivial_bound(self, spec: dict, n: int) -> int:
        """The trivial upper bound the protocol defines for this family."""

    @abstractmethod
    def mutate_random(self, spec: dict, rng: random.Random) -> dict:
        """A valid spec differing from ``spec`` in exactly one field (ns aside)."""

    @abstractmethod
    def schema_text(self) -> str:
        """Description of the spec fields, shown to the model in the mutate prompt."""

    @abstractmethod
    def default_ns(self, spec: dict) -> list[int]:
        """Default list of n values for this spec."""

    # ---- shared ---------------------------------------------------------
    def normalize(self, spec: dict) -> dict:
        """Copy with every field present and list values sorted; used to diff specs."""
        out = {f: None for f in self.fields()}
        for k, v in copy.deepcopy(spec).items():
            out[k] = sorted(set(v)) if isinstance(v, list) and k != "ns" else v
        return out

    def canonical(self, spec: dict) -> dict:
        """Identity form for problem ids: ``normalize`` plus dropping fields that do
        not affect the problem. Override as needed."""
        return self.normalize(spec)

    def params(self, spec: dict) -> dict[str, float]:
        """Numeric spec parameters exposed to bound expressions besides ``n``."""
        out = {}
        for k, v in spec.items():
            if k == "ns":
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = v
        return out

    def ns_regime(self, spec: dict):
        """Specs with the same regime can share an ns list (used for mutated children)."""
        return None

    def problem_id(self, spec: dict) -> str:
        """Hash of the canonical sorted JSON of the spec, excluding ``ns``."""
        canon = {k: v for k, v in self.canonical(spec).items() if k != "ns"}
        canon["family"] = self.name
        blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def fields(self) -> list[str]:
        """Spec field names (excluding ns)."""
        raise NotImplementedError

    def statement_length(self, spec: dict) -> int:
        """L(spec), PROTOCOL.md "Run 2": atoms of the canonical spec excluding ns. Each
        present field counts 1 for its name plus 1 per number in its value; a string or
        boolean value counts 1. Fields the canonical form drops (null, or irrelevant like
        p without a mod condition) count 0. A generic equation {a, b, c} counts 1 + 3."""
        return sum(1 + _atoms(v) for k, v in self.canonical(spec).items() if k != "ns" and v is not None)


def _atoms(v) -> int:
    if isinstance(v, dict):
        return sum(_atoms(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return sum(_atoms(x) for x in v)
    return 1


def changed_fields(parent: dict, child: dict) -> list[str]:
    """Spec fields (excluding ns) whose values differ between two specs."""
    keys = (set(parent) | set(child)) - {"ns"}
    return sorted(k for k in keys if parent.get(k) != child.get(k))


def validate_mutation(family: Family, parent: dict, child: dict) -> list[str]:
    """Errors for a proposed mutation: it must be valid and change exactly one field.

    Fields are compared on the family's normalised form, so reordering a residue
    list is not a change.
    """
    errors = []
    unknown = set(child) - set(family.fields()) - {"ns"}
    if unknown:
        errors.append(f"unknown fields: {sorted(unknown)}")
    errors += family.validate(child)
    if errors:
        return errors
    diff = changed_fields(family.normalize(parent), family.normalize(child))
    if len(diff) != 1:
        errors.append(f"mutation must change exactly one spec field (ns aside); changed {diff}")
    return errors
