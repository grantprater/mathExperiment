"""Family A: families of subsets of [n] with size and pairwise-intersection conditions.

f(n) is the largest family, found as a maximum clique in the compatibility graph
whose vertices are the subsets allowed by the size conditions and whose edges
join two subsets whose intersection meets the intersection conditions.
"""

from __future__ import annotations

import copy
import math
import random

from .base import Family
from .clique import Timeout, max_clique

PRIMES = (2, 3, 5, 7)
FIELDS = ("p", "size_mod", "size_abs", "int_mod", "int_abs", "int_min")
MAX_N = 8


def _fmt_set(vals: list[int]) -> str:
    if len(vals) == 1:
        return str(vals[0])
    return " or ".join(str(v) for v in vals)


class SetFamily(Family):
    name = "setfam"
    label = "A: set families on [n]"

    def fields(self) -> list[str]:
        return list(FIELDS)

    def default_ns(self, spec: dict) -> list[int]:
        return [3, 4, 5, 6, 7]

    def canonical(self, spec: dict) -> dict:
        s = self.normalize(spec)
        # p only matters through the mod conditions.
        if s.get("size_mod") is None and s.get("int_mod") is None:
            s["p"] = None
        return s

    def params(self, spec: dict) -> dict[str, float]:
        out = {"p": spec.get("p", 2)}
        if spec.get("int_min") is not None:
            out["int_min"] = spec["int_min"]
        if spec.get("size_abs") and len(spec["size_abs"]) == 1:
            out["k"] = spec["size_abs"][0]
        if spec.get("int_abs") and len(spec["int_abs"]) == 1:
            out["lam"] = spec["int_abs"][0]
        return out

    # ---- validation -----------------------------------------------------
    def validate(self, spec: dict) -> list[str]:
        errs = []
        unknown = set(spec) - set(FIELDS) - {"ns"}
        if unknown:
            errs.append(f"unknown fields {sorted(unknown)}")
        p = spec.get("p")
        if p not in PRIMES:
            errs.append(f"p must be one of {PRIMES}, got {p!r}")
            p = None
        for f in ("size_mod", "int_mod"):
            v = spec.get(f)
            if v is None:
                continue
            if not isinstance(v, list) or not v or not all(isinstance(r, int) and not isinstance(r, bool) for r in v):
                errs.append(f"{f} must be null or a non-empty list of integers")
            elif p is not None and not all(0 <= r < p for r in v):
                errs.append(f"{f} residues must lie in 0..p-1 (p={p})")
            elif p is not None and len(set(v)) == p:
                errs.append(f"{f} allows every residue mod {p}; use null instead")
        for f in ("size_abs", "int_abs"):
            v = spec.get(f)
            if v is None:
                continue
            if not isinstance(v, list) or not v or not all(isinstance(r, int) and not isinstance(r, bool) and 0 <= r <= MAX_N for r in v):
                errs.append(f"{f} must be null or a non-empty list of integers in 0..{MAX_N}")
        im = spec.get("int_min")
        if im is not None and not (isinstance(im, int) and not isinstance(im, bool) and 0 <= im <= MAX_N):
            errs.append(f"int_min must be null or an integer in 0..{MAX_N}")
        if not errs and not any(self._size_ok(spec, s) for s in range(MAX_N + 1)):
            errs.append("the size conditions admit no subset size (the problem is empty)")
        ns = spec.get("ns")
        if ns is not None:
            if not isinstance(ns, list) or not ns or not all(isinstance(x, int) and 1 <= x <= MAX_N for x in ns):
                errs.append(f"ns must be a non-empty list of integers in 1..{MAX_N}")
        return errs

    # ---- rendering ------------------------------------------------------
    def render(self, spec: dict) -> str:
        p = spec.get("p")
        size_conds = []
        if spec.get("size_mod") is not None:
            size_conds.append(f"|A| ≡ {_fmt_set(sorted(spec['size_mod']))} (mod {p})")
        if spec.get("size_abs") is not None:
            v = sorted(spec["size_abs"])
            size_conds.append(f"|A| = {v[0]}" if len(v) == 1 else "|A| ∈ {" + ", ".join(map(str, v)) + "}")
        int_conds = []
        if spec.get("int_mod") is not None:
            int_conds.append(f"|A ∩ B| ≡ {_fmt_set(sorted(spec['int_mod']))} (mod {p})")
        if spec.get("int_abs") is not None:
            v = sorted(spec["int_abs"])
            int_conds.append(f"|A ∩ B| = {v[0]}" if len(v) == 1 else "|A ∩ B| ∈ {" + ", ".join(map(str, v)) + "}")
        if spec.get("int_min") is not None:
            int_conds.append(f"|A ∩ B| ≥ {spec['int_min']}")
        parts = []
        if size_conds:
            parts.append("every member A has " + " and ".join(size_conds))
        if int_conds:
            parts.append("any two distinct members A ≠ B have " + " and ".join(int_conds))
        text = "Let [n] = {1,…,n}. Consider families F of distinct subsets of [n]"
        if parts:
            text += " such that " + ", and ".join(parts)
        return text + ". Let f(n) be the largest possible |F|."

    # ---- truth ----------------------------------------------------------
    def _size_ok(self, spec: dict, s: int) -> bool:
        p = spec.get("p")
        if spec.get("size_mod") is not None and s % p not in spec["size_mod"]:
            return False
        if spec.get("size_abs") is not None and s not in spec["size_abs"]:
            return False
        return True

    def _int_ok(self, spec: dict, t: int) -> bool:
        p = spec.get("p")
        if spec.get("int_mod") is not None and t % p not in spec["int_mod"]:
            return False
        if spec.get("int_abs") is not None and t not in spec["int_abs"]:
            return False
        if spec.get("int_min") is not None and t < spec["int_min"]:
            return False
        return True

    def truth(self, spec: dict, n: int, time_limit: float | None) -> int | None:
        return self.truth_witness(spec, n, time_limit)[0]

    def truth_witness(self, spec: dict, n: int, time_limit: float | None) -> tuple[int | None, list | None]:
        """(f(n), an extremal family as sorted lists of elements of [n] = {1..n}), or
        (None, None) on timeout. The family is the maximum clique's vertex set."""
        verts = [m for m in range(1 << n) if self._size_ok(spec, m.bit_count())]
        int_ok = [self._int_ok(spec, t) for t in range(n + 1)]
        adj = [0] * len(verts)
        for i, a in enumerate(verts):
            row = 0
            for j, b in enumerate(verts):
                if i != j and int_ok[(a & b).bit_count()]:
                    row |= 1 << j
            adj[i] = row
        try:
            size, clique = max_clique(adj, time_limit)
        except Timeout:
            return None, None
        fam = sorted((sorted(i + 1 for i in range(n) if verts[v] >> i & 1) for v in clique),
                     key=lambda a: (len(a), a))
        return size, fam

    def check_witness(self, spec: dict, n: int, witness: list) -> list[str]:
        """Errors if ``witness`` is not a valid family for ``spec`` at ``n``."""
        errs = []
        sets = [frozenset(a) for a in witness]
        if len(set(sets)) != len(sets):
            errs.append("repeated member")
        for a in sets:
            if not a <= set(range(1, n + 1)):
                errs.append(f"member {sorted(a)} not inside [n]")
            if not self._size_ok(spec, len(a)):
                errs.append(f"member {sorted(a)} fails the size condition")
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                if not self._int_ok(spec, len(a & b)):
                    errs.append(f"members {sorted(a)}, {sorted(b)} meet in {len(a & b)}")
        return errs

    def witness_text(self, spec: dict, n: int, witness: list) -> str:
        sets = ["{" + ", ".join(map(str, a)) + "}" for a in witness]
        return f"an extremal family F at n = {n} ({len(witness)} members): " + ", ".join(sets)

    def trivial_bound(self, spec: dict, n: int) -> int:
        return sum(math.comb(n, s) for s in range(n + 1) if self._size_ok(spec, s))

    def trivial_expr(self, spec: dict) -> str:
        """A valid loose upper bound as an expression (used by the mock backend)."""
        return "2**n"

    # ---- mutation -------------------------------------------------------
    def mutate_random(self, spec: dict, rng: random.Random) -> dict:
        for _ in range(200):
            child = self._mutate_once(spec, rng)
            if child is None:
                continue
            from .base import validate_mutation

            if not validate_mutation(self, spec, child) and self.problem_id(child) != self.problem_id(spec):
                return child
        raise RuntimeError("no valid single-field mutation found")

    def _mutate_once(self, spec: dict, rng: random.Random) -> dict | None:
        s = copy.deepcopy(spec)
        p = s["p"]
        op = rng.choice(["toggle_mod", "toggle_mod", "move_p", "size_abs", "mod_to_none", "add_abs"])
        if op == "toggle_mod":
            f = rng.choice(["size_mod", "int_mod"])
            r = rng.randrange(p)
            cur = set(s.get(f) or [])
            cur ^= {r}
            s[f] = sorted(cur) if cur and len(cur) < p else None
        elif op == "move_p":
            # Residues are kept as integers; only primes that keep every residue
            # in range are eligible, so the move changes the single field p.
            mx = max([*(s.get("size_mod") or []), *(s.get("int_mod") or []), 0])
            choices = [q for q in PRIMES if q != p and q > mx]
            if not choices:
                return None
            s["p"] = rng.choice(choices)
        elif op == "size_abs":
            cur = s.get("size_abs")
            if cur is None:
                s["size_abs"] = [rng.randrange(1, 5)]
            elif rng.random() < 0.4:
                s["size_abs"] = None
            else:
                v = cur[0] + rng.choice([-1, 1])
                if v < 0:
                    return None
                s["size_abs"] = [v]
        elif op == "mod_to_none":
            # One half of "swap a mod condition for an abs one": drop the mod.
            f = rng.choice(["size_mod", "int_mod"])
            if s.get(f) is None:
                return None
            s[f] = None
        elif op == "add_abs":
            # The other half: introduce an exact-intersection condition.
            if s.get("int_abs") is None:
                s["int_abs"] = [rng.randrange(0, 3)]
            else:
                s["int_abs"] = None
        return s

    def schema_text(self) -> str:
        return """Family A: families of distinct subsets of [n] = {1..n}. Spec fields (JSON):
- "p": prime modulus, one of 2, 3, 5, 7 (used only by the *_mod fields).
- "size_mod": null, or a non-empty list of residues r in 0..p-1: every member A has |A| mod p in this list.
- "size_abs": null, or a non-empty list of integers: every member A has |A| in this list (one value = uniform).
- "int_mod": null, or a list of residues: distinct members A, B have |A ∩ B| mod p in this list.
- "int_abs": null, or a list of integers: distinct members A, B have |A ∩ B| in this list.
- "int_min": null, or an integer t: distinct members have |A ∩ B| >= t.
- "ns": list of n values to compute (n <= 8); leave it out to keep the parent's.
Null means "no condition". A list containing every residue mod p is not allowed (use null)."""
