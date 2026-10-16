"""Family B: the largest subset of an abelian group (or of [1..n]) with no
solution to a fixed linear equation.

The forbidden configurations are precomputed as hyperedges (sets of at most four
elements that together form a solution); f(n) is the largest set containing no
hyperedge, found by backtracking with incremental forbidden-element propagation:
when an element joins the set, every hyperedge it completes up to one missing
element forbids that element.
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import defaultdict

from .base import Family

GROUPS = ("int", "cyclic", "fpk")
NAMED_EQUATIONS = ("3ap", "sumfree", "sidon")
PRIMES = (2, 3, 5, 7)
FIELDS = ("group", "p", "equation", "equation2")
MAX_COEF = 9
MAX_AMBIENT = 128


class _Timeout(Exception):
    pass


class Ambient:
    """Elements 0..N-1 with an addition table and scalar multiplication."""

    def __init__(self, group: str, n: int, p: int | None):
        self.group, self.n, self.p = group, n, p
        if group == "int":
            self.N = n
            self.values = list(range(1, n + 1))
        elif group == "cyclic":
            self.N = n
            self.values = list(range(n))
        else:
            self.N = p ** n
            self.values = list(range(self.N))

    def _digits(self, x: int) -> list[int]:
        out = []
        for _ in range(self.n):
            out.append(x % self.p)
            x //= self.p
        return out

    def _undigits(self, d: list[int]) -> int:
        x = 0
        for v in reversed(d):
            x = x * self.p + v
        return x

    def lin(self, coefs: list[int], elems: list[int]):
        """sum c_i * x_i as a hashable group value (ints: an integer; no wrap)."""
        if self.group == "int":
            return sum(c * self.values[e] for c, e in zip(coefs, elems))
        if self.group == "cyclic":
            return sum(c * e for c, e in zip(coefs, elems)) % self.n
        acc = [0] * self.n
        for c, e in zip(coefs, elems):
            for i, d in enumerate(self._digits(e)):
                acc[i] = (acc[i] + c * d) % self.p
        return tuple(acc)


def _eq_key(eq) -> str:
    return json.dumps(eq, sort_keys=True, separators=(",", ":"))


def _invariant(eq) -> bool:
    """x -> x + t maps solutions to solutions iff the coefficients balance
    (3ap: 1+1=2; sidon: always; generic: a+b=c)."""
    return eq in ("3ap", "sidon") or (isinstance(eq, dict) and eq["a"] + eq["b"] == eq["c"])


def _solutions(eq, amb: Ambient) -> list[frozenset]:
    """Every forbidden configuration of one equation as a set of element indices."""
    N = amb.N
    edges: set[frozenset] = set()
    if eq == "sidon":
        by_sum = defaultdict(list)
        for a in range(N):
            for b in range(a, N):
                by_sum[amb.lin([1, 1], [a, b])].append((a, b))
        for pairs in by_sum.values():
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    edges.add(frozenset(pairs[i] + pairs[j]))
        return list(edges)
    if eq == "3ap":
        a, b, c, need_xy_distinct, need_not_all_eq = 1, 1, 2, True, False
    elif eq == "sumfree":
        a, b, c, need_xy_distinct, need_not_all_eq = 1, 1, 1, False, False
    else:
        a, b, c = eq["a"], eq["b"], eq["c"]
        need_xy_distinct, need_not_all_eq = False, True
    cz = defaultdict(list)
    for z in range(N):
        cz[amb.lin([c], [z])].append(z)
    for x in range(N):
        for y in range(N):
            if need_xy_distinct and x == y:
                continue
            for z in cz.get(amb.lin([a, b], [x, y]), ()):
                if need_not_all_eq and x == y == z:
                    continue
                edges.add(frozenset((x, y, z)))
    return list(edges)


def max_free_set(N: int, edges: list[frozenset], time_limit: float | None,
                 force_first: bool = False, return_set: bool = False):
    """Largest subset of range(N) containing no edge. Raises _Timeout.

    Returns its size, or ``(size, members)`` with the argmax as a sorted index list
    when ``return_set`` is true."""
    deadline = None if time_limit is None else time.monotonic() + time_limit
    forbidden0 = 0
    by_elem: list[list[int]] = [[] for _ in range(N)]
    for e in edges:
        if len(e) == 1:
            forbidden0 |= 1 << next(iter(e))
            continue
        mask = 0
        for v in e:
            mask |= 1 << v
        for v in e:
            by_elem[v].append(mask)
    # rem[i] = bits i..N-1
    full = (1 << N) - 1
    best = 0
    best_set = 0
    nodes = 0

    def rec(chosen: int, cnt: int, forb: int, start: int) -> None:
        nonlocal best, best_set, nodes
        nodes += 1
        if deadline is not None and (nodes & 1023) == 0 and time.monotonic() > deadline:
            raise _Timeout()
        avail = (full >> start << start) & ~forb & ~chosen
        if cnt + avail.bit_count() <= best:
            return
        if not avail:
            if cnt > best:
                best, best_set = cnt, chosen
            return
        low = avail & -avail
        e = low.bit_length() - 1
        new = chosen | low
        f2 = forb
        ok = True
        for h in by_elem[e]:
            rest = h & ~new
            if rest == 0:
                ok = False
                break
            if rest & (rest - 1) == 0:
                f2 |= rest
        if ok:
            rec(new, cnt + 1, f2, e + 1)
        rec(chosen, cnt, forb | low, e + 1)

    if force_first and N > 0 and not (forbidden0 & 1):
        # Translation-invariant problem in a group: WLOG element 0 is in a
        # non-empty optimal set.
        f2 = forbidden0
        for h in by_elem[0]:
            rest = h & ~1
            if rest & (rest - 1) == 0:
                f2 |= rest
        best, best_set = 1, 1
        rec(1, 1, f2, 1)
    else:
        rec(0, 0, forbidden0, 0)
    if return_set:
        return best, [i for i in range(N) if best_set >> i & 1]
    return best


class AddFree(Family):
    name = "addfree"
    label = "B: equation-free subsets"

    def fields(self) -> list[str]:
        return list(FIELDS)

    def default_ns(self, spec: dict) -> list[int]:
        if spec.get("group") == "fpk":
            p = spec.get("p") or 3
            return [k for k in (1, 2, 3, 4) if p ** k <= 27]
        return list(range(4, 21))

    def ns_regime(self, spec: dict):
        return (spec.get("group"), spec.get("p") if spec.get("group") == "fpk" else None)

    def normalize(self, spec: dict) -> dict:
        out = {f: None for f in FIELDS}
        out.update(copy.deepcopy(spec))
        return out

    def canonical(self, spec: dict) -> dict:
        s = self.normalize(spec)
        if s.get("group") != "fpk":
            s["p"] = None
        if s.get("equation2") is None:
            # absent, not null: specs without a second equation hash exactly as in run 1
            s.pop("equation2", None)
        elif s.get("equation") is not None and _eq_key(s["equation2"]) < _eq_key(s["equation"]):
            s["equation"], s["equation2"] = s["equation2"], s["equation"]
        return s

    def params(self, spec: dict) -> dict[str, float]:
        out = {}
        if spec.get("group") == "fpk":
            out["p"] = spec["p"]
        eq = spec.get("equation")
        if isinstance(eq, dict):
            out.update({k: eq[k] for k in ("a", "b", "c")})
        eq2 = spec.get("equation2")
        if isinstance(eq2, dict):
            out.update({k + "2": eq2[k] for k in ("a", "b", "c")})
        return {k: v for k, v in out.items() if v is not None}

    def validate(self, spec: dict) -> list[str]:
        errs = []
        unknown = set(spec) - set(FIELDS) - {"ns"}
        if unknown:
            errs.append(f"unknown fields {sorted(unknown)}")
        g = spec.get("group")
        if g not in GROUPS:
            errs.append(f"group must be one of {GROUPS}")
        p = spec.get("p")
        if g == "fpk" and p not in PRIMES:
            errs.append(f"fpk needs p in {PRIMES}")
        if p is not None and p not in PRIMES:
            errs.append(f"p must be null or one of {PRIMES}")
        eq = spec.get("equation")
        if eq is None:
            errs.append("equation missing")
        else:
            errs += self._equation_errors("equation", eq, g, p)
        eq2 = spec.get("equation2")
        if eq2 is not None:
            errs += self._equation_errors("equation2", eq2, g, p)
            if eq is not None and _eq_key(eq2) == _eq_key(eq):
                errs.append("equation2 must differ from equation (use null for a single equation)")
        ns = spec.get("ns")
        if ns is not None:
            if not isinstance(ns, list) or not ns or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in ns):
                errs.append("ns must be a non-empty list of positive integers")
            elif g == "fpk" and p in PRIMES and any(p ** x > MAX_AMBIENT for x in ns):
                errs.append(f"fpk ambient p^n must be <= {MAX_AMBIENT}")
            elif g in ("int", "cyclic") and any(x > 40 for x in ns):
                errs.append("int/cyclic n must be <= 40")
        return errs

    @staticmethod
    def _equation_errors(field: str, eq, g, p) -> list[str]:
        if isinstance(eq, str):
            if eq not in NAMED_EQUATIONS:
                return [f"{field} must be one of {NAMED_EQUATIONS} or {{a,b,c}}"]
        elif isinstance(eq, dict):
            if set(eq) != {"a", "b", "c"}:
                return [f"generic {field} needs exactly the keys a, b, c"]
            if not all(isinstance(eq[k], int) and not isinstance(eq[k], bool) and 1 <= abs(eq[k]) <= MAX_COEF for k in "abc"):
                return [f"{field} coefficients must be non-zero integers with |c| <= {MAX_COEF}"]
            if g == "fpk" and p in PRIMES and any(eq[k] % p == 0 for k in "abc"):
                return [f"{field} coefficients must be non-zero mod p"]
        else:
            return [f"{field} must be a name or an {{a,b,c}} object"]
        return []

    def _equation_text(self, spec: dict, eq) -> str:
        if eq == "3ap":
            return "A contains no x, y, z with x ≠ y and x + y = 2z"
        if eq == "sumfree":
            return "A contains no x, y, z (not necessarily distinct) with x + y = z"
        if eq == "sidon":
            return "all sums a + b with a, b ∈ A and a ≤ b are distinct" if spec["group"] == "int" else \
                "the sums a + b over unordered pairs {a, b} ⊆ A (a = b allowed) are pairwise distinct"
        a, b, c = eq["a"], eq["b"], eq["c"]

        def term(k, v):
            return v if k == 1 else (f"-{v}" if k == -1 else f"{k}·{v}")
        lhs = term(a, "x") + (" + " + term(b, "y") if b > 0 else " - " + term(-b, "y"))
        return f"A contains no x, y, z, not all equal, with {lhs} = {term(c, 'z')}"

    def render(self, spec: dict) -> str:
        g = spec["group"]
        if g == "int":
            amb = "Let [n] = {1,…,n}, with ordinary integer addition (no wraparound). Consider subsets A ⊆ [n]"
        elif g == "cyclic":
            amb = "Let Z_n be the integers modulo n. Consider subsets A ⊆ Z_n"
        else:
            amb = (f"Let G = F_{spec['p']}^n, the n-dimensional vector space over the field with "
                   f"{spec['p']} elements. Consider subsets A ⊆ G")
        cond = self._equation_text(spec, spec["equation"])
        if spec.get("equation2") is not None:
            cond += ", and " + self._equation_text(spec, spec["equation2"])
        return f"{amb} such that {cond}. Let f(n) be the largest possible |A|."

    def truth(self, spec: dict, n: int, time_limit: float | None) -> int | None:
        return self.truth_witness(spec, n, time_limit)[0]

    @staticmethod
    def _equations_of(spec: dict) -> list:
        return [spec["equation"]] + ([spec["equation2"]] if spec.get("equation2") is not None else [])

    @staticmethod
    def _element(amb: Ambient, i: int):
        """Element index -> its value as shown to the model (F_p^k: a digit vector,
        coordinate 1 first)."""
        return amb._digits(i) if amb.group == "fpk" else amb.values[i]

    def truth_witness(self, spec: dict, n: int, time_limit: float | None) -> tuple[int | None, list | None]:
        """(f(n), an extremal set A as a sorted list of its elements), or (None, None)
        on timeout. Elements are integers (int: 1..n; cyclic: 0..n-1) or digit vectors."""
        amb = Ambient(spec["group"], n, spec.get("p"))
        t0 = time.monotonic()
        eqs = self._equations_of(spec)
        edges = list({e for eq in eqs for e in _solutions(eq, amb)})
        left = None if time_limit is None else max(0.0, time_limit - (time.monotonic() - t0))
        # In a group, WLOG 0 is in the set when every equation is translation invariant.
        invariant = spec["group"] != "int" and all(_invariant(eq) for eq in eqs)
        try:
            size, members = max_free_set(amb.N, edges, left, force_first=invariant, return_set=True)
        except _Timeout:
            return None, None
        return size, [self._element(amb, i) for i in members]

    def check_witness(self, spec: dict, n: int, witness: list) -> list[str]:
        """Errors if ``witness`` is not a solution-free set for ``spec`` at ``n``."""
        amb = Ambient(spec["group"], n, spec.get("p"))
        index = {json.dumps(self._element(amb, i)): i for i in range(amb.N)}
        idx = set()
        for x in witness:
            if json.dumps(x) not in index:
                return [f"element {x!r} is not in the ambient set"]
            idx.add(index[json.dumps(x)])
        errs = [] if len(idx) == len(witness) else ["repeated element"]
        for e in {e for eq in self._equations_of(spec) for e in _solutions(eq, amb)}:
            if e <= idx:
                errs.append(f"contains the solution {sorted(map(str, (self._element(amb, i) for i in e)))}")
        return errs

    def witness_text(self, spec: dict, n: int, witness: list) -> str:
        fmt = (lambda v: "(" + ", ".join(map(str, v)) + ")") if spec["group"] == "fpk" else str
        return (f"an extremal set A at n = {n} ({len(witness)} elements): {{"
                + ", ".join(fmt(x) for x in witness) + "}")

    def trivial_bound(self, spec: dict, n: int) -> int:
        return spec["p"] ** n if spec["group"] == "fpk" else n

    def trivial_expr(self, spec: dict) -> str:
        return "p**n" if spec["group"] == "fpk" else "n"

    def mutate_random(self, spec: dict, rng: random.Random) -> dict:
        from .base import validate_mutation

        for _ in range(200):
            child = self._mutate_once(spec, rng)
            if child is None:
                continue
            same = self.ns_regime(child) == self.ns_regime(spec)
            child["ns"] = (spec.get("ns") or self.default_ns(spec)) if same else self.default_ns(child)
            if not validate_mutation(self, spec, child) and self.problem_id(child) != self.problem_id(spec):
                return child
        raise RuntimeError("no valid single-field mutation found")

    def _mutate_once(self, spec: dict, rng: random.Random) -> dict | None:
        s = copy.deepcopy(spec)
        op = rng.choice(["equation", "coef", "coef", "group", "p", "equation2"])
        eq = s["equation"]
        if op == "equation":
            opts = [e for e in NAMED_EQUATIONS if e != eq] + [{"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 3}]
            s["equation"] = rng.choice(opts)
        elif op == "coef":
            base = eq if isinstance(eq, dict) else {"3ap": {"a": 1, "b": 1, "c": 2},
                                                     "sumfree": {"a": 1, "b": 1, "c": 1},
                                                     "sidon": {"a": 1, "b": 1, "c": 2}}[eq]
            new = dict(base)
            k = rng.choice("abc")
            new[k] = base[k] + rng.choice([-1, 1])
            if new[k] == 0:
                return None
            s["equation"] = new
        elif op == "group":
            s["group"] = rng.choice([g for g in GROUPS if g != s["group"]])
            if s["group"] == "fpk" and s.get("p") is None:
                return None  # p must already be present for a one-field move
        elif op == "p":
            if s["group"] != "fpk":
                return None
            s["p"] = rng.choice([q for q in PRIMES if q != s["p"]])
        elif op == "equation2":
            # add, change or drop the second equation
            if s.get("equation2") is not None and rng.random() < 0.4:
                s["equation2"] = None
            else:
                opts = [e for e in NAMED_EQUATIONS if e != eq and e != s.get("equation2")] + \
                    [{"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 1, "c": 3}]
                s["equation2"] = rng.choice(opts)
        return s

    def schema_text(self) -> str:
        return """Family B: the largest subset A of an ambient set with no solution to an equation. Spec fields (JSON):
- "group": "int" (A ⊆ [1..n], integer addition, no wraparound), "cyclic" (A ⊆ Z_n) or "fpk" (A ⊆ F_p^n; n plays the role of the dimension k).
- "p": a prime in 2, 3, 5, 7. Used only when group is "fpk"; keep it present (it is ignored otherwise) so that a group change is a one-field change.
- "equation": one of
    "3ap"     no x, y, z in A with x != y and x + y = 2z;
    "sumfree" no x, y, z in A (x = y allowed) with x + y = z;
    "sidon"   all sums a + b (a <= b, a, b in A) distinct;
    {"a": int, "b": int, "c": int}  no x, y, z in A, not all equal, with a*x + b*y = c*z (non-zero coefficients, |coef| <= 9).
- "equation2": null, or a second equation in the same forms, different from "equation"; A must then avoid solutions of both. Leave it null for a single equation.
- "ns": list of n values (int/cyclic n <= 40; fpk p^n <= 128); leave it out to keep the parent's."""
