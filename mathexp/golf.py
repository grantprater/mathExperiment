"""Golf length: the node count of the shortest expression that reproduces a sequence.

PROTOCOL.md "Run 2". Bottom-up enumeration with observational equivalence over a
fixed tuple of n values:

- terminals ``n, 1, 2, 3`` (size 1);
- binary ``+ - * // % **``, ``comb(a, b)``, ``binom_sum(a, b)`` (= sum_{i<=b} C(a, i)),
  ``min``, ``max``; unary ``isqrt`` (x >= 0) and ``ilog2`` (x >= 1, floor);
- integer semantics only. An expression is invalid on the ns tuple if any component is
  invalid: division or mod by 0, a negative exponent or one over 30, any |value| > 10^9,
  or comb/binom_sum outside 0 <= b <= a <= 200. Invalid expressions are dropped.

Size is the node count. Each distinct full value vector keeps one representative, the
first (hence smallest) expression found for it. Entries store (size, op, child keys);
witnesses are rebuilt on demand. Keys are the value vectors packed as int32 bytes, which
keeps the table near half the memory of tuple keys.

A sequence shorter than the table's ns (a truth table cut off by a timeout) is matched on
the first len(seq) values. Caveat, by construction: an expression valid on those values
but invalid at a later n of the table is not in the table, so it cannot witness the
shorter match. The same holds for the prefix match.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import pickle
import struct
import time
from array import array
from pathlib import Path

from . import REPO

GOLF_VERSION = "golf-v1"
#: Chosen by measurement (see RESULTS.md "Run 2 scan"): cap 11 on ns 3..8 peaks at 1.84 GB.
DEFAULT_CAP = 10
MAX_ABS = 10 ** 9
MAX_EXP = 30
MAX_COMB = 200
CACHE_DIR = REPO / "cache" / "golf"
#: Occam surprise: first prequential step (k known values, predict the next) and the
#: cost of one grammar node, log2(16 symbols: 4 terminals, 10 binary, 2 unary).
OCCAM_MIN_K = 4
BITS_PER_NODE = 4.0

TERMINALS = ("n", "1", "2", "3")
COMMUTATIVE = ("+", "*", "min", "max")
NONCOMMUTATIVE = ("-", "//", "%", "**", "comb", "binom_sum")
INFIX = ("+", "-", "*", "//", "%", "**")

_COMB = [[math.comb(a, b) for b in range(a + 1)] for a in range(MAX_COMB + 1)]
_BSUM = []
for _a in range(MAX_COMB + 1):
    _acc, _row = 0, []
    for _b in range(_a + 1):
        _acc += _COMB[_a][_b]
        _row.append(_acc)
    _BSUM.append(_row)


# ---- elementwise ops: None when invalid -----------------------------------------
def _ok(v):
    return v if -MAX_ABS <= v <= MAX_ABS else None


def _add(a, b):
    return _ok(a + b)


def _sub(a, b):
    return _ok(a - b)


def _mul(a, b):
    return _ok(a * b)


def _div(a, b):
    return a // b if b else None


def _mod(a, b):
    return a % b if b else None


def _pow(a, b):
    if b < 0 or b > MAX_EXP:
        return None
    if (a > 1 or a < -1) and (abs(a).bit_length() - 1) * b > 30:
        return None  # |a|**b >= 2**31 > 10**9
    return _ok(a ** b)


def _comb(a, b):
    return _ok(_COMB[a][b]) if 0 <= b <= a <= MAX_COMB else None


def _bsum(a, b):
    return _ok(_BSUM[a][b]) if 0 <= b <= a <= MAX_COMB else None


def _min(a, b):
    return a if a < b else b


def _max(a, b):
    return a if a > b else b


def _isqrt(a):
    return math.isqrt(a) if a >= 0 else None


def _ilog2(a):
    return a.bit_length() - 1 if a >= 1 else None


BINARY_FN = {"+": _add, "-": _sub, "*": _mul, "//": _div, "%": _mod, "**": _pow,
             "comb": _comb, "binom_sum": _bsum, "min": _min, "max": _max}
UNARY_FN = {"isqrt": _isqrt, "ilog2": _ilog2}


def apply_vec(fn, a: tuple, b: tuple | None = None) -> tuple | None:
    """Apply an elementwise op; None if any component is invalid."""
    out = []
    app = out.append
    if b is None:
        for x in a:
            v = fn(x)
            if v is None:
                return None
            app(v)
    else:
        for x, y in zip(a, b):
            v = fn(x, y)
            if v is None:
                return None
            app(v)
    return tuple(out)


def table_key(ns: tuple[int, ...], cap: int) -> str:
    blob = f"{GOLF_VERSION}|{cap}|{','.join(map(str, ns))}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pack(vec) -> bytes:
    """A value vector as int32 bytes (every valid value has |v| <= 10^9 < 2^31)."""
    return array("i", vec).tobytes()


class GolfTable:
    """Bottom-up enumeration over ``ns`` up to ``size_cap`` nodes, cached on disk.

    ``entries`` maps a packed value vector to ``(size, op, key_a, key_b)``; the child keys
    are the packed vectors of other entries (None where an operand is absent).
    """

    def __init__(self, ns, size_cap: int = DEFAULT_CAP, cache_dir: Path | str | None = CACHE_DIR,
                 log=None):
        self.ns = tuple(int(n) for n in ns)
        self.cap = int(size_cap)
        self.log = log or (lambda *_: None)
        self.entries: dict[bytes, tuple] = {}
        self.level_sizes: list[int] = []
        self.build_seconds = 0.0
        self.loaded_from_cache = False
        path = Path(cache_dir) / f"{table_key(self.ns, self.cap)}.pkl" if cache_dir else None
        if path and path.exists():
            with path.open("rb") as fh:
                data = pickle.load(fh)
            self.entries, self.level_sizes = data["entries"], data["level_sizes"]
            self.build_seconds = data["build_seconds"]
            self.loaded_from_cache = True
        else:
            self._build()
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as fh:
                    pickle.dump({"ns": self.ns, "cap": self.cap, "entries": self.entries,
                                 "level_sizes": self.level_sizes,
                                 "build_seconds": self.build_seconds}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)

    # ---- build --------------------------------------------------------------------
    def _build(self) -> None:
        t0 = time.perf_counter()
        E = self.entries
        P = struct.Struct(f"{len(self.ns)}i").pack
        # levels[s] = (vectors, packed keys) of minimal size s. The top level is never
        # an operand, so its vectors are not kept.
        levels: list[tuple[list, list]] = [([], [])]
        vs, ks = [], []
        for t in TERMINALS:
            v = self.ns if t == "n" else tuple(int(t) for _ in self.ns)
            key = P(*v)
            if key not in E:
                E[key] = (1, t, None, None)
                vs.append(v)
                ks.append(key)
        levels.append((vs, ks))
        sizes = [0, len(vs)]
        for s in range(2, self.cap + 1):
            keep = s < self.cap
            vs, ks = [], []
            before = len(E)

            def emit(v, op, ka, kb):
                key = P(*v)
                if key in E:
                    return
                E[key] = (s, op, ka, kb)
                if keep:
                    vs.append(v)
                    ks.append(key)

            A, KA = levels[s - 1]
            for op, fn in UNARY_FN.items():
                for a, ka in zip(A, KA):
                    v = apply_vec(fn, a)
                    if v is not None:
                        emit(v, op, ka, None)
            for i in range(1, s - 1):
                j = s - 1 - i
                (A, KA), (B, KB) = levels[i], levels[j]
                if i <= j:
                    for op in COMMUTATIVE:
                        fn = BINARY_FN[op]
                        for ia, a in enumerate(A):
                            ka = KA[ia]
                            start = ia if i == j else 0
                            for b, kb in zip(B[start:], KB[start:]):
                                v = apply_vec(fn, a, b)
                                if v is not None:
                                    emit(v, op, ka, kb)
                for op in NONCOMMUTATIVE:
                    fn = BINARY_FN[op]
                    for a, ka in zip(A, KA):
                        for b, kb in zip(B, KB):
                            v = apply_vec(fn, a, b)
                            if v is not None:
                                emit(v, op, ka, kb)
            levels.append((vs, ks))
            sizes.append(len(E) - before)
            self.log(f"  golf ns={self.ns[0]}..{self.ns[-1]} size {s}: {sizes[-1]} new, "
                     f"{len(E)} total, {time.perf_counter() - t0:.1f}s")
        self.level_sizes = sizes
        self.build_seconds = time.perf_counter() - t0

    # ---- expressions -------------------------------------------------------------------
    def expr(self, key: bytes) -> str:
        _, op, a, b = self.entries[key]
        if a is None:
            return op
        if b is None:
            return f"{op}({self.expr(a)})"
        ea, eb = self.expr(a), self.expr(b)
        if op in INFIX:
            return f"({ea} {op} {eb})"
        return f"{op}({ea}, {eb})"

    # ---- matching ------------------------------------------------------------------
    def jump_many(self, seqs) -> list[dict]:
        """Golf many sequences in one pass over the table.

        A sequence of length L is matched on the first L values of ns. Per sequence:
        ``G_prefix`` (minimal size matching all but the last value), ``G_full`` (minimal
        size matching every value), each cap + 1 when nothing matches; ``J`` = G_full -
        G_prefix; both witnesses; and ``predicts_last``: some minimal-size prefix match
        also gets the last value right (equivalently G_full == G_prefix <= cap).
        """
        L = len(self.ns)
        states = []
        wanted: dict[int, dict[bytes, list[dict]]] = {}
        for seq in seqs:
            seq = tuple(int(x) for x in seq)
            if not 2 <= len(seq) <= L:
                raise ValueError(f"sequence of length {len(seq)} does not fit ns {self.ns}")
            k = len(seq) - 1
            st = {"pre": None, "full": None}
            if all(abs(x) <= MAX_ABS for x in seq[:k]):
                st["last"] = pack(seq[k:]) if abs(seq[k]) <= MAX_ABS else None
                wanted.setdefault(4 * k, {}).setdefault(pack(seq[:k]), []).append(st)
            states.append(st)
        if wanted:
            cuts = sorted(wanted)
            for key, ent in self.entries.items():
                size = ent[0]
                for c in cuts:
                    hits = wanted[c].get(key[:c])
                    if hits is None:
                        continue
                    for st in hits:
                        if st["pre"] is None or size < st["pre"][0]:
                            st["pre"] = (size, key)
                        if key[c:c + 4] == st["last"] and (st["full"] is None or size < st["full"][0]):
                            st["full"] = (size, key)
        out = []
        none = self.cap + 1
        for st in states:
            pre, full = st["pre"], st["full"]
            gp = pre[0] if pre else none
            gf = full[0] if full else none
            out.append({"G_full": gf, "witness_full": self.expr(full[1]) if full else None,
                        "G_prefix": gp, "witness_prefix": self.expr(pre[1]) if pre else None,
                        "predicts_last": bool(full and pre and gf == gp), "J": gf - gp})
        return out

    def jump(self, seq) -> dict:
        return self.jump_many([seq])[0]

    def golf(self, seq) -> tuple[int, str | None]:
        """(G, witness) for the whole sequence; G = cap + 1 if nothing matches."""
        seq = tuple(int(x) for x in seq)
        if len(seq) == len(self.ns):
            if any(abs(x) > MAX_ABS for x in seq):
                return self.cap + 1, None
            key = pack(seq)
            ent = self.entries.get(key)
            return (ent[0], self.expr(key)) if ent else (self.cap + 1, None)
        r = self.jump(seq)
        return r["G_full"], r["witness_full"]

    def golf_prefix(self, seq) -> dict:
        """Golf every value but the last, and whether a minimal-size prefix match
        predicts the last value."""
        r = self.jump(seq)
        return {"G": r["G_prefix"], "witness": r["witness_prefix"], "predicts_last": r["predicts_last"]}

    # ---- Occam surprise (PROTOCOL.md "Run 2": Occam surprise S) ----------------------
    def _matrix(self):
        """(values, sizes, keys): an (entries x len(ns)) int32 matrix in entry order, the
        sizes as int8, and the keys (for witnesses). Built once per table (numpy)."""
        if getattr(self, "_mat", None) is None:
            import numpy as np

            keys = list(self.entries)
            vals = np.frombuffer(b"".join(keys), dtype=np.int32).reshape(len(keys), len(self.ns))
            sizes = np.fromiter((self.entries[k][0] for k in keys), dtype=np.int8, count=len(keys))
            self._mat = (vals, sizes, keys)
        return self._mat

    def occam(self, seq, trivials, min_k: int = OCCAM_MIN_K, bits_per_node: float = BITS_PER_NODE) -> dict:
        """Prequential walk. For k = min_k .. len(seq) - 1 (0-based: the first k values
        are known, f[k] is next): G_k is the minimal size over entries equal to seq on
        the first k components, and those minimal entries predict their value at k.
        The prefix compresses iff bits_per_node * G_k < Lambda_k = sum_{i<k}
        log2(trivial_i + 1). A break: the prefix compresses and f[k] is not among the
        predicted values; it costs log2(trivial_k + 1). S = sum of break costs.
        A prefix with no matching entry (G_k = cap + 1) predicts nothing: no break."""
        import numpy as np

        vals, sizes, keys = self._matrix()
        seq = [int(x) for x in seq]
        if len(seq) > len(self.ns) or len(trivials) != len(seq):
            raise ValueError("sequence and trivial bounds must align with ns")
        costs = [math.log2(t + 1) for t in trivials]
        idx = np.arange(len(keys))
        breaks, steps = [], []
        lam = 0.0
        for k in range(len(seq)):
            if k >= min_k and idx.size:
                sz = sizes[idx]
                g = int(sz.min())
                best = idx[sz == g]
                predicted = vals[best, k]
                compresses = bits_per_node * g < lam
                hit = bool((predicted == seq[k]).any()) if abs(seq[k]) <= MAX_ABS else False
                steps.append({"k": k, "n": self.ns[k], "G": g, "Lambda": round(lam, 3),
                              "compresses": compresses, "predicted_ok": hit})
                if compresses and not hit:
                    w = int(best[0])
                    breaks.append({"k": k, "n": self.ns[k], "G": g, "cost": costs[k],
                                   "witness": self.expr(keys[w]), "predicted": int(vals[w, k]),
                                   "actual": seq[k], "Lambda": lam})
            elif k >= min_k:
                steps.append({"k": k, "n": self.ns[k], "G": self.cap + 1, "Lambda": round(lam, 3),
                              "compresses": False, "predicted_ok": False})
            lam += costs[k]
            if abs(seq[k]) > MAX_ABS:
                idx = idx[:0]
            elif idx.size:
                idx = idx[vals[idx, k] == seq[k]]
        return {"S": sum(b["cost"] for b in breaks), "breaks": breaks, "steps": steps}

    def stats(self) -> dict:
        return {"ns": list(self.ns), "cap": self.cap, "entries": len(self.entries),
                "level_sizes": self.level_sizes, "build_seconds": round(self.build_seconds, 2),
                "cached": self.loaded_from_cache}


def evaluate(expr: str, n: int) -> int | None:
    """Evaluate a golf witness at one n with the golf semantics (None if invalid)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id != "n":
                raise ValueError(f"unknown name {node.id}")
            return n
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if a is None or b is None:
                return None
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Mod: "%",
                  ast.Pow: "**"}[type(node.op)]
            return BINARY_FN[op](a, b)
        if isinstance(node, ast.Call):
            args = [ev(x) for x in node.args]
            if any(x is None for x in args):
                return None
            name = node.func.id
            return UNARY_FN[name](*args) if name in UNARY_FN else BINARY_FN[name](*args)
        raise ValueError(f"unexpected node {ast.dump(node)}")
    return ev(ast.parse(expr, mode="eval"))
