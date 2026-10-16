"""Weave status, exactly as PROTOCOL.md "Definitions" fixes it.

- violated:     f(n) > g(n) + EPS for some computed n
- consistent:   g(n) + EPS >= f(n) for every computed n, with >= MIN_COMPUTED values
- insufficient: fewer than MIN_COMPUTED values computed (and no violation)
- error:        g failed to parse or evaluate
- tight_at:     the computed n with floor(g(n) + EPS) == f(n)
- win:          consistent and g(n_max) < trivial(n_max); ratio = g/trivial at n_max
"""

from __future__ import annotations

import math

from .problems.base import Family
from .safe_eval import SafeEvalError, evaluate

EPS = 1e-9
MIN_COMPUTED = 3


def check_bound(expr: str, family: Family, spec: dict, truth: dict[int, int | None]) -> dict:
    computed = sorted(n for n, f in truth.items() if f is not None)
    params = family.params(spec)
    values: dict[int, float] = {}
    try:
        for n in computed:
            g = evaluate(expr, {**params, "n": n})
            values[n] = float(g)
    except SafeEvalError as e:
        return {"status": "error", "error": str(e), "values": {}, "tight_at": [], "win": False,
                "ratio": None, "n_max": None, "trivial_n_max": None, "n_computed": len(computed)}
    violated_at = [n for n in computed if truth[n] > values[n] + EPS]
    tight_at = [n for n in computed if math.floor(values[n] + EPS) == truth[n]]
    if violated_at:
        status = "violated"
    elif len(computed) < MIN_COMPUTED:
        status = "insufficient"
    else:
        status = "consistent"
    out = {"status": status, "error": None, "values": {str(n): values[n] for n in computed},
           "violated_at": violated_at, "tight_at": tight_at, "win": False, "ratio": None,
           "n_max": None, "trivial_n_max": None, "n_computed": len(computed)}
    if computed:
        n_max = computed[-1]
        triv = family.trivial_bound(spec, n_max)
        out["n_max"], out["trivial_n_max"] = n_max, triv
        out["ratio"] = values[n_max] / triv if triv else None
        out["win"] = status == "consistent" and values[n_max] < triv
    return out
