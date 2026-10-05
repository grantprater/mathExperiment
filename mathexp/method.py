"""Run 3, arm A4b: method surprise. Constants and pure helpers shared by the loop
and the report (PROTOCOL.md "A4b: widen on method surprise").

Bands, for r = g(n_max) / f(n_max) of the first probe's best consistent bound:
tight r = 1; close 1 < r <= 1.25; moderate 1.25 < r <= 2; loose r > 2; failed: no
consistent bound. The cut points come from run 2's fixed arm and are constants here,
not flags. g is floored before the ratio: f is an integer, so g(n_max) = 7.5 bounds
f(n_max) by 7, and the checker counts it as tight at n_max (checking.py ``tight_at``).
"""

from __future__ import annotations

import math

from .checking import EPS

BANDS = ("tight", "close", "moderate", "loose", "failed")
#: Upper edges of the tight, close and moderate bands (fixed by PROTOCOL.md "Run 3").
BAND_CUTS = (1.0, 1.25, 2.0)
#: The binary method-surprise trigger: the observed band had probability <= 1/8.
METHOD_SURPRISE_BITS = 3.0
#: -log2 of a zero probability scores this (as run 2's value surprise).
MAX_BITS = 30.0

#: Bridges that give the same bound under different names count as one group.
BRIDGE_GROUPS = (frozenset({"charvec_fp", "multilinear_poly", "incidence_matrix"}),
                 frozenset({"fourier_indicator", "slice_rank"}))

#: A4b sizes (PROTOCOL.md): P has at most 2 bridges; probe k = 1; wide k = 3; m = 3.
P_CAP, PROBE_K, WIDE3_K, FULL_M = 2, 1, 3, 3


def bridge_group(bridge: str) -> frozenset:
    for g in BRIDGE_GROUPS:
        if bridge in g:
            return g
    return frozenset({bridge})


def excluded_bridges(prior: list[str]) -> set[str]:
    """Every bridge in a group that contains a P bridge."""
    out: set[str] = set()
    for b in prior:
        out |= bridge_group(b)
    return out


def g_at_nmax(w: dict) -> float | None:
    """g(n_max) of a checked bound (weaves.jsonl row), floored; None if not consistent."""
    if w.get("status") != "consistent" or w.get("n_max") is None:
        return None
    g = (w.get("values") or {}).get(str(w["n_max"]))
    return None if g is None else math.floor(g + EPS)


def ratio(g_floor: float, f: int) -> float:
    """r = floor(g(n_max)) / f(n_max); f = 0 gives 1 if g < 1 and infinity otherwise."""
    if f > 0:
        return g_floor / f
    return 1.0 if g_floor <= 0 else math.inf


def band_of(r: float | None) -> str:
    if r is None:
        return "failed"
    if r <= BAND_CUTS[0]:
        return "tight"
    if r <= BAND_CUTS[1]:
        return "close"
    if r <= BAND_CUTS[2]:
        return "moderate"
    return "loose"


def band_bits(dist: dict[str, float], band: str) -> tuple[float, float]:
    """(M, p): M = -log2 P(band), capped at MAX_BITS."""
    p = float(dist.get(band, 0.0))
    return (MAX_BITS if p <= 0 else min(MAX_BITS, -math.log2(p))), p


def median_band(dist: dict[str, float]) -> str:
    """The forecast's median band, in order best (tight) to worst (failed)."""
    acc = 0.0
    for b in BANDS:
        acc += dist.get(b, 0.0)
        if acc >= 0.5 - 1e-12:
            return b
    return BANDS[-1]


def direction(dist: dict[str, float], observed: str) -> str:
    """'disappointment' if the observed band is worse than the forecast's median band,
    'pleasant' if better, 'as_expected' if equal."""
    d = BANDS.index(observed) - BANDS.index(median_band(dist))
    return "disappointment" if d > 0 else "pleasant" if d < 0 else "as_expected"


def best_probe(weaves: list[dict], f_nmax: int | None) -> dict:
    """Summary of one probe's bounds: best (lowest) floored g(n_max) over consistent
    bounds, its r (infinite only when f(n_max) = 0 < g) and band, whether any of its
    bounds is a win, and whether the best is tight at n_max."""
    win = any(w.get("win") for w in weaves)
    cons = [(g_at_nmax(w), w) for w in weaves]
    cons = [(g, w) for g, w in cons if g is not None]
    if not cons or f_nmax is None:
        return {"best_g": None, "r": None, "band": "failed", "win": win, "tight": False, "bridge": None}
    g, w = min(cons, key=lambda x: x[0])
    r = ratio(g, f_nmax)
    return {"best_g": g, "r": r, "band": band_of(r), "win": win, "tight": band_of(r) == "tight",
            "bridge": w.get("bridge_id")}
