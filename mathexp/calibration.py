"""Calibration per (domain, kind, level, model) over the joined prediction rows.

Metrics: n, n_scored, accuracy, mean confidence, gap (accuracy - mean
confidence), Brier, mean surprise in bits (distribution kind, run 2), 80% interval
coverage and median signed log2 error of the point estimate (interval kind, run 1
stores only), a runs test and a Bonferroni-corrected change-point test.

``runs``, ``change_point`` and ``_fisher`` are ported from
``C:/Users/grant/reticle/reticle/judgement.py`` (functions of the same names,
with its constants ``MIN_SIDE = 5``, ``ALPHA = 0.05`` and ``RUNS_Z = 1.64``);
that file was read, not modified.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

MIN_SIDE = 5
ALPHA = 0.05
RUNS_Z = 1.64
SCORED = ("right", "wrong")


# ---- ported from reticle/judgement.py ---------------------------------------
def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a, b], [c, d]]. Exact, so small n is fine."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, r2, c1 = a + b, c + d, a + c

    def p(x: int) -> float:
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)

    lo, hi = max(0, c1 - r2), min(r1, c1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12))


def runs(outcomes: list[int]) -> tuple[int, float, float] | None:
    """Runs test. Returns (n_runs, expected, z); z<0 means CLUSTERED."""
    n = len(outcomes)
    n1 = sum(outcomes)
    n0 = n - n1
    if n1 == 0 or n0 == 0 or n < 4:
        return None
    r = 1 + sum(1 for i in range(1, n) if outcomes[i] != outcomes[i - 1])
    exp = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n * n * (n - 1))
    if var <= 0:
        return None
    return r, exp, (r - exp) / math.sqrt(var)


def change_point(outcomes: list[int], min_side: int = MIN_SIDE):
    """Best split of a 0/1 sequence, Bonferroni-corrected for the number of splits.

    Returns None when underpowered (fewer than ``min_side`` outcomes either side).
    """
    n = len(outcomes)
    splits = [k for k in range(min_side, n - min_side + 1)]
    if not splits:
        return None
    best = None
    for k in splits:
        lo, hi = outcomes[:k], outcomes[k:]
        a, b = sum(lo), len(lo) - sum(lo)
        c, d = sum(hi), len(hi) - sum(hi)
        p = _fisher(a, b, c, d)
        if best is None or p < best["p_raw"]:
            best = {"k": k, "p_raw": p,
                    "before": sum(lo) / len(lo), "after": sum(hi) / len(hi),
                    "n_before": len(lo), "n_after": len(hi)}
    best["n_splits"] = len(splits)
    best["p"] = min(1.0, best["p_raw"] * len(splits))
    best["changed"] = best["p"] <= ALPHA
    return best


# ---- this experiment ---------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    """Metrics for one group of joined prediction rows (file order = time order)."""
    scored = [r for r in rows if r.get("outcome") in SCORED]
    o = [1 if r["outcome"] == "right" else 0 for r in scored]
    conf = [float(r["confidence"]) for r in scored]
    out = {"n": len(rows), "n_scored": len(o),
           "acc": sum(o) / len(o) if o else None,
           "mean_conf": sum(conf) / len(conf) if conf else None,
           "brier": sum((c - y) ** 2 for c, y in zip(conf, o)) / len(o) if o else None,
           "runs": runs(o), "change": change_point(o)}
    out["gap"] = out["acc"] - out["mean_conf"] if o else None
    intervals = [r for r in scored if r.get("kind") == "interval"]
    out["coverage80"] = (sum(1 for r in intervals if r["outcome"] == "right") / len(intervals)
                         if intervals else None)
    errs = [r["log2_err"] for r in rows if r.get("log2_err") is not None]
    out["median_log2_err"] = statistics.median(errs) if errs else None
    b = [r["bits"] for r in rows if r.get("bits") is not None]
    out["mean_bits"] = sum(b) / len(b) if b else None
    return out


def table(joined: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in joined:
        groups[(r.get("domain"), r.get("kind"), r.get("level"), r.get("by"))].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))}


def _f(x, fmt="{:.2f}"):
    return "--" if x is None else fmt.format(x)


def render_table(joined: list[dict]) -> str:
    lines = [f"{'domain':<8} {'kind':<16} {'level':<9} {'model':<20} {'n':>4} {'scr':>4} "
             f"{'acc':>5} {'conf':>5} {'gap':>6} {'brier':>6} {'cov80':>6} {'medlog2':>8} "
             f"{'bits':>6} {'runs z':>7} {'chg p':>6}"]
    for (dom, kind, level, by), m in table(joined).items():
        rz = f"{m['runs'][2]:+.2f}" if m["runs"] else "--"
        cp = _f(m["change"]["p"]) if m["change"] else "--"
        lines.append(f"{str(dom):<8} {str(kind):<16} {str(level):<9} {str(by)[:20]:<20} "
                     f"{m['n']:>4} {m['n_scored']:>4} {_f(m['acc']):>5} {_f(m['mean_conf']):>5} "
                     f"{_f(m['gap'], '{:+.2f}'):>6} {_f(m['brier'], '{:.3f}'):>6} "
                     f"{_f(m['coverage80']):>6} {_f(m['median_log2_err'], '{:+.2f}'):>8} "
                     f"{_f(m['mean_bits']):>6} {rz:>7} {cp:>6}")
    return "\n".join(lines)


def feedback_text(joined: list[dict], domain: str, model: str) -> str:
    """Arm A2: this family's running calibration for the same model, for the prompt."""
    rows = [r for r in joined if r.get("domain") == domain and r.get("by") == model]
    if not rows:
        return "No scored predictions yet for this problem family."
    lines = []
    for kind in ("distribution", "top1", "bound_holds", "mutation_effect", "recognition"):
        m = summarize([r for r in rows if r.get("kind") == kind])
        if not m["n_scored"]:
            continue
        if kind == "distribution":
            lines.append(f"- value distributions: {m['n_scored']} scored, mean surprise "
                         f"{_f(m['mean_bits'])} bits per value (-log2 of the probability you gave the "
                         f"true value); the true value was among your candidates {m['acc']:.0%} of the "
                         f"time, against a stated candidate mass of {m['mean_conf']:.0%}.")
        else:
            lines.append(f"- {kind}: {m['n_scored']} scored, stated mean confidence {m['mean_conf']:.2f}, "
                         f"actual accuracy {m['acc']:.2f}, Brier {m['brier']:.3f}.")
    return "\n".join(lines) if lines else "No scored predictions yet for this problem family."
