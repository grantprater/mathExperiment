"""`python -m mathexp report`: calibration, surprises, weave headline and statuses, reuse,
A4, A5 (notes), the scan-vs-model side result, and the what-if tree."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from . import calibration
from .store import Store


def _filter(rows, run):
    return [r for r in rows if run is None or r.get("run") == run]


def _pct(a, b):
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else f"{a}/0"


def report(store: Store, run: str | None = None) -> str:
    preds = store.joined_predictions()
    if run:
        preds = [r for r in preds if r.get("run") == run]
    problems = _filter(store.rows("problems"), run)
    weaves = _filter(store.rows("weaves"), run)
    calls = _filter(store.rows("calls"), run)
    done = [p for p in problems if p.get("event") == "done"]
    out = [f"store {store.dir}" + (f"  run {run}" if run else "  (all runs)"),
           f"{len(done)} problems processed, {len(weaves)} weaves, {len(calls)} calls "
           f"({sum(1 for c in calls if c.get('parse_ok'))} parsed)", ""]

    out += ["== Calibration (domain x kind x level x model) ==", calibration.render_table(preds), ""]

    for unit, label in (("bits", "mean bits, -log2 P(f) (run 2)"),
                        ("log2", "mean |log2((pred+1)/(f+1))| (run 1)")):
        ranked = sorted((p for p in done if p.get("surprise") is not None
                         and p.get("surprise_unit", "log2") == unit), key=lambda p: -p["surprise"])
        has_any = any(p.get("surprise") is not None for p in done)
        if not ranked and (unit == "log2" or has_any):
            continue
        out.append(f"== Top surprises ({label}) ==")
        for p in ranked[:8]:
            out.append(f"  {p['surprise']:.2f} (max {p['surprise_max']:.2f}) {p['id']} {p['provenance']:<7} "
                       f"{p.get('seed_name') or ''} f={p.get('truth')}")
        if not ranked:
            out.append("  (none)")
        out.append("")

    out += weave_headline(weaves), ""

    out.append("== Weave status counts ==")
    for prov in PROVENANCES:
        ws = [w for w in weaves if w.get("provenance") == prov]
        if not ws and prov == "scan":
            continue
        c = Counter(w["status"] for w in ws)
        wins = sum(1 for w in ws if w.get("win"))
        tight = sum(1 for w in ws if w.get("tight_at"))
        out.append(f"  {prov:<8} n={len(ws):<4} " + " ".join(f"{s}={c.get(s, 0)}" for s in
                   ("consistent", "violated", "insufficient", "error")) + f"  wins={wins}  with_tight={tight}")
    out.append("")

    out.append("== Reuse / importance of (bridge, lemma) pairs (wins only) ==")
    pairs: dict[tuple, dict] = defaultdict(lambda: {"families": set(), "problems": set(),
                                                     "named": 0, "variant": 0, "scan": 0})
    for w in weaves:
        if not w.get("win"):
            continue
        for lem in w.get("lemma_ids") or ["-"]:
            d = pairs[(w["bridge_id"], lem)]
            d["families"].add(w["family"])
            d["problems"].add(w["problem"])
            d[w.get("provenance", "variant")] += 1
    rank = sorted(pairs.items(), key=lambda kv: (-len(kv[1]["families"]), -len(kv[1]["problems"]), kv[0]))
    for (b, lem), d in rank[:15]:
        ratios = [w["ratio"] for w in weaves if w.get("win") and w["bridge_id"] == b
                  and lem in (w.get("lemma_ids") or ["-"]) and w.get("ratio") is not None]
        best = f"{min(ratios):.3f}" if ratios else "--"
        out.append(f"  {b:<20} {lem:<26} families={len(d['families'])} problems={len(d['problems'])} "
                   f"(named {d['named']}, variant {d['variant']}, scan {d['scan']}) best g/trivial={best}")
    if not rank:
        out.append("  (no wins)")
    out.append("")

    out += a4_section(done, weaves, calls), ""
    out += a5_section(done), ""
    out += side_result(done), ""

    out.append("== What-if tree ==")
    out += what_if_tree(problems)
    return "\n".join(out)


PROVENANCES = ("named", "variant", "scan")


def weave_headline(weaves: list[dict]) -> str:
    """Run 2's headline for weaves: bounds tight at the largest computed n, and the
    median g(n_max)/trivial(n_max) over consistent bounds; wins after them."""
    lines = ["== Weave headline: tight at n_max, median g/trivial (then wins) =="]
    for prov in PROVENANCES:
        ws = [w for w in weaves if w.get("provenance") == prov]
        if not ws:
            continue
        cons = [w for w in ws if w.get("status") == "consistent"]
        tight = [w for w in cons if w.get("n_max") is not None and w["n_max"] in (w.get("tight_at") or [])]
        ratios = [w["ratio"] for w in cons if w.get("ratio") is not None]
        med = f"{statistics.median(ratios):.3f}" if ratios else "--"
        wins = sum(1 for w in ws if w.get("win"))
        lines.append(f"  {prov:<8} bounds={len(ws):<4} consistent={len(cons):<4} tight_at_n_max={len(tight):<4} "
                     f"median g/trivial={med:<6} wins={wins}")
    if len(lines) == 1:
        lines.append("  (no weaves)")
    return "\n".join(lines)


def sign_test(diffs: list[float]) -> float | None:
    """Two-sided exact sign test on the non-zero differences."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return None
    k = min(sum(1 for d in nz if d > 0), sum(1 for d in nz if d < 0))
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman's rho (Pearson on average ranks); None if undefined."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _a5_arms(done: list[dict]) -> dict[bool, list[dict]]:
    arms: dict[bool, list[dict]] = {True: [], False: []}
    for p in done:
        if p.get("predict_only") and p.get("surprise_unit") == "bits" and p.get("surprise") is not None:
            arms[bool(p.get("notes"))].append(p)
    return arms


def a5_section(done: list[dict]) -> str:
    lines = ["== A5: notes and compression progress (predict-only; bits per problem) =="]
    arms = _a5_arms(done)
    if not arms[True] and not arms[False]:
        lines.append("  (no predict-only problems)")
        return "\n".join(lines)
    per_arm: dict[bool, dict[str, float]] = {}
    for flag in (False, True):
        ps = arms[flag]
        name = "notes on " if flag else "notes off"
        if not ps:
            lines.append(f"  {name}: (none)")
            continue
        by_id: dict[str, list[float]] = defaultdict(list)
        for p in ps:
            by_id[p["id"]].append(p["surprise"])
        per_arm[flag] = {k: sum(v) / len(v) for k, v in by_id.items()}
        mean = sum(p["surprise"] for p in ps) / len(ps)
        lines.append(f"  {name}: mean bits per problem {mean:.3f} over {len(ps)} problems "
                     f"({len({p['run'] for p in ps})} runs)")
    if True in per_arm and False in per_arm:
        common = sorted(set(per_arm[True]) & set(per_arm[False]))
        diffs = [per_arm[True][k] - per_arm[False][k] for k in common]
        if diffs:
            p = sign_test(diffs)
            lines.append(f"  paired (notes - no notes) over {len(diffs)} problems: mean {sum(diffs) / len(diffs):+.3f} "
                         f"median {statistics.median(diffs):+.3f}  "
                         f"(+{sum(1 for d in diffs if d > 0)} / -{sum(1 for d in diffs if d < 0)} / "
                         f"0 {sum(1 for d in diffs if d == 0)})  sign test p = "
                         f"{'--' if p is None else f'{p:.3g}'}")
        else:
            lines.append("  paired: no problem processed in both arms")
    lines.append("  mean bits by position tercile (early / middle / late):")
    for flag in (False, True):
        ps = arms[flag]
        if not ps:
            continue
        cells = []
        by_run: dict[str, list[dict]] = defaultdict(list)
        for p in ps:
            by_run[p["run"]].append(p)
        terc: dict[int, list[float]] = defaultdict(list)
        for rps in by_run.values():
            n = max(p.get("position") or 0 for p in rps)
            for p in rps:
                pos = (p.get("position") or 1) - 1
                terc[min(2, 3 * pos // max(n, 1))].append(p["surprise"])
        for t in range(3):
            v = terc.get(t, [])
            cells.append(f"{sum(v) / len(v):.3f} (n={len(v)})" if v else "--")
        lines.append(f"    {'notes on ' if flag else 'notes off'}: " + " / ".join(cells))
    return "\n".join(lines)


def side_result(done: list[dict]) -> str:
    lines = ["== Side result: does cheap surprise predict model surprise? (A5 no-notes arm) =="]
    ps = [p for p in _a5_arms(done)[False] if p.get("scan_score") is not None]
    by_id: dict[str, dict] = {}
    for p in ps:
        d = by_id.setdefault(p["id"], {"score": p["scan_score"], "J": p.get("scan_J"), "bits": []})
        d["bits"].append(p["surprise"])
    rows = [(d["score"], d["J"], sum(d["bits"]) / len(d["bits"])) for d in by_id.values()]
    if not rows:
        lines.append("  (no scan problems in a no-notes predict-only run)")
        return "\n".join(lines)
    for i, label in ((0, "scan score J/L"), (1, "golf jump J")):
        rho = spearman([r[i] for r in rows], [r[2] for r in rows])
        lines.append(f"  Spearman({label}, mean bits) = {'--' if rho is None else f'{rho:+.3f}'}  n = {len(rows)}")
    return "\n".join(lines)


def a4_section(done: list[dict], weaves: list[dict], calls: list[dict]) -> str:
    lines = ["== A4: assume the prior, widen on surprise (trigger: 4 bits; run 1: |log2| >= 1) =="]
    done = [p for p in done if not p.get("predict_only")]
    runs_by_breadth: dict[str, set] = defaultdict(set)
    for p in done:
        runs_by_breadth[p.get("breadth", "?")].add(p["run"])
    for breadth, runs in sorted(runs_by_breadth.items()):
        n_calls = sum(1 for c in calls if c["run"] in runs)
        n_wins = sum(1 for w in weaves if w["run"] in runs and w.get("win"))
        rate = f"{100 * n_wins / n_calls:.1f}" if n_calls else "--"
        lines.append(f"  {breadth:<9} wins per 100 calls: {rate}  ({n_wins} wins / {n_calls} calls, "
                     f"{len(runs)} runs)")
    wins_by_problem: dict[str, list[dict]] = defaultdict(list)
    for w in weaves:
        if w.get("win"):
            wins_by_problem[(w["run"], w["problem"])].append(w)
    woven = {(w["run"], w["problem"]) for w in weaves}
    # only narrow problems whose narrow weave actually ran (the budget can stop a run first)
    narrow = [p for p in done if p.get("breadth") == "adaptive" and p.get("mode") == "narrow"
              and (p.get("escalated") or (p["run"], p["id"]) in woven)]
    held = [p for p in narrow if not p.get("escalated")
            and any(w.get("weave_call") == 1 for w in wins_by_problem[(p["run"], p["id"])])]
    lines.append(f"  adaptive, unsurprising (narrow): prior held {_pct(len(held), len(narrow))}"
                 f"  (escalated {sum(1 for p in narrow if p.get('escalated'))})")
    lines.append("  fixed breadth: problems with a win from a bridge outside the prior")
    fixed = [p for p in done if p.get("breadth") == "fixed" and p.get("surprising") is not None]
    for label, flag in (("surprising", True), ("unsurprising", False)):
        ps = [p for p in fixed if p["surprising"] is flag]
        hit = [p for p in ps if any(not w.get("in_prior") for w in wins_by_problem[(p["run"], p["id"])])]
        lines.append(f"    {label:<13} {_pct(len(hit), len(ps))}")
    return "\n".join(lines)


def what_if_tree(problems: list[dict]) -> list[str]:
    nodes: dict[str, dict] = {}
    children: dict[str | None, list[str]] = defaultdict(list)
    rejected = refused = 0
    for r in problems:
        ev = r.get("event")
        if ev == "rejected":
            rejected += 1
        if ev == "refused_uncheckable":
            refused += 1
        if ev == "queued" and r["id"] not in nodes:
            nodes[r["id"]] = dict(r)
            children[r.get("parent")].append(r["id"])
        elif ev == "done" and r["id"] in nodes:
            nodes[r["id"]].update({k: r.get(k) for k in ("truth", "surprise", "mode", "escalated", "surprising")})
    lines = []

    def walk(pid, depth):
        nd = nodes[pid]
        lab = nd.get("seed_name") or nd.get("what_if") or ""
        f = nd.get("truth")
        s = nd.get("surprise")
        extra = f" f={f}" if f else " (not processed)"
        extra += f" surprise={s:.2f}" if s is not None else ""
        extra += f" mode={nd['mode']}" + ("+esc" if nd.get("escalated") else "") if nd.get("mode") else ""
        lines.append("  " + "  " * depth + f"- {pid} {lab[:70]}{extra}")
        for c in children.get(pid, []):
            walk(c, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    lines.append(f"  ({rejected} mutations rejected, {refused} refused as uncheckable)")
    return lines
