"""`python -m mathexp report`: calibration, surprises, weave statuses, reuse, A4, what-if tree."""

from __future__ import annotations

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

    out.append("== Top surprises (mean |log2((pred+1)/(f+1))|) ==")
    ranked = sorted((p for p in done if p.get("surprise") is not None), key=lambda p: -p["surprise"])
    for p in ranked[:8]:
        out.append(f"  {p['surprise']:.2f} (max {p['surprise_max']:.2f}) {p['id']} {p['provenance']:<7} "
                   f"{p.get('seed_name') or ''} f={p.get('truth')}")
    if not ranked:
        out.append("  (none)")
    out.append("")

    out.append("== Weave status counts ==")
    for prov in ("named", "variant"):
        ws = [w for w in weaves if w.get("provenance") == prov]
        c = Counter(w["status"] for w in ws)
        wins = sum(1 for w in ws if w.get("win"))
        tight = sum(1 for w in ws if w.get("tight_at"))
        out.append(f"  {prov:<8} n={len(ws):<4} " + " ".join(f"{s}={c.get(s, 0)}" for s in
                   ("consistent", "violated", "insufficient", "error")) + f"  wins={wins}  with_tight={tight}")
    out.append("")

    out.append("== Reuse / importance of (bridge, lemma) pairs (wins only) ==")
    pairs: dict[tuple, dict] = defaultdict(lambda: {"families": set(), "problems": set(),
                                                     "named": 0, "variant": 0})
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
                   f"(named {d['named']}, variant {d['variant']}) best g/trivial={best}")
    if not rank:
        out.append("  (no wins)")
    out.append("")

    out += a4_section(done, weaves, calls), ""

    out.append("== What-if tree ==")
    out += what_if_tree(problems)
    return "\n".join(out)


def a4_section(done: list[dict], weaves: list[dict], calls: list[dict]) -> str:
    lines = ["== A4: assume the prior, widen on surprise =="]
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
    narrow = [p for p in done if p.get("breadth") == "adaptive" and p.get("mode") == "narrow"]
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
    rejected = 0
    for r in problems:
        ev = r.get("event")
        if ev == "rejected":
            rejected += 1
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
    lines.append(f"  ({rejected} mutations rejected)")
    return lines
