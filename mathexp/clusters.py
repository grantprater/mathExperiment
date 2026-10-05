"""Run 3 problem lists, built before any model call (PROTOCOL.md "Run 3").

``python -m mathexp clusters`` writes two files:

- ``clusters.json`` -- arm A5b's clusters. A parent is one of the top ``parents``
  eligible specs per family by scan score, excluding run 2's 40 (``scan/selected.jsonl``).
  Children are drawn with ``mutate_random`` (one field changed, seeded per parent) until
  ``children`` qualify or ``draws`` draws pass. A child qualifies if it is checkable
  (at least 4 values), has an interaction (f is not the trivial bound everywhere), its f
  sequence differs from the parent's and its siblings', and it is neither one of run 2's
  40 nor a member of another cluster. A parent with too few qualifying children is
  replaced by the next eligible spec, and the replacement is listed. The cluster order
  is seeded: clusters in random order, and within a cluster the parent, then the
  children in draw order.
- ``a4b_seeds.jsonl`` -- arm A4b's seeds, in the scan-selection format that
  ``run --seeds-from`` reads: the remaining eligible setfam specs, then the next
  addfree specs by scan rank.

"Eligible" is run 2's rule: kept after the truth-sequence dedup, with S > 0, ranked by
score (ties: S, then L, then id). Every value computed here goes through the truth
cache, which also keeps the extremal witness at every n.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

from .problems import get_family
from .problems.base import Family, changed_fields
from .scan import FAMILY_ORDER, MIN_VALUES, SCAN_DIR, scan_ns
from .truth import TruthCache

#: PROTOCOL.md "Run 3", A5b: 6 parents per family, 4 children each, at most 50 draws.
DEFAULT_PARENTS, DEFAULT_CHILDREN, DEFAULT_DRAWS = 6, 4, 50
#: A4b seeds: every remaining eligible setfam spec (19), then the next 20 addfree specs.
DEFAULT_A4B_SETFAM, DEFAULT_A4B_ADDFREE = None, 20
#: Seconds per (spec, n): the scan's limit, so values match the scan's.
DEFAULT_CLUSTER_TIME_LIMIT = 5.0
RUN3_DIR = SCAN_DIR / "run3"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run2_ids(scan_dir: Path = SCAN_DIR) -> set[str]:
    """The 40 specs run 2 used (scan/selected.jsonl); run 3 excludes them everywhere."""
    return {r["id"] for r in _rows(Path(scan_dir) / "selected.jsonl")}


def eligible(family: str, scan_dir: Path = SCAN_DIR, exclude: set[str] | None = None) -> list[dict]:
    """Kept specs with S > 0, best first; ``family_rank`` counts every eligible spec
    (run 2's included), so "the next 20 by scan rank" is readable off it."""
    rows = [r for r in _rows(Path(scan_dir) / f"{family}.jsonl") if r["kept"] and r["S"] > 0]
    rows.sort(key=lambda r: (-r["score"], -r["S"], r["L"], r["id"]))
    for i, r in enumerate(rows, 1):
        r["family_rank"] = i
    return [r for r in rows if not exclude or r["id"] not in exclude]


def member_ns(fam: Family, spec: dict) -> list[int]:
    """The scan's ns for a spec (F_5^k and F_7^k, outside the scan, use default_ns)."""
    try:
        return scan_ns(fam, spec)
    except KeyError:
        return fam.default_ns(spec)


def sequence(fam: Family, spec: dict, ns: list[int], cache: TruthCache, time_limit: float):
    """(ns, values) in increasing n up to the first timeout, as the scan computes them."""
    out_ns, vals = [], []
    for n in sorted(ns):
        v, _, _ = cache.get(fam, spec, n, time_limit)
        if v is None:
            break
        out_ns.append(n)
        vals.append(v)
    return out_ns, vals


def same_sequence(a: dict, b: dict) -> bool:
    """Two members have the same f sequence if they agree on every common n (and share
    at least one): a sequence cut short by a timeout is the same sequence."""
    ta = dict(zip(a["ns"], a["truth"]))
    tb = dict(zip(b["ns"], b["truth"]))
    common = set(ta) & set(tb)
    return bool(common) and all(ta[n] == tb[n] for n in common)


def _member(fam: Family, spec: dict, ns: list[int], vals: list[int], role: str, **extra) -> dict:
    bare = {k: v for k, v in spec.items() if k != "ns" and v is not None}
    return {"role": role, "family": fam.name, "id": fam.problem_id(bare), "spec": {**bare, "ns": list(ns)},
            "ns": list(ns), "truth": list(vals), "trivial": [fam.trivial_bound(bare, n) for n in ns],
            "statement": fam.render(bare), "L": fam.statement_length(bare), **extra}


def build_clusters(scan_dir: Path = SCAN_DIR, cache: TruthCache | None = None,
                   parents: int = DEFAULT_PARENTS, children: int = DEFAULT_CHILDREN,
                   draws: int = DEFAULT_DRAWS, seed: int = 0,
                   time_limit: float = DEFAULT_CLUSTER_TIME_LIMIT,
                   families: tuple[str, ...] = FAMILY_ORDER, log=print) -> dict:
    t0 = time.perf_counter()
    cache = cache or TruthCache()
    excluded = run2_ids(scan_dir)
    used: set[str] = set()          # every member of a built cluster
    clusters, replacements, skipped = [], [], []
    for name in families:
        fam = get_family(name)
        built = 0
        for row in eligible(name, scan_dir, excluded):
            if built >= parents:
                break
            if row["id"] in used:
                skipped.append({"family": name, "id": row["id"], "family_rank": row["family_rank"],
                                "reason": "already a child in another cluster"})
                continue
            pspec = {k: v for k, v in row["spec"].items() if v is not None}
            pns, pvals = sequence(fam, pspec, row["ns"], cache, time_limit)
            if pvals != row["truth"][:len(pvals)]:
                raise RuntimeError(f"parent {row['id']}: brute force {pvals} disagrees with the scan {row['truth']}")
            if len(pvals) < MIN_VALUES:
                replacements.append({"family": name, "replaced": row["id"], "family_rank": row["family_rank"],
                                     "qualified": 0, "draws": 0,
                                     "reason": f"parent has only {len(pvals)} values within {time_limit}s"})
                continue
            parent = _member(fam, pspec, pns, pvals, "parent", scan_rank=row["family_rank"],
                             scan_score=row["score"], scan_S=row["S"], scan_J=row.get("J"), scan_L=row["L"])
            rng = random.Random(f"{seed}:{row['id']}")
            kids: list[dict] = []
            drawn: set[str] = set()
            reasons: Counter = Counter()
            d = 0
            while len(kids) < children and d < draws:
                d += 1
                try:
                    child = fam.mutate_random({**pspec, "ns": list(pns)}, rng)
                except RuntimeError:
                    reasons["no_mutation"] += 1
                    continue
                bare = {k: v for k, v in child.items() if k != "ns" and v is not None}
                cid = fam.problem_id(bare)
                if cid in drawn:
                    reasons["repeat_draw"] += 1
                    continue
                drawn.add(cid)
                if cid in excluded:
                    reasons["run2_spec"] += 1
                    continue
                if cid in used or cid == parent["id"]:
                    reasons["in_another_cluster"] += 1
                    continue
                cns, cvals = sequence(fam, bare, member_ns(fam, bare), cache, time_limit)
                if len(cvals) < MIN_VALUES:
                    reasons["uncheckable"] += 1
                    continue
                if cvals == [fam.trivial_bound(bare, n) for n in cns]:
                    reasons["no_interaction"] += 1
                    continue
                m = _member(fam, bare, cns, cvals, "child", draw=d,
                            changed=changed_fields(fam.normalize(pspec), fam.normalize(bare)))
                if any(same_sequence(m, o) for o in [parent, *kids]):
                    reasons["same_sequence"] += 1
                    continue
                kids.append(m)
            if len(kids) < children:
                replacements.append({"family": name, "replaced": row["id"], "family_rank": row["family_rank"],
                                     "qualified": len(kids), "draws": d, "rejections": dict(reasons),
                                     "reason": f"{len(kids)} of {children} children qualified in {d} draws"})
                log(f"  [{name}] parent {row['id']} (rank {row['family_rank']}) replaced: "
                    f"{len(kids)} children in {d} draws {dict(reasons)}")
                continue
            used.add(parent["id"])
            used.update(k["id"] for k in kids)
            clusters.append({"cluster": f"{name}-{parent['id']}", "family": name, "draws": d,
                             "rejections": dict(reasons), "members": [parent, *kids]})
            built += 1
            log(f"  [{name}] cluster {built}: parent {parent['id']} (rank {row['family_rank']}) "
                f"children {[k['id'] for k in kids]} in {d} draws")
        if built < parents:
            log(f"  [{name}] only {built} of {parents} clusters could be built")
    random.Random(seed).shuffle(clusters)
    for i, c in enumerate(clusters, 1):
        c["order"] = i
        for pos, m in enumerate(c["members"], 1):
            m["cluster"], m["cluster_pos"], m["cluster_size"] = c["cluster"], pos, len(c["members"])
    cache.flush()
    return {"version": 1, "seed": seed, "parents_per_family": parents, "children_per_cluster": children,
            "draws": draws, "time_limit": time_limit, "families": list(families),
            "excluded_run2": sorted(excluded), "clusters": clusters, "replacements": replacements,
            "skipped": skipped, "wall_seconds": round(time.perf_counter() - t0, 1)}


def a4b_seeds(scan_dir: Path = SCAN_DIR, setfam: int | None = DEFAULT_A4B_SETFAM,
              addfree: int | None = DEFAULT_A4B_ADDFREE) -> list[dict]:
    """A4b's seeds in the scan-selection format: the remaining eligible setfam specs
    (``setfam`` of them; None = all), then the next ``addfree`` addfree specs by scan
    rank. Ranks run 1..N in that order."""
    excluded = run2_ids(scan_dir)
    picked = eligible("setfam", scan_dir, excluded)[:setfam] + eligible("addfree", scan_dir, excluded)[:addfree]
    out = []
    for i, r in enumerate(picked, 1):
        out.append({"rank": i, "family": r["family"], "id": r["id"], "family_rank": r["family_rank"],
                    "score": r["score"], "S": r["S"], "J": r["J"], "L": r["L"], "ns": r["ns"],
                    "truth": r["truth"], "spec": r["spec"], "statement": r["statement"]})
    return out


def write(out_dir: Path, clusters: dict | None, seeds: list[dict] | None) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if clusters is not None:
        p = out_dir / "clusters.json"
        p.write_text(json.dumps(clusters, indent=1, ensure_ascii=False), encoding="utf-8")
        paths["clusters"] = str(p)
    if seeds is not None:
        p = out_dir / "a4b_seeds.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in seeds:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths["a4b_seeds"] = str(p)
    return paths


def load_clusters(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_text(cl: dict, seeds: list[dict] | None = None) -> str:
    lines = [f"clusters: {len(cl['clusters'])} built (seed {cl['seed']}, {cl['parents_per_family']} parents per "
             f"family, {cl['children_per_cluster']} children, at most {cl['draws']} draws; run 2's "
             f"{len(cl['excluded_run2'])} specs excluded)"]
    for c in cl["clusters"]:
        p = c["members"][0]
        lines.append(f"  {c['order']:>2}. {c['cluster']} (scan rank {p.get('scan_rank')}, {c['draws']} draws): "
                     + ", ".join(m["id"] for m in c["members"]))
    lines.append(f"replacements: {len(cl['replacements'])}")
    for r in cl["replacements"]:
        lines.append(f"  {r['family']} {r['replaced']} (rank {r['family_rank']}): {r['reason']}")
    for s in cl.get("skipped", []):
        lines.append(f"  skipped {s['family']} {s['id']} (rank {s['family_rank']}): {s['reason']}")
    if seeds is not None:
        by = Counter(r["family"] for r in seeds)
        lines.append(f"A4b seeds: {len(seeds)} ({', '.join(f'{k} {v}' for k, v in by.items())})")
    return "\n".join(lines)
