"""`python -m mathexp scan`: surprise per unit of description (PROTOCOL.md "Run 2").

For each family, every valid spec in a fixed enumeration space up to a statement-length
cap is:

1. deduplicated by problem_id;
2. brute-forced on the family's scan ns, in increasing n, stopping at the first
   timeout (cached in ``cache/scan_truth.jsonl``, append-only, so a scan resumes);
3. refused if fewer than ``MIN_VALUES`` values were computed;
4. deduplicated by truth sequence: among specs with the same (ns, f) sequence only the
   shortest statement is kept (ties: the lexicographically smallest canonical JSON);
5. golfed: J = G(full) - G(all but the last value), score = J / L.

Outputs: ``scan/<family>.jsonl`` (every scored spec, with ``kept`` / ``dup_of`` for the
truth-sequence dedup), ``scan/summary.json``, and ``scan/selected.jsonl``: the top K by
score with a per-family quota, built from every ``scan/<family>.jsonl`` present.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

from . import REPO
from .golf import DEFAULT_CAP, GolfTable
from .problems import get_family
from .problems.base import Family

SCAN_DIR = REPO / "scan"
SCAN_CACHE = REPO / "cache" / "scan_truth.jsonl"
MIN_VALUES = 4
DEFAULT_SCAN_TIME_LIMIT = 5.0
DEFAULT_TOP, DEFAULT_QUOTA = 40, 20
#: Statement-length caps, chosen so each family yields a few hundred to a few thousand
#: specs (counts in RESULTS.md "Run 2 scan").
DEFAULT_MAX_LEN = {"setfam": 8, "addfree": 12}
FAMILY_ORDER = ("setfam", "addfree")

SETFAM_NS = list(range(3, 9))
INTCYC_NS = list(range(4, 31))
FPK_NS = {2: list(range(1, 7)), 3: list(range(1, 5))}


def scan_ns(fam: Family, spec: dict) -> list[int]:
    if fam.name == "setfam":
        return list(SETFAM_NS)
    if spec["group"] == "fpk":
        return list(FPK_NS[spec["p"]])
    return list(INTCYC_NS)


# ---- enumeration spaces (PROTOCOL.md "Run 2"; the brief fixes the ranges) -------------
def _proper_subsets(p: int) -> list[list[int]]:
    out = []
    for r in range(1, p):
        out += [list(c) for c in itertools.combinations(range(p), r)]
    return out


def space_setfam():
    int_abs_opts = [None] + [[a] for a in range(4)] + [list(c) for c in itertools.combinations(range(4), 2)]
    for p in (2, 3, 5):
        mods = [None] + _proper_subsets(p)
        for size_mod, int_mod, size_abs, int_abs, int_min in itertools.product(
                mods, mods, [None, [1], [2], [3], [4]], int_abs_opts, [None, 1, 2]):
            spec = {"p": p, "size_mod": size_mod, "size_abs": size_abs, "int_mod": int_mod,
                    "int_abs": int_abs, "int_min": int_min}
            if all(spec[k] is None for k in spec if k != "p"):
                continue  # at least one condition
            yield spec


def _equations(p: int | None) -> list:
    eqs = ["3ap", "sumfree", "sidon"]
    for a, b, c in itertools.product(range(1, 4), repeat=3):
        if p is not None and (a % p == 0 or b % p == 0 or c % p == 0):
            continue
        eqs.append({"a": a, "b": b, "c": c})
    return eqs


def space_addfree():
    for group, p in (("int", None), ("cyclic", None), ("fpk", 2), ("fpk", 3)):
        eqs = _equations(p)
        for i, eq in enumerate(eqs):
            for eq2 in [None] + eqs[i + 1:]:
                yield {"group": group, "p": p, "equation": eq, "equation2": eq2}


SPACES = {"setfam": space_setfam, "addfree": space_addfree}


def _canon_json(fam: Family, spec: dict) -> str:
    return json.dumps({k: v for k, v in fam.canonical(spec).items() if k != "ns"},
                      sort_keys=True, separators=(",", ":"))


def enumerate_specs(fam: Family, max_len: int) -> tuple[list[dict], dict]:
    """Valid specs up to the length cap, one per problem_id. Returns (specs, counts)."""
    seen: set[str] = set()
    out = []
    counts = {"generated": 0, "invalid": 0, "duplicate_id": 0, "over_len": 0, "enumerated": 0}
    for raw in SPACES[fam.name]():
        counts["generated"] += 1
        spec = {k: v for k, v in raw.items() if v is not None}
        if fam.validate(spec):
            counts["invalid"] += 1
            continue
        pid = fam.problem_id(spec)
        if pid in seen:
            counts["duplicate_id"] += 1
            continue
        seen.add(pid)
        if fam.statement_length(spec) > max_len:
            counts["over_len"] += 1
            continue
        out.append(spec)
    counts["enumerated"] = len(out)
    return out, counts


# ---- truth cache: append-only JSONL --------------------------------------------------
class ScanTruthCache:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else Path(os.environ.get("MATHEXP_SCAN_CACHE", SCAN_CACHE))
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.data[r["key"]] = r
        self.computed = 0
        self.compute_seconds = 0.0

    def get(self, fam: Family, spec: dict, n: int, time_limit: float) -> int | None:
        key = f"{fam.problem_id(spec)}:{n}"
        hit = self.data.get(key)
        if hit is not None and (hit["value"] is not None or hit["time_limit"] >= time_limit):
            return hit["value"]
        t0 = time.perf_counter()
        value = fam.truth(spec, n, time_limit)
        secs = time.perf_counter() - t0
        self.computed += 1
        self.compute_seconds += secs
        row = {"key": key, "family": fam.name, "value": value, "seconds": round(secs, 4),
               "time_limit": time_limit}
        self.data[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return value


def truth_sequence(fam: Family, spec: dict, cache: ScanTruthCache, time_limit: float) -> tuple[list[int], list[int]]:
    """(ns, values) computed in increasing n up to the first timeout."""
    ns, vals = [], []
    for n in scan_ns(fam, spec):
        v = cache.get(fam, spec, n, time_limit)
        if v is None:
            break
        ns.append(n)
        vals.append(v)
    return ns, vals


# ---- the scan ---------------------------------------------------------------------------
def scan_family(fam: Family, max_len: int, time_limit: float, cache: ScanTruthCache,
                golf_cap: int = DEFAULT_CAP, out_dir: Path = SCAN_DIR, log=print,
                golf_cache_dir=None) -> dict:
    t0 = time.perf_counter()
    specs, counts = enumerate_specs(fam, max_len)
    log(f"[{fam.name}] {counts['enumerated']} specs up to L={max_len} "
        f"({counts['generated']} generated, {counts['invalid']} invalid, "
        f"{counts['duplicate_id']} duplicate ids, {counts['over_len']} over the cap)")
    rows, refused = [], 0
    for i, spec in enumerate(specs):
        ns, vals = truth_sequence(fam, spec, cache, time_limit)
        if len(vals) < MIN_VALUES:
            refused += 1
            continue
        rows.append({"family": fam.name, "id": fam.problem_id(spec), "spec": spec,
                     "L": fam.statement_length(spec), "ns": ns, "truth": vals,
                     "statement": fam.render(spec), "_canon": _canon_json(fam, spec)})
        if (i + 1) % 200 == 0:
            log(f"  truth {i + 1}/{len(specs)}  computed {cache.computed} values in "
                f"{cache.compute_seconds:.0f}s  wall {time.perf_counter() - t0:.0f}s")
    truth_wall = time.perf_counter() - t0
    # truth-sequence dedup: shortest statement, then lexicographic canonical JSON
    best: dict[tuple, dict] = {}
    for r in sorted(rows, key=lambda r: (r["L"], r["_canon"])):
        k = (tuple(r["ns"]), tuple(r["truth"]))
        if k in best:
            r["kept"], r["dup_of"] = False, best[k]["id"]
        else:
            r["kept"], r["dup_of"] = True, None
            best[k] = r
    # golf, one table per ns tuple (a cut-off sequence is matched on its first values)
    tables: dict[tuple, list[dict]] = {}
    for r in rows:
        full = tuple(scan_ns(fam, r["spec"]))
        tables.setdefault(full, []).append(r)
    golf_stats = []
    gkw = {} if golf_cache_dir is None else {"cache_dir": golf_cache_dir}
    for full_ns, rs in tables.items():
        tg = time.perf_counter()
        table = GolfTable(full_ns, golf_cap, log=log, **gkw)
        res = table.jump_many([r["truth"] for r in rs])
        for r, g in zip(rs, res):
            r.update(g)
            r["score"] = r["J"] / r["L"]
        st = table.stats()
        st["match_seconds"] = round(time.perf_counter() - tg - (0 if st["cached"] else st["build_seconds"]), 2)
        st["sequences"] = len(rs)
        golf_stats.append(st)
        log(f"  golf ns={full_ns[0]}..{full_ns[-1]}: {st}")
        del table
    for r in rows:
        r.pop("_canon", None)
    rows.sort(key=lambda r: (-r["score"], -r["J"], r["L"], r["id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{fam.name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = [r for r in rows if r["kept"]]
    summary = {"family": fam.name, "max_len": max_len, "time_limit": time_limit, "golf_cap": golf_cap,
               **counts, "refused_uncheckable": refused, "scored": len(rows),
               "truth_duplicates": len(rows) - len(kept), "kept": len(kept),
               "values_computed_this_run": cache.computed,
               "compute_seconds_this_run": round(cache.compute_seconds, 1),
               "truth_wall_seconds": round(truth_wall, 1),
               "wall_seconds": round(time.perf_counter() - t0, 1), "golf_tables": golf_stats,
               "score_distribution": score_distribution(kept, golf_cap)}
    return summary


def score_distribution(rows: list[dict], cap: int = DEFAULT_CAP) -> dict:
    if not rows:
        return {}
    scores = sorted(r["score"] for r in rows)
    js: dict[int, int] = {}
    for r in rows:
        js[r["J"]] = js.get(r["J"], 0) + 1

    def q(p):
        return round(scores[min(len(scores) - 1, int(p * len(scores)))], 3)
    return {"n": len(scores), "positive": sum(1 for s in scores if s > 0), "max": round(scores[-1], 3),
            "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "J_counts": dict(sorted(js.items())),
            "G_full_at_cap_plus_1": sum(1 for r in rows if r["G_full"] > cap),
            "predicts_last": sum(1 for r in rows if r["predicts_last"])}


def select(out_dir: Path = SCAN_DIR, top: int = DEFAULT_TOP, quota: int = DEFAULT_QUOTA) -> list[dict]:
    """Top ``top`` kept specs by score, at most ``quota`` per family; written to
    ``selected.jsonl`` with a 1-based rank."""
    picked = []
    for name in FAMILY_ORDER:
        p = out_dir / f"{name}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = [r for r in rows if r["kept"]]
        rows.sort(key=lambda r: (-r["score"], -r["J"], r["L"], r["id"]))
        picked += rows[:quota]
    picked.sort(key=lambda r: (-r["score"], -r["J"], r["L"], r["family"], r["id"]))
    picked = picked[:top]
    out = []
    for i, r in enumerate(picked, 1):
        out.append({"rank": i, "family": r["family"], "id": r["id"], "score": r["score"], "J": r["J"],
                    "L": r["L"], "G_full": r["G_full"], "G_prefix": r["G_prefix"],
                    "witness_full": r["witness_full"], "witness_prefix": r["witness_prefix"],
                    "predicts_last": r["predicts_last"], "ns": r["ns"], "truth": r["truth"],
                    "spec": r["spec"], "statement": r["statement"]})
    with (out_dir / "selected.jsonl").open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def run_scan(families: list[str], max_len: dict[str, int] | int | None, time_limit: float,
             top: int, quota: int, out_dir: Path = SCAN_DIR, log=print, golf_cap: int = DEFAULT_CAP,
             cache: ScanTruthCache | None = None, golf_cache_dir=None) -> dict:
    cache = cache or ScanTruthCache()
    out_dir = Path(out_dir)
    summ_path = out_dir / "summary.json"
    summary = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}
    t0 = time.perf_counter()
    for name in families:
        fam = get_family(name)
        ml = max_len if isinstance(max_len, int) else (max_len or {}).get(name, DEFAULT_MAX_LEN[name])
        summary[name] = scan_family(fam, ml, time_limit, cache, golf_cap=golf_cap, out_dir=out_dir,
                                    log=log, golf_cache_dir=golf_cache_dir)
    sel = select(out_dir, top, quota)
    summary["selection"] = {"top": top, "quota": quota, "selected": len(sel),
                            "by_family": {f: sum(1 for r in sel if r["family"] == f) for f in FAMILY_ORDER},
                            "positive_score": sum(1 for r in sel if r["score"] > 0)}
    summary["last_wall_seconds"] = round(time.perf_counter() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
