"""CLI: python -m mathexp {run, report, truth, show-prompt}."""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import REPO, guard, prompts
from .llm import make_backend
from .loop import Config, Loop, NARROW_K, WIDE_K
from .problems import get_family
from .report import report
from .store import MOCK_DEFAULT, Store
from .truth import DEFAULT_TIME_LIMIT, TruthCache


def _store_path(args) -> str:
    if args.store:
        return args.store
    return str(MOCK_DEFAULT if getattr(args, "backend", None) == "mock" else REPO / "store")


def cmd_run(args) -> int:
    if args.top is not None and not args.seeds_from:
        print("--top applies to --seeds-from only", file=sys.stderr)
        return 2
    if args.seeds_from and args.seeds != "all":
        print("--seeds and --seeds-from are exclusive", file=sys.stderr)
        return 2
    if args.breadth != "fixed" and (args.bridges_per_problem is not None or args.mutations_per_problem is not None):
        print("--bridges-per-problem/--mutations-per-problem apply to --breadth fixed only", file=sys.stderr)
        return 2
    if args.breadth == "full":
        if args.policy == "surprise":
            print("--breadth full uses --policy random (surprise-first ordering selects the population on V)",
                  file=sys.stderr)
            return 2
        if not args.menu == "on":
            print("--breadth full needs --menu on (the probe and wide menus are restricted)", file=sys.stderr)
            return 2
    policy = args.policy or ("random" if args.breadth == "full" else "surprise")
    if (args.notes_scope == "cluster" or args.explain == "on" or args.clusters) and not (
            args.clusters and args.predict_only):
        print("--clusters, --notes-scope cluster and --explain on are arm A5b: they need --clusters and "
              "--predict-only", file=sys.stderr)
        return 2
    if args.explain == "on" and not (args.notes == "on" and args.notes_scope == "cluster"):
        print("--explain on needs --notes on --notes-scope cluster", file=sys.stderr)
        return 2
    if args.clusters and (args.seeds_from or args.seeds != "all"):
        print("--clusters, --seeds and --seeds-from are exclusive", file=sys.stderr)
        return 2
    exclude = frozenset()
    if args.exclude_run2:
        from .clusters import run2_ids

        exclude = frozenset(run2_ids())
    backend = make_backend(args.backend, args.model, seed=args.seed, allow_paid=args.allow_paid)
    store = Store(_store_path(args), backend.name)
    cfg = Config(backend=backend, store=store, policy=policy, feedback=args.feedback == "on",
                 menu=args.menu == "on", breadth=args.breadth,
                 k_fixed=args.bridges_per_problem or 3, m_fixed=args.mutations_per_problem or 3,
                 max_calls=args.max_calls, max_problems=args.max_problems,
                 seeds=None if args.seeds == "all" else [s.strip() for s in args.seeds.split(",") if s.strip()],
                 seeds_from=args.seeds_from, seeds_top=args.top, predict_only=args.predict_only,
                 notes=args.notes == "on", notes_scope=args.notes_scope, explain=args.explain == "on",
                 clusters=args.clusters, clusters_top=args.clusters_top, exclude=exclude,
                 run_id=args.run_id, seed=args.seed, time_limit=args.time_limit)
    summary = Loop(cfg).run_loop()
    print(json.dumps(summary))
    return 0


def cmd_report(args) -> int:
    print(report(Store(args.store, "report"), args.run))
    return 0


def cmd_report3(args) -> int:
    from .report3 import a4b_report, a5b_report

    if not args.a4b and not args.a5b:
        print("give --a4b STORE and/or --a5b STORE", file=sys.stderr)
        return 2
    if args.a5b:
        print(a5b_report(Store(args.a5b, "report")))
    if args.a4b:
        print(a4b_report(Store(args.a4b, "report"), seed=args.seed))
    return 0


def cmd_clusters(args) -> int:
    from . import clusters as cl

    cache = TruthCache(args.cache, save_every=30)
    data = cl.build_clusters(cl.SCAN_DIR, cache, parents=args.parents, children=args.children, draws=args.draws,
                             seed=args.seed, time_limit=args.time_limit)
    seeds = cl.a4b_seeds(cl.SCAN_DIR, setfam=args.a4b_setfam, addfree=args.a4b_addfree)
    paths = cl.write(args.out or cl.RUN3_DIR, data, seeds)
    print(cl.summary_text(data, seeds))
    print(json.dumps(paths))
    return 0


def cmd_run3(args) -> int:
    from .run3 import run3

    return run3(args)


def _add_build_flags(p, run3: bool = False) -> None:
    from .clusters import (DEFAULT_A4B_ADDFREE, DEFAULT_CHILDREN, DEFAULT_CLUSTER_TIME_LIMIT, DEFAULT_DRAWS,
                           DEFAULT_PARENTS)

    p.add_argument("--parents", type=int, default=DEFAULT_PARENTS, help="A5b parents per family (6)")
    p.add_argument("--children", type=int, default=DEFAULT_CHILDREN, help="A5b children per cluster (4)")
    p.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="mutate_random draws per parent (50)")
    p.add_argument("--a4b-setfam", type=int, default=None, help="A4b setfam seeds (default: every remaining one, 19)")
    p.add_argument("--a4b-addfree", type=int, default=DEFAULT_A4B_ADDFREE, help="A4b addfree seeds (20)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=None, help="truth cache path (default cache/truth.json or $MATHEXP_CACHE)")
    p.add_argument("--build-time-limit" if run3 else "--time-limit", dest="build_time_limit" if run3 else "time_limit",
                   type=float, default=DEFAULT_CLUSTER_TIME_LIMIT,
                   help="seconds per (spec, n) while building (the scan's 5)")


def cmd_truth(args) -> int:
    s = prompts.seed(args.seed)
    fam = get_family(s["family"])
    spec = dict(s["spec"])
    spec.setdefault("ns", fam.default_ns(spec))
    cache = TruthCache() if not args.no_cache else None
    print(fam.render(spec))
    total = 0.0
    for n in spec["ns"]:
        if cache:
            v, secs, cached = cache.get(fam, spec, n, args.time_limit)
        else:
            t0 = time.perf_counter()
            v = fam.truth(spec, n, args.time_limit)
            secs, cached = time.perf_counter() - t0, False
        total += 0 if cached else secs
        print(f"  n={n:<3} f={v!s:<6} trivial={fam.trivial_bound(spec, n):<6} "
              f"{secs:.3f}s{' (cached)' if cached else ''}")
    print(f"  total compute {total:.2f}s  problem_id={fam.problem_id(spec)}")
    return 0


def cmd_scan(args) -> int:
    from . import scan

    families = [args.family] if args.family else list(scan.FAMILY_ORDER)
    out = scan.run_scan(families, args.max_len, args.time_limit, args.top, args.quota,
                        out_dir=args.out or scan.SCAN_DIR, golf_cap=args.golf_cap)
    print(json.dumps({k: v for k, v in out.items() if k in families or k in ("selection", "last_wall_seconds")},
                     indent=1))
    return 0


def cmd_show_prompt(args) -> int:
    s = prompts.seed(args.seed)
    fam = get_family(s["family"])
    spec = dict(s["spec"])
    spec.setdefault("ns", fam.default_ns(spec))
    ns = ", ".join(map(str, spec["ns"]))
    statement = fam.render(spec)
    params = fam.params(spec)
    ptxt = (" (you may also use the names " + ", ".join(f"{p} = {v}" for p, v in params.items()) + ")") if params else ""
    _, system = prompts.system_prompt()
    step = args.step
    if step == "predict":
        ver, text = prompts.render("predict", problem=statement, ns=ns, feedback="", notes="")
    elif step in ("weave_menu", "weave_free"):
        mode = args.mode
        k = NARROW_K if mode == "narrow" else WIDE_K if mode == "wide" else 3
        breadth = {"fixed": f"Give up to {k} bounds, using different bridges where you can.",
                   "narrow": "Give exactly 1 bound, from the bridge you judge most promising.",
                   "wide": f"Give up to {k} bounds, using at least 3 different bridges."}[mode]
        kw = dict(problem=statement, ns=ns, k=k, params=ptxt, breadth=breadth, feedback="")
        if step == "weave_menu":
            kw.update(bridges=prompts.bridge_menu(), lemmas=prompts.lemma_menu())
        ver, text = prompts.render(step, **kw)
    elif step == "mutate":
        bare = {k: v for k, v in spec.items() if k != "ns"}
        ver, text = prompts.render("mutate", problem=statement, spec=json.dumps(bare), m=3,
                                   schema=fam.schema_text(), truth="(computed values appear here)")
    elif step in ("method", "explain"):
        cache = TruthCache()
        known = {}
        for n in spec["ns"]:
            v, _, _ = cache.get(fam, spec, n, args.time_limit)
            if v is not None:
                known[n] = v
        truth = ", ".join(f"f({n}) = {f}" for n, f in known.items())
        n_max = max(known)
        if step == "method":
            ver, text = prompts.render("method", problem=statement, truth=truth, n_max=n_max, f_max=known[n_max],
                                       bridges=prompts.bridge_menu(["charvec_fp", "double_counting"]),
                                       choose="", bridge_shape="null")
        else:
            _, w = cache.get_witness(fam, spec, n_max, args.time_limit)
            ver, text = prompts.render("explain", problem=statement, truth=truth,
                                       witness=fam.witness_text(spec, n_max, w))
    else:
        print(f"unknown step {step}; one of {prompts.STEPS}", file=sys.stderr)
        return 2
    print(f"--- system ---\n{system}\n--- {step} ({ver}) ---\n{text}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mathexp", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the weaving loop")
    r.add_argument("--backend", choices=["claude", "openrouter", "mock"], default="claude")
    r.add_argument("--model", default=None, help="claude: alias or id (default sonnet); openrouter: a ':free' id")
    r.add_argument("--allow-paid", action="store_true", help="openrouter: allow a model not ending in ':free'")
    r.add_argument("--policy", choices=["surprise", "random"], default=None,
                   help="default surprise; --breadth full uses random")
    r.add_argument("--feedback", choices=["on", "off"], default="off")
    r.add_argument("--menu", choices=["on", "off"], default="on")
    r.add_argument("--breadth", choices=["fixed", "adaptive", "full"], default="adaptive",
                   help="full = run 3's A4b (method forecast, probe, re-probe, wide)")
    r.add_argument("--bridges-per-problem", type=int, default=None, help="fixed breadth only (default 3)")
    r.add_argument("--mutations-per-problem", type=int, default=None, help="fixed breadth only (default 3)")
    r.add_argument("--max-calls", type=int, default=30)
    r.add_argument("--max-problems", type=int, default=None, help="default: no cap; --max-calls ends the run")
    r.add_argument("--seeds", default="all", help="'all' or comma-separated seed names")
    r.add_argument("--seeds-from", default=None, help="a scan selection (scan/selected.jsonl): provenance 'scan'")
    r.add_argument("--top", type=int, default=None, help="with --seeds-from: only the N best-ranked problems")
    r.add_argument("--predict-only", action="store_true",
                   help="arm A5: predict and compute only (no weave, no mutation); one call per problem")
    r.add_argument("--notes", choices=["on", "off"], default="off",
                   help="arm A5: show earlier problems of the same family and their true f in the predict prompt")
    r.add_argument("--notes-scope", choices=["family", "cluster"], default="family",
                   help="run 3 (A5b): notes on the earlier problems of the same cluster")
    r.add_argument("--explain", choices=["on", "off"], default="off",
                   help="run 3 (A5b, N2): add a written explanation to each cluster note")
    r.add_argument("--clusters", default=None, help="run 3 (A5b): clusters.json from `clusters`, in its order")
    r.add_argument("--clusters-top", type=int, default=None, help="with --clusters: only the first N clusters")
    r.add_argument("--exclude-run2", action="store_true",
                   help="run 3: refuse run 2's 40 specs (scan/selected.jsonl) as seeds and as mutants")
    r.add_argument("--store", default=None, help="default: store/ (mock: scratch/store-mock/)")
    r.add_argument("--run-id", default="auto")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, help="seconds per (problem, n)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="print the report for a store")
    rp.add_argument("--store", required=True)
    rp.add_argument("--run", default=None)
    rp.set_defaults(func=cmd_report)

    r3 = sub.add_parser("report3", help="run 3 reports: A5b and/or A4b (with tight importance)")
    r3.add_argument("--a4b", default=None, help="the A4b store")
    r3.add_argument("--a5b", default=None, help="the A5b store")
    r3.add_argument("--seed", type=int, default=0, help="bootstrap and Random-signal seed")
    r3.set_defaults(func=cmd_report3)

    c = sub.add_parser("clusters", help="run 3: build A5b's clusters and A4b's seed list (brute force)")
    _add_build_flags(c)
    c.add_argument("--out", default=None, help="output directory (default scan/run3)")
    c.set_defaults(func=cmd_clusters)

    from .run3 import DEFAULT_BUDGET_A4B, DEFAULT_BUDGET_A5B, DEFAULT_MAX_PROBLEMS, RUN3_MODEL

    x = sub.add_parser("run3", help="run 3 end to end (build, probe call, A5b, A4b, reports); never commits")
    x.add_argument("--out", required=True, help="output directory for stores, lists and reports")
    x.add_argument("--backend", choices=["claude", "mock"], default="claude")
    x.add_argument("--model", default=None, help=f"exact model id (default {RUN3_MODEL}; aliases are refused)")
    x.add_argument("--budget-a5b", type=int, default=DEFAULT_BUDGET_A5B, help="A5b call cap, retries included (228 nominal; default 285)")
    x.add_argument("--budget-a4b", type=int, default=DEFAULT_BUDGET_A4B, help="A4b calls, retries included (600)")
    x.add_argument("--max-problems", type=int, default=DEFAULT_MAX_PROBLEMS, help="A4b completed problems (80)")
    x.add_argument("--a5b-clusters", type=int, default=None, help="A5b: at most N clusters (default all built)")
    x.add_argument("--phase", choices=["all", "a5b", "a4b"], default="all",
                   help="run both arms, or one (each builds only what it needs; for separate routines)")
    x.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, help="loop seconds per (problem, n)")
    x.add_argument("--no-precompute-seeds", dest="precompute_seeds", action="store_false",
                   help="skip brute-forcing the A4b seeds during the build (the loop computes them lazily)")
    _add_build_flags(x, run3=True)
    x.set_defaults(func=cmd_run3)

    t = sub.add_parser("truth", help="compute a seed's truth table (debug)")
    t.add_argument("--seed", required=True)
    t.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    t.add_argument("--no-cache", action="store_true")
    t.set_defaults(func=cmd_truth)

    from .golf import DEFAULT_CAP
    from .scan import DEFAULT_QUOTA, DEFAULT_SCAN_TIME_LIMIT, DEFAULT_TOP

    sc = sub.add_parser("scan", help="enumerate, brute-force and golf specs; select run 2's problems")
    sc.add_argument("--family", choices=["setfam", "addfree"], default=None, help="default: both")
    sc.add_argument("--max-len", type=int, default=None,
                    help="statement-length cap (default per family: setfam 8, addfree 12)")
    sc.add_argument("--top", type=int, default=DEFAULT_TOP)
    sc.add_argument("--quota", type=int, default=DEFAULT_QUOTA, help="per-family cap in the selection")
    sc.add_argument("--time-limit", type=float, default=DEFAULT_SCAN_TIME_LIMIT, help="seconds per (spec, n)")
    sc.add_argument("--golf-cap", type=int, default=DEFAULT_CAP, help=argparse.SUPPRESS)
    sc.add_argument("--out", default=None, help=argparse.SUPPRESS)
    sc.set_defaults(func=cmd_scan)

    sp = sub.add_parser("show-prompt", help="render a prompt without calling a model")
    sp.add_argument("--step", required=True, choices=list(prompts.STEPS))
    sp.add_argument("--seed", required=True)
    sp.add_argument("--mode", choices=["fixed", "narrow", "wide"], default="fixed")
    sp.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, help="method/explain: brute force limit")
    sp.set_defaults(func=cmd_show_prompt)

    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    guard.apply()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
