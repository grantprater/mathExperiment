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
    if args.breadth == "adaptive" and (args.bridges_per_problem is not None or args.mutations_per_problem is not None):
        print("--bridges-per-problem/--mutations-per-problem apply to --breadth fixed only", file=sys.stderr)
        return 2
    backend = make_backend(args.backend, args.model, seed=args.seed, allow_paid=args.allow_paid)
    store = Store(_store_path(args), backend.name)
    cfg = Config(backend=backend, store=store, policy=args.policy, feedback=args.feedback == "on",
                 menu=args.menu == "on", breadth=args.breadth,
                 k_fixed=args.bridges_per_problem or 3, m_fixed=args.mutations_per_problem or 3,
                 max_calls=args.max_calls, max_problems=args.max_problems,
                 seeds=None if args.seeds == "all" else [s.strip() for s in args.seeds.split(",") if s.strip()],
                 run_id=args.run_id, seed=args.seed, time_limit=args.time_limit)
    summary = Loop(cfg).run_loop()
    print(json.dumps(summary))
    return 0


def cmd_report(args) -> int:
    print(report(Store(args.store, "report"), args.run))
    return 0


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
        ver, text = prompts.render("predict", problem=statement, ns=ns, feedback="")
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
    r.add_argument("--policy", choices=["surprise", "random"], default="surprise")
    r.add_argument("--feedback", choices=["on", "off"], default="off")
    r.add_argument("--menu", choices=["on", "off"], default="on")
    r.add_argument("--breadth", choices=["fixed", "adaptive"], default="adaptive")
    r.add_argument("--bridges-per-problem", type=int, default=None, help="fixed breadth only (default 3)")
    r.add_argument("--mutations-per-problem", type=int, default=None, help="fixed breadth only (default 3)")
    r.add_argument("--max-calls", type=int, default=30)
    r.add_argument("--max-problems", type=int, default=None, help="default: no cap; --max-calls ends the run")
    r.add_argument("--seeds", default="all", help="'all' or comma-separated seed names")
    r.add_argument("--store", default=None, help="default: store/ (mock: scratch/store-mock/)")
    r.add_argument("--run-id", default="auto")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT, help="seconds per (problem, n)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="print the report for a store")
    rp.add_argument("--store", required=True)
    rp.add_argument("--run", default=None)
    rp.set_defaults(func=cmd_report)

    t = sub.add_parser("truth", help="compute a seed's truth table (debug)")
    t.add_argument("--seed", required=True)
    t.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    t.add_argument("--no-cache", action="store_true")
    t.set_defaults(func=cmd_truth)

    sp = sub.add_parser("show-prompt", help="render a prompt without calling a model")
    sp.add_argument("--step", required=True, choices=list(prompts.STEPS))
    sp.add_argument("--seed", required=True)
    sp.add_argument("--mode", choices=["fixed", "narrow", "wide"], default="fixed")
    sp.set_defaults(func=cmd_show_prompt)

    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    guard.apply()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
