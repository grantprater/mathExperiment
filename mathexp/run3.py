"""``python -m mathexp run3``: run 3 end to end, for a one-shot cloud routine.

On a fresh clone with an empty cache it:

1. builds the truths, witnesses, clusters and A4b seed list (``clusters.json``,
   ``a4b_seeds.jsonl``, ``build.txt`` in the out directory);
2. makes one probe call and aborts with a clear message if the backend fails or
   returns a model other than the pinned one;
3. runs A5b (N0, N1, N2 on the same clusters in the same order), then A4b;
4. writes ``a5b_report.txt`` and ``a4b_report.txt`` (and each store's generic
   ``report.txt``) into the out directory.

It never commits or pushes. Model calls use the isolated ``claude -p`` form of
llm.py with the model pinned to an exact id (default ``claude-sonnet-5-5``); every
call row records the model id the CLI returned.

Budgets: A5b runs as many whole clusters as ``--budget-a5b`` covers at the nominal
cost (N0 and N1 one call per problem; N2 adds an explain call for every member but
each cluster's last child: 4s - c calls for s problems in c clusters, 228 for the
protocol's 12 clusters of 5). The default budget is a safety cap with slack for
retries, as A4b's 600 is. Each condition may spend the slack left after reserving
the later conditions' nominal calls, so retries early cannot starve N2. A4b stops at
``--max-problems`` completed problems or ``--budget-a4b`` calls.

``--phase a5b`` or ``--phase a4b`` runs one arm only and builds only what it needs
(the clusters and their witnesses, or the A4b seed list), so the two arms can run as
separate routines. run3 never resumes: it refuses an out directory whose store for
the phase already has rows, so a rerun can never double-count. ``MAX_BACKEND_ERRORS``
backend failures in a row (not parse errors) stop the run; the reports are still
written from what is stored.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from . import clusters as cl
from . import prompts
from .llm import LLMError, make_backend
from .loop import BackendDown, Config, Loop
from .problems import get_family
from .report import report
from .report3 import a4b_report, a5b_report
from .store import Store
from .truth import TruthCache

RUN3_MODEL = "claude-sonnet-5-5"
#: A5b's conditions, in run order: (label, notes, explain).
CONDITIONS = (("N0", False, False), ("N1", True, False), ("N2", True, True))
#: PROTOCOL.md sizes: A5b 228 calls nominal, here a safety cap with 25% slack for
#: retries (as A4b's 600 is for its 80 problems at 6 calls = 480).
DEFAULT_BUDGET_A5B, DEFAULT_BUDGET_A4B, DEFAULT_MAX_PROBLEMS = 285, 600, 80
PHASES = ("all", "a5b", "a4b")
#: Backend errors in a row that stop the run, and the wait before a retry (claude only).
MAX_BACKEND_ERRORS, RETRY_WAIT = 6, 30.0


class Run3Abort(RuntimeError):
    pass


def _suffix(phase: str) -> str:
    """File-name suffix for a one-arm phase, so the arms can share an out directory."""
    return "" if phase == "all" else f"-{phase}"


def check_model_id(backend_name: str, model: str) -> None:
    """The claude backend must be pinned to an exact id, never an alias such as
    'sonnet', which CLI versions resolve differently."""
    if backend_name == "claude" and not re.fullmatch(r"claude-[a-z]+-\d+(-\d+)*(-\d{8})?", model or ""):
        raise Run3Abort(f"--model must be an exact model id like {RUN3_MODEL}, not {model!r}")


def ping(backend, out: Path, log, phase: str = "all") -> dict:
    """One probe call; aborts with a clear message if it fails."""
    _, system = prompts.system_prompt()
    ver, prompt = prompts.render("ping")
    t0 = time.monotonic()
    row = {"step": "ping", "prompt_version": ver, "backend": backend.name, "requested_model": backend.model}
    try:
        raw, meta = backend.complete(system, prompt, prompts.SCHEMAS["ping"], {"step": "ping"})
        row.update({"raw": raw, "meta": meta, "model": meta.get("model")})
        prompts.parse_ping(raw)
        row["ok"] = True
    except (LLMError, prompts.ParseError, OSError) as e:
        row.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    row["wall_s"] = round(time.monotonic() - t0, 2)
    (out / f"probe_call{_suffix(phase)}.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    if not row["ok"]:
        raise Run3Abort(f"probe call to the {backend.name} backend failed: {row['error']}. "
                        f"Check that `claude -p` works and is logged in; nothing was run.")
    used = set((row.get("meta") or {}).get("model_usage") or {}) | {row.get("model")}
    if backend.name == "claude" and backend.model not in used:
        raise Run3Abort(f"probe call returned model {row.get('model')!r}, expected the pinned {backend.model!r}")
    log(f"probe call ok: model {row.get('model')} in {row['wall_s']}s")
    return row


def a5b_plan(clusters_data: dict, budget: int, limit: int | None) -> tuple[int, list[int]]:
    """(clusters to run, nominal calls per condition) for the budget."""
    sizes = [len(c["members"]) for c in sorted(clusters_data["clusters"], key=lambda c: c["order"])]
    if limit is not None:
        sizes = sizes[:limit]
    best = 0
    for c in range(len(sizes) + 1):
        s = sum(sizes[:c])
        if 4 * s - c <= budget:  # N0 s + N1 s + N2 (2s - c)
            best = c
    s = sum(sizes[:best])
    return best, [s, s, 2 * s - best]


def build(args, out: Path, cache: TruthCache, log, phase: str = "all") -> dict:
    """Build what the phase needs: A5b the clusters and their witnesses; A4b the seed
    list (and, unless ``--no-precompute-seeds``, the seeds' truths and witnesses)."""
    t0 = time.perf_counter()
    data = seeds = None
    if phase in ("all", "a5b"):
        data = cl.build_clusters(cl.SCAN_DIR, cache, parents=args.parents, children=args.children,
                                 draws=args.draws, seed=args.seed, time_limit=args.build_time_limit, log=log)
    if phase in ("all", "a4b"):
        seeds = cl.a4b_seeds(cl.SCAN_DIR, setfam=args.a4b_setfam, addfree=args.a4b_addfree)
    paths = cl.write(out, data, seeds)
    n_wit = 0
    for c in (data or {}).get("clusters", []):
        for m in c["members"]:
            fam = get_family(m["family"])
            _, w = cache.get_witness(fam, m["spec"], m["ns"][-1], args.build_time_limit)
            n_wit += w is not None
    if seeds is not None and args.precompute_seeds:
        for r in seeds:
            fam = get_family(r["family"])
            spec = {k: v for k, v in r["spec"].items() if v is not None}
            ns, vals = cl.sequence(fam, spec, r["ns"], cache, args.build_time_limit)
            if ns:
                _, w = cache.get_witness(fam, spec, ns[-1], args.build_time_limit)
                n_wit += w is not None
    cache.flush()
    text = (cl.summary_text(data, seeds) if data is not None
            else f"A4b seeds: {len(seeds)} (phase a4b: no clusters built)")
    text += f"\nwitnesses at n_max: {n_wit}\nbuild wall {time.perf_counter() - t0:.1f}s"
    (out / f"build{_suffix(phase)}.txt").write_text(text + "\n", encoding="utf-8")
    log(text)
    return {"paths": paths, "clusters": len(data["clusters"]) if data else None,
            "replacements": len(data["replacements"]) if data else None,
            "seeds": len(seeds) if seeds is not None else None, "witnesses": n_wit}


def refuse_existing(out: Path, phase: str) -> None:
    """run3 never resumes: appending to a store that already has rows, under the same
    run ids, would make the reports count both runs."""
    for arm in (("a5b", "a4b") if phase == "all" else (phase,)):
        d = out / arm
        if d.exists() and any(p.stat().st_size > 0 for p in d.glob("*.jsonl")):
            raise Run3Abort(f"{d} already has stored rows; run3 does not resume. Use a fresh --out, "
                            f"or --phase for an arm that has not run.")


def write_reports(out: Path, seed: int) -> None:
    """Each arm's reports from what its store holds (also after an abort)."""
    if (out / "a5b" / "calls.jsonl").exists():
        st = Store(out / "a5b", "report")
        (out / "a5b_report.txt").write_text(a5b_report(st) + "\n", encoding="utf-8")
        (out / "a5b" / "report.txt").write_text(report(st) + "\n", encoding="utf-8")
    if (out / "a4b" / "calls.jsonl").exists():
        st = Store(out / "a4b", "report")
        (out / "a4b_report.txt").write_text(a4b_report(st, seed=seed) + "\n", encoding="utf-8")
        (out / "a4b" / "report.txt").write_text(report(st) + "\n", encoding="utf-8")


def run_a5b(args, out: Path, backend, cache, exclude, guard: dict, summary: dict, log) -> None:
    data = cl.load_clusters(out / "clusters.json")
    n_clusters, nominal = a5b_plan(data, args.budget_a5b, args.a5b_clusters)
    log(f"== A5b: {n_clusters} clusters, nominal calls N0/N1/N2 = {nominal} = {sum(nominal)} "
        f"(cap {args.budget_a5b})")
    a5b_store = Store(out / "a5b", backend.name)
    spent, runs = 0, {}
    summary["a5b"] = {"clusters": n_clusters, "nominal": nominal, "calls": spent, "runs": runs}
    if n_clusters:
        for i, (label, notes, explain) in enumerate(CONDITIONS):
            cap = args.budget_a5b - spent - sum(nominal[i + 1:])
            cfg = Config(backend=backend, store=a5b_store, predict_only=True, clusters=str(out / "clusters.json"),
                         clusters_top=n_clusters, notes=notes, notes_scope="cluster", explain=explain,
                         max_calls=cap, run_id=f"a5b-{label}", seed=args.seed, time_limit=args.time_limit,
                         exclude=exclude, cache=cache, log=log, **guard)
            loop = Loop(cfg)
            try:
                runs[label] = loop.run_loop()
            finally:
                spent += loop.calls
                summary["a5b"]["calls"] = spent
    log(f"A5b done: {spent} calls")


def run_a4b(args, out: Path, backend, cache, exclude, guard: dict, summary: dict, log) -> None:
    a4b_store = Store(out / "a4b", backend.name)
    cfg = Config(backend=backend, store=a4b_store, policy="random", breadth="full",
                 seeds_from=str(out / "a4b_seeds.jsonl"), max_calls=args.budget_a4b,
                 max_problems=args.max_problems, run_id="a4b", seed=args.seed, time_limit=args.time_limit,
                 exclude=exclude, cache=cache, log=log, **guard)
    log(f"== A4b: up to {args.max_problems} completed problems, budget {args.budget_a4b} calls")
    loop = Loop(cfg)
    try:
        summary["a4b"] = loop.run_loop()
    finally:
        summary.setdefault("a4b", {"calls": loop.calls, "unfinished": True})
    log(f"A4b done: {summary['a4b']}")


def run3(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "run.log").open("a", encoding="utf-8")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    phase = getattr(args, "phase", "all")
    summary: dict = {}
    try:
        refuse_existing(out, phase)
        model = args.model or (RUN3_MODEL if args.backend == "claude" else None)
        check_model_id(args.backend, model)
        summary.update({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": args.backend, "model": model,
                        "seed": args.seed, "phase": phase})
        cache = TruthCache(args.cache, save_every=30)
        log(f"== run 3 ({phase}): build -> {out}")
        summary["build"] = build(args, out, cache, log, phase)
        backend = make_backend(args.backend, model, seed=args.seed)
        summary["probe"] = {k: v for k, v in ping(backend, out, log, phase).items() if k in ("ok", "model", "wall_s")}
        exclude = frozenset(cl.run2_ids(cl.SCAN_DIR))
        guard = {"retry_wait": RETRY_WAIT if backend.name == "claude" else 0.0,
                 "max_backend_errors": MAX_BACKEND_ERRORS}
        if phase in ("all", "a5b"):
            run_a5b(args, out, backend, cache, exclude, guard, summary, log)
            write_reports(out, args.seed)
        if phase in ("all", "a4b"):
            run_a4b(args, out, backend, cache, exclude, guard, summary, log)
            write_reports(out, args.seed)
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        log(f"run 3 ({phase}) finished -> {out}")
        return 0
    except BackendDown as e:
        summary["aborted"] = f"backend down: {e}"
        log(f"ABORT: the backend is down ({e}). Everything stored before this point is valid; "
            f"the reports cover it.")
        write_reports(out, args.seed)
        print(f"run3 aborted: {e}", file=sys.stderr)
        return 4
    except Run3Abort as e:
        if summary:  # a refusal before the start writes no summary over an earlier run's
            summary["aborted"] = str(e)
        log(f"ABORT: {e}")
        print(f"run3 aborted: {e}", file=sys.stderr)
        return 3
    finally:
        if summary:
            (out / f"run3_summary{_suffix(phase)}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        logf.close()
