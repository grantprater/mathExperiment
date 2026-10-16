"""The weaving loop: predict -> compute -> weave -> check -> what-if, per problem.

Order inside one problem (PROTOCOL.md "One iteration", with run 2's bits):

1. predict a distribution over f(n) for each n (+ recognition) -- written to the
   store first, as a ``distribution`` row and a ``top1`` row per n;
2. compute f(n) by brute force, only for n whose value prediction is on disk
   (refusal (a)), and score the predictions in bits; score pending mutation-effect
   predictions whose child is this problem;
3. choose the breadth (arm A4) from the surprise and the prior bridges;
4. weave (one call; adaptive narrow mode may escalate to a second, wide call),
   writing each bound's P(holds) before checking it;
5. propose what-if mutations, validate them (refusal (c)), refuse uncheckable ones
   (fewer than ``MIN_CHECKABLE`` values) and queue the rest.

``--predict-only`` (arm A5) stops after step 2.

Run 3 adds two modes (PROTOCOL.md "Run 3"):

- ``--breadth full`` (arm A4b) replaces steps 3-4: fix the prior P (at most 2 bridges),
  store a ``method_band`` forecast from a separate call that sees the true values (the
  probe is refused until it is on disk), then a probe and a re-probe (k = 1, menu
  restricted to P) and a wide weave (k = 3, menu excluding every bridge group that
  contains a P bridge; a bound labelled with an excluded bridge is recorded but not
  counted), then m = 3 mutations under ``--policy random``.
- ``--notes-scope cluster`` (arm A5b, predict-only, problems from ``--clusters``): the
  notes are the earlier problems of the same cluster; with ``--explain on`` each note
  also carries an explanation written by a separate call that sees the statement, the
  true values and the extremal witness at n_max. The explain call is skipped for each
  cluster's last child, whose note is never shown.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import random
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import calibration, method, prompts
from .checking import check_bound
from .llm import Backend, LLMError
from .problems import get_family, validate_mutation
from .problems.base import Family
from .store import Refusal, Store, new_id, now
from .truth import DEFAULT_TIME_LIMIT, TruthCache

#: A problem is *surprising* iff for some n (f known) the true value got at most
#: 2**-4 probability: surprise >= 4 bits. Fixed by PROTOCOL.md "Run 2"; not a flag.
SURPRISING_BITS = 4.0
#: Surprise at one n is capped here (a zero-probability value scores this).
MAX_BITS = 30.0
#: A mutated child is queued only if brute force yields at least this many values.
MIN_CHECKABLE = 4
#: Arm A5: at most this many notes (earlier problems of the same family) in the prompt.
NOTES_CAP = 12



class BackendDown(RuntimeError):
    """``max_backend_errors`` backend failures in a row (not parse errors): the backend
    is down or out of quota, so the run stops instead of spending its budget on
    failures. Everything already stored stays valid."""


#: Arm A4 breadths.
NARROW_K, NARROW_M = 1, 1
WIDE_K, WIDE_M, WIDE_MIN_OUTSIDE = 5, 5, 3


def bits(dist: dict, f: int, trivial: int) -> tuple[float, float]:
    """(bits, p_true) for one n, PROTOCOL.md "Model surprise in bits".

    P(f) is the candidate's mass if f was listed; otherwise the residual mass
    ``other`` shared equally over the integers in [0, trivial] that were not listed.
    Surprise = -log2 P(f), capped at MAX_BITS (P = 0 scores the cap)."""
    p_true = None
    for v, pr in dist["candidates"]:
        if v == f:
            p_true = pr
    if p_true is None:
        listed = {v for v, _ in dist["candidates"] if 0 <= v <= trivial}
        rest = trivial + 1 - len(listed)
        p_true = dist["other"] / rest if rest > 0 and 0 <= f <= trivial else 0.0
    b = MAX_BITS if p_true <= 0 else min(MAX_BITS, max(0.0, -math.log2(p_true)))
    return b, p_true


def surprise_stats(bits_by_n: dict[int, float]) -> tuple[float | None, float | None]:
    """(mean, max) of the per-n bits; (None, None) if no n was scored."""
    if not bits_by_n:
        return None, None
    vals = list(bits_by_n.values())
    return sum(vals) / len(vals), max(vals)


def is_surprising(surprise_max: float | None) -> bool:
    return surprise_max is not None and surprise_max >= SURPRISING_BITS


def bridge_importance(weaves: list[dict], model: str | None = None) -> dict[str, dict]:
    """Per bridge: the families and problems in which it has a win."""
    out: dict[str, dict] = {}
    for w in weaves:
        if not w.get("win") or (model is not None and w.get("by") != model):
            continue
        d = out.setdefault(w["bridge_id"], {"families": set(), "problems": set()})
        d["families"].add(w["family"])
        d["problems"].add(w["problem"])
    return out


def prior_bridges(store: Store, family: str, parent: str | None, model: str | None) -> tuple[list[str], str]:
    """PROTOCOL.md "Prior bridges": the parent's winning bridges; else the family's
    top 2 by importance (same model, same store); else empty (the model chooses)."""
    weaves = store.rows("weaves")
    if parent:
        ids = list(dict.fromkeys(w["bridge_id"] for w in weaves if w.get("problem") == parent and w.get("win")))
        if ids:
            return ids, "parent"
    return prior_bridges_from(weaves, family, model)


def prior_full(store: Store, family: str, parent: str | None, model: str | None) -> tuple[list[str], str]:
    """Run 3's prior P (PROTOCOL.md "A4b", step 2), at most ``method.P_CAP`` bridges,
    filled in order: the parent's winning bridges ranked by their best (lowest)
    g(n_max); then the family's ranking by run 2's rule, skipping bridges already in P;
    if both are empty, P is empty (the method call chooses).

    Only counted bounds whose bridge label is a menu id count, so a model's free-form
    or excluded label never enters P (run 2's ``in_prior`` trusted the label)."""
    menu = {b["id"] for b in prompts.bridges()}
    weaves = [w for w in store.rows("weaves") if w.get("counted", True) and w.get("bridge_id") in menu
              and (model is None or w.get("by") == model)]
    prior: list[str] = []
    if parent:
        best: dict[str, float] = {}
        for w in weaves:
            if w.get("problem") == parent and w.get("win"):
                g = method.g_at_nmax(w)
                if g is not None:
                    best[w["bridge_id"]] = min(best.get(w["bridge_id"], math.inf), g)
        prior = sorted(best, key=lambda b: (best[b], b))[:method.P_CAP]
    from_parent = len(prior)
    ranked, _ = prior_bridges_from(weaves, family, model)
    for b in ranked:
        if len(prior) >= method.P_CAP:
            break
        if b not in prior:
            prior.append(b)
    if not prior:
        return [], "none"
    if from_parent == len(prior):
        return prior, "parent"
    return prior, "parent+family" if from_parent else "family"


def prior_bridges_from(weaves: list[dict], family: str, model: str | None) -> tuple[list[str], str]:
    """Run 2's family ranking over the given weaves: (top 2, "family") or ([], "none")."""
    imp = bridge_importance(weaves, model)
    fam = {w["bridge_id"] for w in weaves if w.get("win") and w.get("family") == family
           and (model is None or w.get("by") == model)}
    ranked = sorted(fam, key=lambda b: (-len(imp[b]["families"]), -len(imp[b]["problems"]), b))
    return (ranked[:2], "family") if ranked else ([], "none")


def overlap(child: dict[int, int], shown: list[dict[int, int]]) -> float | None:
    """PROTOCOL.md "A5b": the maximum, over the problems shown in a child's notes, of
    the share of common n where the two f values are equal (only common n count)."""
    best = None
    for other in shown:
        common = [n for n in child if n in other]
        if common:
            share = sum(1 for n in common if child[n] == other[n]) / len(common)
            best = share if best is None else max(best, share)
    return best


@dataclass
class Config:
    backend: Backend
    store: Store
    policy: str = "surprise"
    feedback: bool = False
    menu: bool = True
    breadth: str = "adaptive"
    k_fixed: int = 3
    m_fixed: int = 3
    max_calls: int = 30
    max_problems: int | None = None
    seeds: list[str] | None = None
    seeds_from: str | None = None
    seeds_top: int | None = None
    predict_only: bool = False
    notes: bool = False
    notes_scope: str = "family"          # run 3: "cluster" (A5b)
    explain: bool = False                # run 3: N2's explanation notes
    clusters: str | None = None          # run 3: clusters.json (A5b problems, in its order)
    clusters_top: int | None = None      # run 3: only the first N clusters in order
    exclude: frozenset = frozenset()     # run 3: problem ids no arm may use (run 2's 40)
    run_id: str = "auto"
    seed: int = 0
    time_limit: float = DEFAULT_TIME_LIMIT
    cache: TruthCache | None = None
    log: callable = print
    retry_wait: float = 0.0              # seconds to wait before retrying after a backend error
    max_backend_errors: int | None = None  # this many backend errors in a row raise BackendDown


@dataclass(order=True)
class _Item:
    key: tuple
    data: dict = field(compare=False)


class Loop:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = cfg.store
        self.backend = cfg.backend
        self.rng = random.Random(cfg.seed)
        self.cache = cfg.cache or TruthCache()
        self.run = cfg.run_id if cfg.run_id != "auto" else \
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)
        self.calls = 0
        self.backend_errors = 0  # consecutive backend (not parse) failures
        self.model_id = getattr(self.backend, "model", None)
        self.frontier: list[_Item] = []
        self.counter = itertools.count()
        self.seen: set[str] = set()
        self.truths: dict[str, dict[int, int | None]] = {}
        self.surprises: dict[str, float | None] = {}
        self.pending_effects: dict[str, list[dict]] = {}
        self.notes: dict[str, list[dict]] = {}  # family or cluster -> processed problems, oldest first
        self.position = 0
        self.cluster_truths: dict[str, list[dict]] = {}  # cluster -> processed members' truths
        self.cur_problem: str | None = None
        self.cur_calls: dict[str, int] = {}
        self.sys_version, self.system = prompts.system_prompt()
        self.log = cfg.log

    # ---- budget and calls -------------------------------------------------------
    def budget_left(self) -> int:
        return self.cfg.max_calls - self.calls

    def ask(self, step: str, template: str, vars: dict, schema: str, context: dict, parse):
        """One model call with one retry on failure. Returns (parsed | None, call_id, version)."""
        version, prompt = prompts.render(template, **vars)
        for attempt in (1, 2):
            if self.budget_left() <= 0:
                return None, None, version
            self.calls += 1
            self.cur_calls[step] = self.cur_calls.get(step, 0) + 1
            call_id = new_id("call")
            row = {"id": call_id, "when": now(), "run": self.run, "step": step, "attempt": attempt,
                   "problem": self.cur_problem, "backend": self.backend.name,
                   "requested_model": getattr(self.backend, "model", None), "model": None,
                   "prompt_version": version,
                   "system_version": self.sys_version, "system": self.system, "prompt": prompt,
                   "raw": None, "parse_ok": False, "parse_error": None, "meta": {}}
            try:
                try:
                    raw, meta = self.backend.complete(self.system, prompt, prompts.SCHEMAS[schema], context)
                except LLMError:
                    self.backend_errors += 1
                    raise
                self.backend_errors = 0
                row["raw"], row["meta"] = raw, meta
                row["model"] = meta.get("model")
                if meta.get("model"):
                    self.model_id = meta["model"]
                parsed = parse(raw)
                row["parse_ok"] = True
                self.store.append("calls", row)
                return parsed, call_id, version
            except (LLMError, prompts.ParseError, ValueError, TypeError, KeyError) as e:
                row["parse_error"] = f"{type(e).__name__}: {e}"
                self.store.append("calls", row)
                self.log(f"  [{step}] attempt {attempt} failed: {row['parse_error'][:200]}")
                if isinstance(e, LLMError):
                    if self.cfg.max_backend_errors and self.backend_errors >= self.cfg.max_backend_errors:
                        raise BackendDown(f"{self.backend_errors} backend errors in a row; last: "
                                          f"{row['parse_error'][:300]}") from None
                    if self.cfg.retry_wait > 0 and attempt == 1:
                        time.sleep(self.cfg.retry_wait)
        return None, call_id, version

    # ---- frontier -------------------------------------------------------------
    def push(self, item: dict, priority: float) -> None:
        heapq.heappush(self.frontier, _Item((-priority, next(self.counter)), item))

    def scan_items(self) -> list[dict]:
        """Problems from a scan selection file, in a seeded random order of the
        selected list (the same order for the same --seed)."""
        rows = [json.loads(x) for x in Path(self.cfg.seeds_from).read_text(encoding="utf-8").splitlines()
                if x.strip()]
        rows.sort(key=lambda r: r["rank"])
        if self.cfg.seeds_top is not None:
            rows = rows[: self.cfg.seeds_top]
        random.Random(self.cfg.seed).shuffle(rows)
        items = []
        for r in rows:
            fam = get_family(r["family"])
            spec = {k: v for k, v in r["spec"].items() if v is not None}
            spec["ns"] = list(r["ns"])
            errs = fam.validate(spec)
            if errs:
                raise ValueError(f"scan problem {r['id']} invalid: {errs}")
            pid = fam.problem_id(spec)
            if pid != r["id"]:
                raise ValueError(f"scan problem {r['id']} hashes to {pid}; selection file is stale")
            if pid in self.cfg.exclude:
                raise ValueError(f"scan problem {pid} is an excluded (run 2) spec")
            items.append({"family": fam.name, "spec": spec, "id": pid, "parent": None, "what_if": None,
                          "provenance": "scan", "seed_name": None, "known_as": [],
                          "scan_rank": r["rank"], "scan_score": r["score"], "scan_S": r.get("S"), "scan_J": r["J"],
                          "scan_L": r["L"]})
        return items

    def cluster_items(self) -> list[dict]:
        """A5b problems from a clusters file, in its seeded order: clusters by ``order``,
        and within a cluster the parent, then the children."""
        data = json.loads(Path(self.cfg.clusters).read_text(encoding="utf-8"))
        clusters = sorted(data["clusters"], key=lambda c: c["order"])
        if self.cfg.clusters_top is not None:
            clusters = clusters[: self.cfg.clusters_top]
        items = []
        for c in clusters:
            for m in sorted(c["members"], key=lambda m: m["cluster_pos"]):
                fam = get_family(m["family"])
                spec = {k: v for k, v in m["spec"].items() if v is not None}
                errs = fam.validate(spec)
                if errs or fam.problem_id(spec) != m["id"]:
                    raise ValueError(f"cluster member {m['id']} invalid or stale: {errs}")
                if m["id"] in self.cfg.exclude:
                    raise ValueError(f"cluster member {m['id']} is an excluded (run 2) spec")
                parent = c["members"][0]["id"] if m["role"] == "child" else None
                items.append({"family": fam.name, "spec": spec, "id": m["id"], "parent": parent,
                              "what_if": None if parent is None else f"changed {m.get('changed')}",
                              "provenance": "scan" if m["role"] == "parent" else "variant",
                              "seed_name": None, "known_as": [], "cluster": c["cluster"],
                              "cluster_order": c["order"], "cluster_pos": m["cluster_pos"],
                              "cluster_size": len(c["members"]), "cluster_role": m["role"],
                              **({"scan_rank": m["scan_rank"], "scan_score": m["scan_score"], "scan_S": m["scan_S"],
                                  "scan_J": m["scan_J"], "scan_L": m["scan_L"]} if m["role"] == "parent" else {})})
        return items

    def seed_frontier(self) -> None:
        if self.cfg.clusters:
            for item in self.cluster_items():
                self.seen.add(item["id"])
                self.store.append("problems", {"event": "queued", "when": now(), "run": self.run, **item})
                self.push(item, math.inf)
            return
        if self.cfg.seeds_from:
            for item in self.scan_items():
                if item["id"] in self.seen:
                    continue
                self.seen.add(item["id"])
                self.store.append("problems", {"event": "queued", "when": now(), "run": self.run, **item})
                self.push(item, math.inf)
            return
        names = self.cfg.seeds
        for s in prompts.seeds():
            if names and s["name"] not in names:
                continue
            fam = get_family(s["family"])
            spec = dict(s["spec"])
            spec.setdefault("ns", fam.default_ns(spec))
            errs = fam.validate(spec)
            if errs:
                raise ValueError(f"seed {s['name']} invalid: {errs}")
            pid = fam.problem_id(spec)
            if pid in self.seen:
                continue
            self.seen.add(pid)
            item = {"family": fam.name, "spec": spec, "id": pid, "parent": None, "what_if": None,
                    "provenance": "named", "seed_name": s["name"], "known_as": s.get("known_as", [])}
            self.store.append("problems", {"event": "queued", "when": now(), "run": self.run, **item})
            self.push(item, math.inf)
        if names:
            missing = set(names) - {s["name"] for s in prompts.seeds()}
            if missing:
                raise KeyError(f"unknown seeds {sorted(missing)}")

    def run_loop(self) -> dict:
        """Process problems until the frontier, ``max_problems`` or the call budget runs
        out. In full breadth (A4b) ``max_problems`` counts completed problems (probe,
        re-probe and wide weave all stored), as the protocol sizes the arm."""
        self.seed_frontier()
        done = completed = 0
        while self.frontier and self.budget_left() > 0:
            count = completed if self.cfg.breadth == "full" and not self.cfg.predict_only else done
            if self.cfg.max_problems is not None and count >= self.cfg.max_problems:
                break
            item = heapq.heappop(self.frontier).data
            self.log(f"problem {item['id']} ({item.get('seed_name') or item['provenance']}): "
                     f"{item['family']} {json.dumps({k: v for k, v in item['spec'].items() if k != 'ns'})}")
            rec = self.process(item)
            done += 1
            completed += bool(rec.get("completed"))
        self.cache.flush()
        self.log(f"run {self.run}: {done} problems ({completed} completed), {self.calls} calls, store {self.store.dir}")
        return {"run": self.run, "problems": done, "completed": completed, "calls": self.calls}

    # ---- one problem ------------------------------------------------------------
    def feedback(self, fam: Family) -> str:
        if not self.cfg.feedback:
            return ""
        text = calibration.feedback_text(self.store.joined_predictions(), fam.name, self.model_id)
        return "\nYour running calibration on this problem family so far:\n" + text + "\n"

    def notes_key(self, fam: Family, item: dict) -> str:
        return item["cluster"] if self.cfg.notes_scope == "cluster" else fam.name

    def notes_text(self, fam: Family, item: dict) -> tuple[str, int]:
        """Arm A5: statements and true sequences of the problems already processed in
        this run in the same family, most recent first, at most NOTES_CAP. Run 3 (A5b):
        the same, scoped to the problem's cluster, and with each note's explanation
        when ``explain`` is on."""
        if not self.cfg.notes:
            return "", 0
        prev = list(reversed(self.notes.get(self.notes_key(fam, item), [])))[:NOTES_CAP]
        if not prev:
            return "\nNotes: no problems of this kind have been computed yet in this session.\n", 0
        lines = ["", "Notes: problems of the same kind computed earlier in this session, most recent "
                 "first, with their true values of f:"]
        for i, nt in enumerate(prev, 1):
            vals = ", ".join(f"f({n}) = {f}" for n, f in nt["truth"].items())
            lines.append(f"{i}. {nt['statement']}\n   {vals}")
            if self.cfg.explain and nt.get("explanation"):
                lines.append(f"   Explanation: {nt['explanation']}")
        return "\n".join(lines) + "\n", len(prev)

    def condition(self) -> str | None:
        """A5b's condition label for a predict-only run: N0 no notes, N1 cluster notes,
        N2 cluster notes with explanations (None outside cluster scope)."""
        if not self.cfg.predict_only or (self.cfg.notes and self.cfg.notes_scope != "cluster"):
            return None
        if not self.cfg.notes:
            return "N0" if self.cfg.clusters else None
        return "N2" if self.cfg.explain else "N1"

    def process(self, item: dict) -> dict:
        fam = get_family(item["family"])
        spec, pid = item["spec"], item["id"]
        ns = list(spec["ns"])
        statement = fam.render(spec)
        self.position += 1
        self.cur_problem, self.cur_calls = pid, {}
        rec = {"event": "done", "when": None, "run": self.run, "id": pid, "family": fam.name,
               "parent": item["parent"], "what_if": item["what_if"], "provenance": item["provenance"],
               "seed_name": item.get("seed_name"), "spec": spec, "statement": statement,
               "breadth": None if self.cfg.predict_only else self.cfg.breadth, "policy": self.cfg.policy,
               "feedback": self.cfg.feedback, "menu": self.cfg.menu, "predict_only": self.cfg.predict_only,
               "notes": self.cfg.notes, "notes_shown": 0, "position": self.position,
               **{k: item[k] for k in ("scan_rank", "scan_score", "scan_S", "scan_J", "scan_L") if k in item},
               "truth": {}, "timings": {}, "surprise_unit": "bits", "bits": {}, "surprise": None,
               "surprise_max": None, "surprising": None, "prior_bridges": [], "prior_source": None,
               "mode": None, "escalated": False, "calls_used": 0, "calls_by_step": {}, "stopped": None}
        if self.cfg.notes_scope == "cluster" or item.get("cluster"):
            rec.update({"notes_scope": self.cfg.notes_scope, "explain": self.cfg.explain,
                        "condition": self.condition(), "overlap": None, "explanation": None,
                        **{k: item.get(k) for k in ("cluster", "cluster_order", "cluster_pos", "cluster_size",
                                                    "cluster_role")}})
        if self.cfg.breadth == "full" and not self.cfg.predict_only:
            rec.update({"stratum": "seed" if item["parent"] is None else "mutant", "completed": False})
        calls0 = self.calls
        try:
            self._process(item, fam, spec, pid, ns, statement, rec)
        finally:
            rec["when"] = now()
            rec["calls_used"] = self.calls - calls0
            rec["calls_by_step"] = dict(self.cur_calls)
            self.store.append("problems", rec)
            self.cur_problem = None
            known = {n: f for n, f in self.truths.get(pid, {}).items() if f is not None}
            if known:
                self.notes.setdefault(self.notes_key(fam, item), []).append(
                    {"statement": statement, "truth": known, "explanation": rec.get("explanation")})
                if item.get("cluster"):
                    self.cluster_truths.setdefault(item["cluster"], []).append(known)
        return rec

    def _process(self, item, fam, spec, pid, ns, statement, rec) -> None:
        # 1. predict
        ctx = {"step": "predict", "family": fam, "spec": spec, "ns": ns, "seed_name": item.get("seed_name")}
        notes, rec["notes_shown"] = self.notes_text(fam, item)
        parsed, call_id, ver = self.ask(
            "predict", "predict",
            {"problem": statement, "ns": ", ".join(map(str, ns)), "feedback": self.feedback(fam),
             "notes": notes},
            "predict", ctx, lambda t: prompts.parse_predict(t, ns))
        if parsed is None:
            rec["stopped"] = "no predictions (budget or parse failure); truth not computed"
            return
        preds = parsed["predictions"]
        dist_ids, top_ids = {}, {}
        for n in ns:
            if n not in preds:
                continue
            d = preds[n]
            listed = [v for v, _ in d["candidates"]]
            common = dict(problem=pid, call=call_id, prompt_version=ver, n=n, provenance=item["provenance"],
                          by=self.model_id, domain=fam.name, run=self.run)
            r = self.store.predict(kind="distribution", level="value",
                                   claim=f"f({n}) in {{{', '.join(map(str, listed))}}}",
                                   confidence=min(1.0, sum(p for _, p in d["candidates"])),
                                   candidates=d["candidates"], other=d["other"],
                                   normalized=d["normalized"], trivial=fam.trivial_bound(spec, n), **common)
            dist_ids[n] = r["id"]
            if d["candidates"]:
                top, p_top = d["candidates"][0]
                top_ids[n] = self.store.predict(kind="top1", level="value", claim=f"f({n}) = {top}",
                                                confidence=p_top, top=top, **common)["id"]
        rec_id = None
        if parsed["recognized_as"] and parsed["p_recognized"] is not None:
            rec_id = self.store.predict(
                run=self.run, by=self.model_id, domain=fam.name, kind="recognition", level="ontology",
                claim=f"this problem is the named result {parsed['recognized_as']}",
                confidence=parsed["p_recognized"], problem=pid, call=call_id, prompt_version=ver,
                recognized_as=parsed["recognized_as"], provenance=item["provenance"])["id"]

        # 2. compute (refusal (a) inside), then score
        truth: dict[int, int | None] = {}
        for n in ns:
            if n not in dist_ids:
                continue
            self.store.require_value_predictions(pid, n)
            value, secs, cached = self.cache.get(fam, spec, n, self.cfg.time_limit)
            truth[n] = value
            rec["timings"][str(n)] = {"seconds": round(secs, 4), "cached": cached}
        rec["truth"] = {str(n): v for n, v in truth.items()}
        self.truths[pid] = truth
        bits_by_n: dict[int, float] = {}
        for n, rid in dist_ids.items():
            f, d = truth[n], preds[n]
            if f is None:
                self.store.outcome(rid, "couldnt-tell", None)
                if n in top_ids:
                    self.store.outcome(top_ids[n], "couldnt-tell", None)
                continue
            b, p_true = bits(d, f, fam.trivial_bound(spec, n))
            bits_by_n[n] = b
            listed = any(v == f for v, _ in d["candidates"])
            self.store.outcome(rid, "right" if listed else "wrong", f, bits=b, p_true=p_true)
            if n in top_ids:
                self.store.outcome(top_ids[n], "right" if d["candidates"][0][0] == f else "wrong", f)
        if rec_id:
            if item["provenance"] == "named":
                ok = prompts.recognition_matches(parsed["recognized_as"], item.get("known_as", []))
                self.store.outcome(rec_id, "right" if ok else "wrong", item.get("seed_name"))
            else:
                self.store.outcome(rec_id, "couldnt-tell", item["provenance"])
        self._score_effects(pid, truth)
        mean, mx = surprise_stats(bits_by_n)
        rec["bits"] = {str(n): round(b, 4) for n, b in bits_by_n.items()}
        rec["surprise"], rec["surprise_max"] = mean, mx
        self.surprises[pid] = mean
        rec["surprising"] = is_surprising(mx)
        known = {n: f for n, f in truth.items() if f is not None}
        self.log(f"  f = {known}  surprise mean={mean if mean is None else round(mean, 2)} bits "
                 f"max={mx if mx is None else round(mx, 2)} bits")
        if item.get("cluster"):
            # the earlier members of the cluster: what N1/N2 show, so the same in every condition
            shown = self.cluster_truths.get(item["cluster"], [])
            rec["overlap"] = overlap(known, shown) if item.get("cluster_role") == "child" else None
        if self.cfg.predict_only:
            if self.cfg.explain and known and not (item.get("cluster_pos") == item.get("cluster_size")
                                                   and item.get("cluster_role") == "child"):
                self.explain(fam, spec, pid, statement, known, rec)
            return

        if self.cfg.breadth == "full":
            self.full_breadth(item, fam, spec, pid, ns, statement, truth, rec)
            return

        # 3. breadth (arm A4)
        prior, source = prior_bridges(self.store, fam.name, item["parent"], self.model_id)
        rec["prior_bridges"], rec["prior_source"] = prior, source
        if self.cfg.breadth == "fixed":
            mode, k, m = "fixed", self.cfg.k_fixed, self.cfg.m_fixed
        elif rec["surprising"]:
            mode, k, m = "wide", WIDE_K, WIDE_M
        else:
            mode, k, m = "narrow", NARROW_K, NARROW_M
        rec["mode"] = mode

        # 4. weave
        rows = self.weave(item, fam, spec, pid, ns, statement, truth, mode, k, prior, weave_call=1)
        wins = sum(1 for w in rows or [] if w["win"])
        if mode == "narrow" and wins == 0:
            if self.budget_left() > 0:
                rec["escalated"] = True
                m = WIDE_M
                self.log("  narrow weave found no win: the prior failed, escalating to wide")
                self.weave(item, fam, spec, pid, ns, statement, truth, "wide", WIDE_K, prior, weave_call=2)

        # 5. mutate
        self.mutate(item, fam, spec, pid, truth, m)

    def checkable_values(self, fam: Family, spec: dict) -> int:
        """How many values brute force yields on the spec's ns, counted in increasing n
        and stopping at the first timeout (or as soon as MIN_CHECKABLE are found).

        The values go to the truth cache only: nothing reaches the store or a prompt,
        and refusal (a) still gates the computation the loop scores."""
        ns = sorted(spec["ns"])
        if len(ns) < MIN_CHECKABLE:
            return len(ns)
        count = 0
        for n in ns:
            value, _, _ = self.cache.get(fam, spec, n, self.cfg.time_limit)
            if value is None:
                break
            count += 1
            if count >= MIN_CHECKABLE:
                break
        return count

    def _score_effects(self, child: str, child_truth: dict) -> None:
        for pend in self.pending_effects.pop(child, []):
            parent_truth = self.truths.get(pend["parent"], {})
            common = sorted(n for n in child_truth if child_truth.get(n) is not None
                            and parent_truth.get(n) is not None)
            if not common:
                self.store.outcome(pend["pred"], "couldnt-tell", None)
                continue
            n = common[-1]
            c, p = child_truth[n], parent_truth[n]
            obs = "larger" if c > p else "smaller" if c < p else "equal"
            self.store.outcome(pend["pred"], "right" if obs == pend["effect"] else "wrong", obs,
                               at_n=n, child_f=c, parent_f=p)

    def _breadth_text(self, mode: str, k: int, prior: list[str]) -> str:
        if mode == "probe":
            return "Give exactly 1 bound, using one of the bridges in the menu below."
        if mode == "wide3":
            return f"Give up to {k} bounds, using different bridges from the menu below where you can."
        if mode == "fixed":
            return f"Give up to {k} bounds, using different bridges where you can."
        if mode == "narrow":
            if prior:
                return ("Give exactly 1 bound. Use one of these bridges, which won on related problems: "
                        + ", ".join(prior) + ".")
            return "Give exactly 1 bound, from the bridge you judge most promising."
        if prior:
            return (f"Give up to {k} bounds. The prior (bridges that won on related problems) is: "
                    + ", ".join(prior) + f". At least {WIDE_MIN_OUTSIDE} of your bounds must use bridges "
                    "NOT in the prior.")
        return f"Give up to {k} bounds, using at least {WIDE_MIN_OUTSIDE} different bridges."

    def weave(self, item, fam, spec, pid, ns, statement, truth, mode, k, prior, weave_call,
              restrict: list[str] | None = None, excluded: set[str] | None = None,
              step: str = "weave") -> list[dict] | None:
        """One weave call; returns the stored weave rows, or None if the call failed
        (budget or two parse failures). ``restrict`` limits the bridge menu (run 2's
        narrow mode restricts it to the prior); a bound whose bridge label is in
        ``excluded`` is stored with ``counted: false`` (run 3's wide weave)."""
        params = fam.params(spec)
        if restrict is None and mode == "narrow" and prior:
            restrict = prior
        vars = {"problem": statement, "ns": ", ".join(map(str, ns)), "k": k,
                "params": (" (you may also use the names " + ", ".join(f"{p} = {v}" for p, v in params.items())
                           + ")") if params else "",
                "breadth": self._breadth_text(mode, k, prior), "feedback": self.feedback(fam)}
        template = "weave_menu" if self.cfg.menu else "weave_free"
        if self.cfg.menu:
            vars["bridges"] = prompts.bridge_menu(restrict)
            vars["lemmas"] = prompts.lemma_menu()
        ctx = {"step": "weave", "family": fam, "spec": spec, "ns": ns, "truth": truth, "k": k,
               "mode": mode, "prior": prior, "allowed": restrict, "weave_call": weave_call}
        bounds, call_id, ver = self.ask(step, template, vars, "weave", ctx,
                                        lambda t: prompts.parse_weave(t, k))
        if bounds is None:
            return None
        bridge_ids = {b["id"] for b in prompts.bridges()}
        lemma_ids = {lm["id"] for lm in prompts.lemmas()}
        rows = []
        for b in bounds:
            pred = self.store.predict(
                run=self.run, by=self.model_id, domain=fam.name, kind="bound_holds", level="mechanism",
                claim=f"g(n) = {b['bound_expr']} >= f(n) for every computed n", confidence=b["p_holds"],
                problem=pid, call=call_id, prompt_version=ver, bound_expr=b["bound_expr"],
                bridge_id=b["bridge_id"], provenance=item["provenance"])
            res = check_bound(b["bound_expr"], fam, spec, truth)
            status = res["status"]
            self.store.outcome(pred["id"], {"consistent": "right", "violated": "wrong"}.get(status, "couldnt-tell"),
                               status)
            counted = not (excluded and b["bridge_id"] in excluded)
            row = {
                "id": new_id("weave"), "when": now(), "run": self.run, "by": self.model_id,
                "problem": pid, "family": fam.name, "provenance": item["provenance"],
                "menu": self.cfg.menu, "breadth": self.cfg.breadth, "mode": mode, "weave_call": weave_call,
                "step": step, "prior_bridges": prior, "in_prior": b["bridge_id"] in prior,
                "menu_ids_ok": (b["bridge_id"] in bridge_ids and all(x in lemma_ids for x in b["lemma_ids"]))
                if self.cfg.menu else None,
                **({"counted": counted, "in_menu_shown": restrict is None or b["bridge_id"] in restrict}
                   if self.cfg.breadth == "full" else {}),
                **b, **res, "prediction": pred["id"], "call": call_id, "prompt_version": ver}
            self.store.append("weaves", row)
            rows.append(row)
            self.log(f"  weave[{weave_call}/{mode}] {b['bridge_id']}+{'+'.join(b['lemma_ids'])}: "
                     f"g = {b['bound_expr']!r} -> {status}{' WIN' if res['win'] else ''}"
                     f" tight_at={res['tight_at']}{'' if counted else ' (excluded label: not counted)'}")
        return rows

    # ---- run 3 ----------------------------------------------------------------
    def explain(self, fam: Family, spec: dict, pid: str, statement: str, known: dict, rec: dict) -> None:
        """A5b's N2 note: a separate call that sees the statement, the true values and the
        extremal witness at n_max, and writes at most 80 words on the structure."""
        n_max = max(known)
        _, witness = self.cache.get_witness(fam, spec, n_max, self.cfg.time_limit)
        wtext = fam.witness_text(spec, n_max, witness) if witness is not None else "(none available)"
        ctx = {"step": "explain", "family": fam, "spec": spec}
        parsed, call_id, ver = self.ask(
            "explain", "explain",
            {"problem": statement, "truth": ", ".join(f"f({n}) = {f}" for n, f in known.items()),
             "witness": wtext}, "explain", ctx, prompts.parse_explain)
        rec["explain_call"] = call_id
        if parsed is not None:
            rec["explanation"] = parsed["explanation"]
            rec["explanation_words"] = parsed["words"]
            rec["explanation_truncated"] = parsed["truncated"]

    def method_forecast(self, item, fam, spec, pid, statement, known, prior, rec) -> dict | None:
        """Step 3 of A4b: the method_band forecast, written to the store before any probe.
        With an empty P the call shows the full menu and its chosen bridge becomes P."""
        n_max = max(known)
        choose = not prior
        menu_ids = [b["id"] for b in prompts.bridges()]
        shown = menu_ids if choose else prior
        ctx = {"step": "method", "family": fam, "spec": spec, "choose": choose, "menu": shown}
        parsed, call_id, ver = self.ask(
            "method", "method",
            {"problem": statement, "truth": ", ".join(f"f({n}) = {f}" for n, f in known.items()),
             "bridges": prompts.bridge_menu(shown), "n_max": n_max, "f_max": known[n_max],
             "choose": ("\nFirst choose the bridge from the menu below that you would give the solver: put its id "
                        "in \"bridge_id\". The solver will be restricted to that bridge.\n") if choose else "",
             "bridge_shape": "<bridge id>" if choose else "null"},
            "method", ctx, lambda t: prompts.parse_method(t, menu_ids if choose else None))
        if parsed is None:
            return None
        dist = parsed["bands"]
        top = max(method.BANDS, key=lambda b: (dist[b], -method.BANDS.index(b)))
        common = dict(run=self.run, by=self.model_id, domain=fam.name, level="mechanism", problem=pid,
                      call=call_id, prompt_version=ver, provenance=item["provenance"], n_max=n_max)
        row = self.store.predict(kind="method_band", claim="the first probe's best bound falls in band "
                                 + ", ".join(f"{b} {dist[b]:.2f}" for b in method.BANDS),
                                 confidence=1.0, bands=dist, chosen_bridge=parsed["bridge_id"],
                                 normalized=parsed["normalized"], **common)
        top_row = self.store.predict(kind="method_band_top", claim=f"the first probe's best bound is {top}",
                                     confidence=dist[top], top=top, **common)
        return {"bands": dist, "top": top, "bridge_id": parsed["bridge_id"], "pred": row["id"],
                "top_pred": top_row["id"], "call": call_id}

    def full_breadth(self, item, fam, spec, pid, ns, statement, truth, rec) -> None:
        """A4b, steps 2-6 (PROTOCOL.md "Run 3"). Everything the replay needs is stored on
        the problem row under ``full``; the signals are fixed before the wide weave."""
        known = {n: f for n, f in truth.items() if f is not None}
        rec["mode"] = "full"
        full = rec["full"] = {"P": [], "P_source": None, "p_from_method": False, "method": None,
                              "probe": None, "reprobe": None, "wide": None, "Y": None}
        if not known:
            rec["stopped"] = "no computed values"
            return
        n_max, f_max = max(known), known[max(known)]
        # 2. prior P (at most 2 bridges)
        prior, source = prior_full(self.store, fam.name, item["parent"], self.model_id)
        # 3. method prediction (a separate call that sees the true values)
        fc = self.method_forecast(item, fam, spec, pid, statement, known, prior, rec)
        if fc is None:
            rec["stopped"] = "no method prediction (budget or parse failure); probe not run"
            return
        if not prior:
            prior, source = [fc["bridge_id"]], "method"
            full["p_from_method"] = True
        full["P"], full["P_source"] = prior, source
        rec["prior_bridges"], rec["prior_source"] = prior, source
        full["method"] = {"bands": fc["bands"], "top": fc["top"], "median": method.median_band(fc["bands"])}
        # refusal: no probe until the method row is on disk
        self.store.require_method_prediction(pid, self.run)
        # 4. probe and re-probe: k = 1, menu restricted to P
        p1 = self.weave(item, fam, spec, pid, ns, statement, truth, "probe", method.PROBE_K, prior, 1,
                        restrict=prior, step="probe")
        if p1 is None:
            rec["stopped"] = "probe failed (budget or parse failure)"
            self.store.outcome(fc["pred"], "couldnt-tell", None)
            self.store.outcome(fc["top_pred"], "couldnt-tell", None)
            return
        pr1 = method.best_probe(p1, f_max)
        band = pr1["band"]
        m_bits, p_obs = method.band_bits(fc["bands"], band)
        # the method rows are scored on the first probe alone
        self.store.outcome(fc["pred"], "right", band, bits=m_bits, p_true=p_obs)
        self.store.outcome(fc["top_pred"], "right" if band == fc["top"] else "wrong", band)
        full["method"].update({"observed": band, "M": m_bits, "p_observed": p_obs,
                               "direction": method.direction(fc["bands"], band),
                               "M_trigger": m_bits >= method.METHOD_SURPRISE_BITS})
        full["probe"] = {**pr1, "stored": True, "n_max": n_max, "f_max": f_max}
        p2 =self.weave(item, fam, spec, pid, ns, statement, truth, "probe", method.PROBE_K, prior, 2,
                        restrict=prior, step="reprobe")
        full["reprobe"] = {**method.best_probe(p2 or [], f_max), "stored": p2 is not None}
        if p2 is None:
            rec["stopped"] = "re-probe failed"
            return
        # 5. wide: k = 3, menu outside every group that contains a P bridge
        excluded = method.excluded_bridges(prior)
        allowed = [b["id"] for b in prompts.bridges() if b["id"] not in excluded]
        full["wide_menu"] = allowed
        w = self.weave(item, fam, spec, pid, ns, statement, truth, "wide3", method.WIDE3_K, prior, 3,
                       restrict=allowed, excluded=excluded, step="wide")
        if w is None:
            rec["stopped"] = "wide weave failed"
            return
        counted = [x for x in w if x["counted"]]
        gs = [g for g in (method.g_at_nmax(x) for x in counted) if g is not None]
        probe_gs = [g for g in (full["probe"]["best_g"], full["reprobe"]["best_g"]) if g is not None]
        best_wide = min(gs) if gs else None
        full["wide"] = {"best_g": best_wide, "n_bounds": len(w), "n_counted": len(counted),
                        "n_excluded_label": len(w) - len(counted),
                        "counted_wins": sum(1 for x in counted if x["win"])}
        full["Y"] = best_wide is not None and (not probe_gs or best_wide < min(probe_gs))
        rec["completed"] = True
        # 6. mutate (m = 3; the run uses --policy random)
        self.mutate(item, fam, spec, pid, truth, method.FULL_M)

    def mutate(self, item, fam, spec, pid, truth, m) -> None:
        if self.budget_left() <= 0:
            return
        bare = {k: v for k, v in spec.items() if k != "ns"}
        known = {n: f for n, f in truth.items() if f is not None}
        ctx = {"step": "mutate", "family": fam, "spec": spec, "m": m}
        muts, call_id, ver = self.ask(
            "mutate", "mutate",
            {"problem": fam.render(spec), "spec": json.dumps(bare), "m": m, "schema": fam.schema_text(),
             "truth": ", ".join(f"f({n}) = {f}" for n, f in known.items()) or "none computed"},
            "mutate", ctx, lambda t: prompts.parse_mutate(t, m))
        for mu in muts or []:
            child = {k: v for k, v in mu["spec"].items() if k != "ns"}
            try:
                same = fam.ns_regime(child) == fam.ns_regime(spec)
                child["ns"] = list(spec["ns"]) if same else fam.default_ns(child)
            except Exception:  # malformed spec: keep the parent's ns, validation rejects it
                child["ns"] = list(spec["ns"])
            base = {"when": now(), "run": self.run, "parent": pid, "family": fam.name, "spec": child,
                    "what_if": mu["what_if"], "provenance": "variant", "call": call_id}
            errs = validate_mutation(fam, spec, child)
            if errs:
                self.store.append("problems", {"event": "rejected", "id": None, "reasons": errs, **base})
                self.log(f"  mutation rejected: {errs}")
                continue
            cid = fam.problem_id(child)
            if cid in self.cfg.exclude:
                self.store.append("problems", {"event": "excluded", "id": cid, "reason": "run 2 spec", **base})
                self.log(f"  mutation excluded: {cid} is one of run 2's specs")
                continue
            if cid in self.seen:
                self.store.append("problems", {"event": "duplicate", "id": cid, **base})
                continue
            self.seen.add(cid)
            n_values = self.checkable_values(fam, child)
            if n_values < MIN_CHECKABLE:
                self.store.append("problems", {"event": "refused_uncheckable", "id": cid,
                                               "values_computable": n_values, **base})
                self.log(f"  mutation refused: only {n_values} computable values on ns {child['ns']}")
                continue
            pred = self.store.predict(
                run=self.run, by=self.model_id, domain=fam.name, kind="mutation_effect", level="mechanism",
                claim=f"child {cid} has {mu['predicted_effect']} f than parent {pid} at the largest common n",
                confidence=mu["p_effect"], problem=cid, parent=pid, call=call_id, prompt_version=ver,
                predicted_effect=mu["predicted_effect"], provenance="variant")
            self.pending_effects.setdefault(cid, []).append(
                {"pred": pred["id"], "parent": pid, "effect": mu["predicted_effect"]})
            item_c = {"family": fam.name, "spec": child, "id": cid, "parent": pid,
                      "what_if": mu["what_if"], "provenance": "variant", "seed_name": None, "known_as": []}
            self.store.append("problems", {"event": "queued", **base, "id": cid})
            prio = (self.surprises.get(pid) or 0.0) if self.cfg.policy == "surprise" else self.rng.random()
            self.push(item_c, prio)
