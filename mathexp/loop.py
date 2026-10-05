"""The weaving loop: predict -> compute -> weave -> check -> what-if, per problem.

Order inside one problem (PROTOCOL.md "One iteration"):

1. predict f(n) with 80% intervals (+ recognition) -- written to the store first;
2. compute f(n) by brute force, only for n whose interval prediction is on disk
   (refusal (a)), and score the predictions; score pending mutation-effect
   predictions whose child is this problem;
3. choose the breadth (arm A4) from the surprise and the prior bridges;
4. weave (one call; adaptive narrow mode may escalate to a second, wide call),
   writing each bound's P(holds) before checking it;
5. propose what-if mutations, validate them (refusal (c)) and queue them.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import calibration, prompts
from .checking import check_bound
from .llm import Backend, LLMError
from .problems import get_family, validate_mutation
from .problems.base import Family
from .store import Refusal, Store, new_id, now
from .truth import DEFAULT_TIME_LIMIT, TruthCache

#: A problem is *surprising* iff max over n (f known) of |log2((point+1)/(f+1))|
#: reaches this. Fixed by PROTOCOL.md "Definitions"; deliberately not a flag.
SURPRISE_TRIGGER_LOG2 = 1.0

#: Arm A4 breadths.
NARROW_K, NARROW_M = 1, 1
WIDE_K, WIDE_M, WIDE_MIN_OUTSIDE = 5, 5, 3


def surprise_stats(preds: dict[int, dict], truth: dict[int, int | None]) -> tuple[float | None, float | None]:
    """(mean, max) over n with known f of |log2((point+1)/(f+1))|; (None, None) if none known."""
    errs = [abs(math.log2((max(preds[n]["point"], 0) + 1) / (f + 1)))
            for n, f in truth.items() if f is not None and n in preds]
    if not errs:
        return None, None
    return sum(errs) / len(errs), max(errs)


def is_surprising(surprise_max: float | None) -> bool:
    return surprise_max is not None and surprise_max >= SURPRISE_TRIGGER_LOG2


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
    imp = bridge_importance(weaves, model)
    fam = {w["bridge_id"] for w in weaves if w.get("win") and w.get("family") == family
           and (model is None or w.get("by") == model)}
    ranked = sorted(fam, key=lambda b: (-len(imp[b]["families"]), -len(imp[b]["problems"]), b))
    if ranked:
        return ranked[:2], "family"
    return [], "none"


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
    run_id: str = "auto"
    seed: int = 0
    time_limit: float = DEFAULT_TIME_LIMIT
    cache: TruthCache | None = None
    log: callable = print


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
        self.model_id = getattr(self.backend, "model", None)
        self.frontier: list[_Item] = []
        self.counter = itertools.count()
        self.seen: set[str] = set()
        self.truths: dict[str, dict[int, int | None]] = {}
        self.surprises: dict[str, float | None] = {}
        self.pending_effects: dict[str, list[dict]] = {}
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
            call_id = new_id("call")
            row = {"id": call_id, "when": now(), "run": self.run, "step": step, "attempt": attempt,
                   "backend": self.backend.name, "prompt_version": version,
                   "system_version": self.sys_version, "system": self.system, "prompt": prompt,
                   "raw": None, "parse_ok": False, "parse_error": None, "meta": {}}
            try:
                raw, meta = self.backend.complete(self.system, prompt, prompts.SCHEMAS[schema], context)
                row["raw"], row["meta"] = raw, meta
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
        return None, call_id, version

    # ---- frontier -------------------------------------------------------------
    def push(self, item: dict, priority: float) -> None:
        heapq.heappush(self.frontier, _Item((-priority, next(self.counter)), item))

    def seed_frontier(self) -> None:
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
        self.seed_frontier()
        done = 0
        while self.frontier and (self.cfg.max_problems is None or done < self.cfg.max_problems) and self.budget_left() > 0:
            item = heapq.heappop(self.frontier).data
            self.log(f"problem {item['id']} ({item.get('seed_name') or item['provenance']}): "
                     f"{item['family']} {json.dumps({k: v for k, v in item['spec'].items() if k != 'ns'})}")
            self.process(item)
            done += 1
        self.log(f"run {self.run}: {done} problems, {self.calls} calls, store {self.store.dir}")
        return {"run": self.run, "problems": done, "calls": self.calls}

    # ---- one problem ------------------------------------------------------------
    def feedback(self, fam: Family) -> str:
        if not self.cfg.feedback:
            return ""
        text = calibration.feedback_text(self.store.joined_predictions(), fam.name, self.model_id)
        return "\nYour running calibration on this problem family so far:\n" + text + "\n"

    def process(self, item: dict) -> None:
        fam = get_family(item["family"])
        spec, pid = item["spec"], item["id"]
        ns = list(spec["ns"])
        statement = fam.render(spec)
        rec = {"event": "done", "when": None, "run": self.run, "id": pid, "family": fam.name,
               "parent": item["parent"], "what_if": item["what_if"], "provenance": item["provenance"],
               "seed_name": item.get("seed_name"), "spec": spec, "statement": statement,
               "breadth": self.cfg.breadth, "policy": self.cfg.policy, "feedback": self.cfg.feedback,
               "menu": self.cfg.menu, "truth": {}, "timings": {}, "surprise": None, "surprise_max": None,
               "surprising": None, "prior_bridges": [], "prior_source": None, "mode": None,
               "escalated": False, "calls_used": 0, "stopped": None}
        calls0 = self.calls
        try:
            self._process(item, fam, spec, pid, ns, statement, rec)
        finally:
            rec["when"] = now()
            rec["calls_used"] = self.calls - calls0
            self.store.append("problems", rec)

    def _process(self, item, fam, spec, pid, ns, statement, rec) -> None:
        # 1. predict
        ctx = {"step": "predict", "family": fam, "spec": spec, "ns": ns, "seed_name": item.get("seed_name")}
        parsed, call_id, ver = self.ask(
            "predict", "predict",
            {"problem": statement, "ns": ", ".join(map(str, ns)), "feedback": self.feedback(fam)},
            "predict", ctx, lambda t: prompts.parse_predict(t, ns))
        if parsed is None:
            rec["stopped"] = "no predictions (budget or parse failure); truth not computed"
            return
        preds = parsed["predictions"]
        interval_ids = {}
        for n in ns:
            if n not in preds:
                continue
            p = preds[n]
            r = self.store.predict(run=self.run, by=self.model_id, domain=fam.name, kind="interval",
                                   level="value", claim=f"f({n}) in [{p['lo']:g}, {p['hi']:g}]",
                                   confidence=0.8, problem=pid, call=call_id, prompt_version=ver,
                                   n=n, point=p["point"], lo=p["lo"], hi=p["hi"],
                                   provenance=item["provenance"])
            interval_ids[n] = r["id"]
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
            if n not in interval_ids:
                continue
            self.store.require_interval_predictions(pid, n)
            value, secs, cached = self.cache.get(fam, spec, n, self.cfg.time_limit)
            truth[n] = value
            rec["timings"][str(n)] = {"seconds": round(secs, 4), "cached": cached}
        rec["truth"] = {str(n): v for n, v in truth.items()}
        self.truths[pid] = truth
        for n, rid in interval_ids.items():
            f, p = truth[n], preds[n]
            if f is None:
                self.store.outcome(rid, "couldnt-tell", None)
                continue
            err = math.log2((max(p["point"], 0) + 1) / (f + 1))
            self.store.outcome(rid, "right" if p["lo"] <= f <= p["hi"] else "wrong", f,
                               log2_err=err, abs_log2_err=abs(err))
        if rec_id:
            if item["provenance"] == "named":
                ok = prompts.recognition_matches(parsed["recognized_as"], item.get("known_as", []))
                self.store.outcome(rec_id, "right" if ok else "wrong", item.get("seed_name"))
            else:
                self.store.outcome(rec_id, "couldnt-tell", "variant")
        self._score_effects(pid, truth)
        mean, mx = surprise_stats(preds, truth)
        rec["surprise"], rec["surprise_max"] = mean, mx
        self.surprises[pid] = mean
        rec["surprising"] = is_surprising(mx)
        known = {n: f for n, f in truth.items() if f is not None}
        self.log(f"  f = {known}  surprise mean={mean if mean is None else round(mean, 3)} "
                 f"max={mx if mx is None else round(mx, 3)}")

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
        wins = self.weave(item, fam, spec, pid, ns, statement, truth, mode, k, prior, weave_call=1)
        if mode == "narrow" and wins == 0:
            if self.budget_left() > 0:
                rec["escalated"] = True
                m = WIDE_M
                self.log("  narrow weave found no win: the prior failed, escalating to wide")
                self.weave(item, fam, spec, pid, ns, statement, truth, "wide", WIDE_K, prior, weave_call=2)

        # 5. mutate
        self.mutate(item, fam, spec, pid, truth, m)

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

    def weave(self, item, fam, spec, pid, ns, statement, truth, mode, k, prior, weave_call) -> int:
        params = fam.params(spec)
        restrict = prior if (mode == "narrow" and prior) else None
        vars = {"problem": statement, "ns": ", ".join(map(str, ns)), "k": k,
                "params": (" (you may also use the names " + ", ".join(f"{p} = {v}" for p, v in params.items())
                           + ")") if params else "",
                "breadth": self._breadth_text(mode, k, prior), "feedback": self.feedback(fam)}
        template = "weave_menu" if self.cfg.menu else "weave_free"
        if self.cfg.menu:
            vars["bridges"] = prompts.bridge_menu(restrict)
            vars["lemmas"] = prompts.lemma_menu()
        ctx = {"step": "weave", "family": fam, "spec": spec, "ns": ns, "truth": truth, "k": k,
               "mode": mode, "prior": prior}
        bounds, call_id, ver = self.ask("weave", template, vars, "weave", ctx,
                                        lambda t: prompts.parse_weave(t, k))
        if not bounds:
            return 0
        bridge_ids = {b["id"] for b in prompts.bridges()}
        lemma_ids = {lm["id"] for lm in prompts.lemmas()}
        wins = 0
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
            wins += bool(res["win"])
            self.store.append("weaves", {
                "id": new_id("weave"), "when": now(), "run": self.run, "by": self.model_id,
                "problem": pid, "family": fam.name, "provenance": item["provenance"],
                "menu": self.cfg.menu, "breadth": self.cfg.breadth, "mode": mode, "weave_call": weave_call,
                "prior_bridges": prior, "in_prior": b["bridge_id"] in prior,
                "menu_ids_ok": (b["bridge_id"] in bridge_ids and all(x in lemma_ids for x in b["lemma_ids"]))
                if self.cfg.menu else None,
                **b, **res, "prediction": pred["id"], "call": call_id, "prompt_version": ver})
            self.log(f"  weave[{weave_call}/{mode}] {b['bridge_id']}+{'+'.join(b['lemma_ids'])}: "
                     f"g = {b['bound_expr']!r} -> {status}{' WIN' if res['win'] else ''}"
                     f" tight_at={res['tight_at']}")
        return wins

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
            if cid in self.seen:
                self.store.append("problems", {"event": "duplicate", "id": cid, **base})
                continue
            self.seen.add(cid)
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
