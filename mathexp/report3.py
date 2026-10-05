"""Run 3 reports (PROTOCOL.md "Run 3"): arm A4b (method surprise), arm A5b (cluster
notes with explanations) and reuse weighted by tightness.

Every definition here is the protocol's; the choices it left open are listed in
PROTOCOL.md "Run 3 build notes".
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import REPO, method
from .calibration import _fisher
from .report import sign_test, spearman
from .store import Store

#: A4b: bootstrap samples for every interval, seeded (PROTOCOL.md: 2000).
BOOT_SAMPLES = 2000
#: A4b policy replay: the share of problems each signal widens.
REPLAY_QS = (0.25, 0.50)
#: A5b: a child's overlap below this is "low" (transfer), else "high" (lookup).
LOW_OVERLAP = 0.5
#: Tight importance: the tie-break needs the pair used on at least this many problems;
#: the out-of-sample score needs at least this many counted, consistent bounds.
MIN_TIEBREAK_PROBLEMS = 3
MIN_OOS_BOUNDS = 3
RUN2_A4_STORES = (REPO / "store" / "run2-a4-fixed", REPO / "store" / "run2-a4-adaptive")


# ---- statistics ---------------------------------------------------------------------
def auc(scores: list[float], labels: list[bool]) -> float | None:
    """P(score of a positive > score of a negative), ties counting one half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    c = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return c / (len(pos) * len(neg))


def stratified_auc(scores: list[float], labels: list[bool], strata: list[str]) -> float | None:
    """AUC over (Y = 1, Y = 0) pairs in the same stratum only, pooled over strata."""
    num = den = 0.0
    for s in set(strata):
        idx = [i for i, x in enumerate(strata) if x == s]
        pos = [scores[i] for i in idx if labels[i]]
        neg = [scores[i] for i in idx if not labels[i]]
        num += sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
        den += len(pos) * len(neg)
    return num / den if den else None


def percentile_interval(xs: list[float], level: float = 0.95) -> tuple[float, float] | None:
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if len(xs) < 20:
        return None
    lo = xs[int(math.floor((1 - level) / 2 * (len(xs) - 1)))]
    hi = xs[int(math.ceil((1 + level) / 2 * (len(xs) - 1)))]
    return lo, hi


def boot_stratified_auc(scores, labels, strata, seed: int, samples: int = BOOT_SAMPLES):
    """Resample problems with replacement within each stratum (band); returns the list
    of finite bootstrap AUCs (samples with no within-stratum pair are dropped)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    keys = sorted(groups)
    out = []
    for _ in range(samples):
        idx = [i for k in keys for i in (rng.choice(groups[k]) for _ in groups[k])]
        a = stratified_auc([scores[i] for i in idx], [labels[i] for i in idx], [strata[i] for i in idx])
        if a is not None:
            out.append(a)
    return out


def logistic(X: list[list[float]], y: list[bool], ridge: float = 1e-6, iters: int = 100):
    """Newton-Raphson logistic regression. Returns (coefficients, converged); a
    coefficient beyond 30 in size is treated as separation (not converged)."""
    import numpy as np

    A = np.asarray(X, dtype=float)
    t = np.asarray(y, dtype=float)
    beta = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -40, 40)))
        W = p * (1 - p)
        H = A.T @ (A * W[:, None]) + ridge * np.eye(A.shape[1])
        g = A.T @ (t - p) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return beta.tolist(), False
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            return beta.tolist(), bool(np.max(np.abs(beta)) < 30)
    return beta.tolist(), False


def wilcoxon(diffs: list[float]) -> tuple[float | None, float | None, int]:
    """Exact two-sided Wilcoxon signed-rank test (zeros dropped, average ranks for
    ties, exact null distribution over the doubled ranks). Returns (W+, p, n)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return None, None, 0
    absd = [abs(d) for d in nz]
    order = sorted(range(n), key=lambda i: absd[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]
    w2 = sum(r for r, d in zip(r2, nz) if d > 0)
    total = sum(r2)
    dist = [0] * (total + 1)
    dist[0] = 1
    for r in r2:
        for s in range(total, r - 1, -1):
            dist[s] += dist[s - r]
    allc = 2 ** n
    lo = sum(dist[: w2 + 1]) / allc
    hi = sum(dist[w2:]) / allc
    return w2 / 2, min(1.0, 2 * min(lo, hi)), n


def _fmt(x, f="{:.3f}"):
    return "--" if x is None else f.format(x)


def _ci(iv):
    return "--" if iv is None else f"[{iv[0]:.3f}, {iv[1]:.3f}]"


def _done(store: Store) -> list[dict]:
    """The last 'done' row per (run, problem)."""
    last = {}
    for r in store.rows("problems"):
        if r.get("event") == "done":
            last[(r["run"], r["id"])] = r
    return list(last.values())


# ---- A4b ----------------------------------------------------------------------------
def _signal_values(ps: list[dict], seed: int, m0_pop: list[dict]) -> dict[str, list[float]]:
    """M, M0, V, R and Random for each problem in ``ps``. M0 is -log2 of the observed
    band's leave-one-out frequency in ``m0_pop`` (capped like M)."""
    counts = Counter(p["full"]["method"]["observed"] for p in m0_pop)
    n_pop = len(m0_pop)
    in_pop = {id(p) for p in m0_pop}
    rng = random.Random(seed)
    rand = {p["id"]: rng.random() for p in sorted(ps, key=lambda p: (p["run"], p["id"]))}
    out = defaultdict(list)
    for p in ps:
        f = p["full"]
        band = f["method"]["observed"]
        c, n = counts[band], n_pop
        if id(p) in in_pop:
            c, n = c - 1, n - 1
        freq = c / n if n > 0 else 0.0
        out["M"].append(f["method"]["M"])
        out["M0"].append(method.MAX_BITS if freq <= 0 else min(method.MAX_BITS, -math.log2(freq)))
        out["V"].append(p.get("surprise_max") if p.get("surprise_max") is not None else 0.0)
        r = f["probe"]["r"]
        out["R"].append(1e9 if r is None or not math.isfinite(r) else r)
        out["Random"].append(rand[p["id"]])
    return out


def _probe_ok(p: dict) -> bool:
    return bool(p.get("full") and p["full"].get("probe") and p["full"].get("method")
                and p["full"]["method"].get("observed"))


def a4b_populations(done: list[dict]) -> dict[str, list[dict]]:
    full = [p for p in done if p.get("breadth") == "full" and not p.get("predict_only")]
    completed = [p for p in full if p.get("completed") and _probe_ok(p)]
    primary = [p for p in completed if not p["full"]["p_from_method"] and p["full"]["probe"]["win"]
               and not p["full"]["probe"]["tight"]]
    return {"all": full, "completed": completed, "primary": primary,
            "prior_failed": [p for p in completed if not p["full"]["probe"]["win"]],
            "tight": [p for p in completed if p["full"]["probe"]["tight"]],
            "p_from_method": [p for p in completed if p["full"]["p_from_method"]]}


def _bounds_by_problem(weaves: list[dict]) -> dict[tuple, dict[str, list[dict]]]:
    out: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for w in weaves:
        out[(w["run"], w["problem"])][w.get("step", "weave")].append(w)
    return out


def _gain(ws: list[dict]) -> tuple[int, int]:
    """(counted wins, counted bounds tight at n_max)."""
    c = [w for w in ws if w.get("counted", True)]
    wins = sum(1 for w in c if w.get("win"))
    tight = sum(1 for w in c if w.get("status") == "consistent" and w.get("n_max") in (w.get("tight_at") or []))
    return wins, tight


def policy_replay(completed: list[dict], weaves: list[dict], signals: dict[str, list[float]],
                  seed: int) -> list[str]:
    """Each signal widens the top q of the problems whose probe won, plus every
    "prior failed" problem. Every policy pays for predict, both probes and mutate; a
    policy pays for the wide weave only where it widens; M also pays for its method
    call on every problem. Calls are the stored calls, retries included."""
    by = _bounds_by_problem(weaves)
    rng = random.Random(seed + 1)
    tiebreak = {p["id"]: rng.random() for p in completed}
    failed = [i for i, p in enumerate(completed) if not p["full"]["probe"]["win"]]
    rest = [i for i, p in enumerate(completed) if p["full"]["probe"]["win"]]
    lines = [f"  {len(completed)} completed problems: {len(failed)} prior failed (always widened), "
             f"{len(rest)} ranked by each signal"]
    lines.append(f"  {'policy':<22} {'widened':>7} {'calls':>6} {'wins':>5} {'tight':>5} "
                 f"{'wins/100':>9} {'tight/100':>9}")

    def account(widen: set[int], pays_method: bool) -> tuple[int, int, int]:
        calls = wins = tight = 0
        for i, p in enumerate(completed):
            cs = p.get("calls_by_step") or {}
            calls += sum(cs.get(s, 0) for s in ("predict", "probe", "reprobe", "mutate"))
            if pays_method:
                calls += cs.get("method", 0)
            b = by[(p["run"], p["id"])]
            g1, g2 = _gain(b["probe"]), _gain(b["reprobe"])
            wins += g1[0] + g2[0]
            tight += g1[1] + g2[1]
            if i in widen:
                calls += cs.get("wide", 0)
                gw = _gain(b["wide"])
                wins += gw[0]
                tight += gw[1]
        return calls, wins, tight

    def row(name, widen, pays_method=False):
        calls, wins, tight = account(widen, pays_method)
        rate = (lambda x: f"{100 * x / calls:.1f}" if calls else "--")
        lines.append(f"  {name:<22} {len(widen):>7} {calls:>6} {wins:>5} {tight:>5} {rate(wins):>9} {rate(tight):>9}")

    row("never (prior failed)", set(failed))
    row("always", set(range(len(completed))))
    for q in REPLAY_QS:
        k = int(q * len(rest) + 0.5)
        for name in ("M", "M0", "V", "R", "Random"):
            vals = signals[name]
            ranked = sorted(rest, key=lambda i: (-vals[i], tiebreak[completed[i]["id"]]))
            row(f"{name} q={int(q * 100)}%", set(failed) | set(ranked[:k]), pays_method=(name == "M"))
    return lines


def a4b_report(store: Store, seed: int = 0, run2_stores=RUN2_A4_STORES) -> str:
    done = _done(store)
    weaves = store.rows("weaves")
    calls = store.rows("calls")
    pops = a4b_populations(done)
    comp, prim = pops["completed"], pops["primary"]
    out = [f"store {store.dir}", f"A4b: {len(pops['all'])} problems processed, {len(comp)} completed, "
           f"{len(calls)} calls ({sum(1 for c in calls if c.get('parse_ok'))} parsed)",
           "models returned: " + (", ".join(f"{k} x{v}" for k, v in
                                           Counter(c.get("model") for c in calls).items()) or "--"), ""]

    out.append("== A4b: primary population ==")
    out.append(f"  completed {len(comp)}: P empty (bridge chosen by the method call) {len(pops['p_from_method'])}, "
               f"prior failed (probe 1 has no win) {len(pops['prior_failed'])}, probe 1 tight "
               f"{len(pops['tight'])}")
    pos = sum(1 for p in prim if p["full"]["Y"])
    out.append(f"  primary: {len(prim)} problems (win, not tight, P from step 2), Y = 1 on {pos}")
    bands_all = Counter(p["full"]["method"]["observed"] for p in comp)
    out.append("  observed band (completed): " + ", ".join(f"{b} {bands_all.get(b, 0)}" for b in method.BANDS))
    for b in method.BANDS:
        ps = [p for p in prim if p["full"]["method"]["observed"] == b]
        if ps:
            out.append(f"    primary {b:<9} n={len(ps):<3} Y=1 {sum(1 for p in ps if p['full']['Y'])}")
    trig = sum(1 for p in prim if p["full"]["method"]["M"] >= method.METHOD_SURPRISE_BITS)
    out.append(f"  M >= {method.METHOD_SURPRISE_BITS:g} bits: {trig} of {len(prim)} primary problems"
               + (f" ({100 * trig / len(prim):.0f}%)" if prim else ""))
    strata = Counter(p.get("stratum") for p in comp)
    out.append("  strata (completed): " + ", ".join(f"{k} {v}" for k, v in sorted(strata.items(), key=str)))
    probe_ws = [w for w in weaves if w.get("step") in ("probe", "reprobe")]
    wide_ws = [w for w in weaves if w.get("step") == "wide"]
    out.append(f"  labels: probe bounds outside P {sum(1 for w in probe_ws if not w.get('in_menu_shown', True))} "
               f"of {len(probe_ws)} (counted); wide bounds with an excluded label "
               f"{sum(1 for w in wide_ws if not w.get('counted', True))} of {len(wide_ws)} (not counted), with "
               f"an off-menu bridge or lemma id {sum(1 for w in wide_ws if not w.get('menu_ids_ok', True))}")
    out.append("")

    sig = _signal_values(prim, seed, prim)
    y = [bool(p["full"]["Y"]) for p in prim]
    bands = [p["full"]["method"]["observed"] for p in prim]
    out.append(f"== A4b: band-stratified AUC of M (primary; {BOOT_SAMPLES}-sample within-band bootstrap, seed {seed}) ==")
    a = stratified_auc(sig["M"], y, bands)
    boots = boot_stratified_auc(sig["M"], y, bands, seed)
    iv = percentile_interval(boots)
    verdict = "--" if iv is None else ("supported (interval above 0.5)" if iv[0] > 0.5 else
                                      "not supported (interval includes or is below 0.5)")
    out.append(f"  stratified AUC(M) = {_fmt(a)}  95% CI {_ci(iv)}  ({len(boots)} usable samples)  {verdict}")
    for name in ("M0", "V", "R", "Random"):
        out.append(f"  stratified AUC({name}) = {_fmt(stratified_auc(sig[name], y, bands))}")
    out.append("")

    out.append("== A4b: secondary measures ==")
    out += _logistic_lines(prim, sig, y, bands, seed)
    am, am0 = auc(sig["M"], y), auc(sig["M0"], y)
    rng = random.Random(seed + 2)
    diffs = []
    for _ in range(BOOT_SAMPLES):
        idx = [rng.randrange(len(prim)) for _ in prim]
        x1 = auc([sig["M"][i] for i in idx], [y[i] for i in idx])
        x0 = auc([sig["M0"][i] for i in idx], [y[i] for i in idx])
        if x1 is not None and x0 is not None:
            diffs.append(x1 - x0)
    d = None if am is None or am0 is None else am - am0
    out.append(f"  unstratified AUC(M) = {_fmt(am)}, AUC(M0) = {_fmt(am0)}, paired difference {_fmt(d, '{:+.3f}')} "
               f"95% CI {_ci(percentile_interval(diffs))}")
    out.append(f"  unstratified AUC(R) = {_fmt(auc(sig['R'], y))}, AUC(V) = {_fmt(auc(sig['V'], y))}, "
               f"AUC(Random) = {_fmt(auc(sig['Random'], y))}")
    out.append("")

    out.append("== A4b: direction of method surprise ==")
    dirs = [p["full"]["method"]["direction"] for p in prim]
    out.append("  " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(dirs).items())))
    for name, flag in (("disappointment", "disappointment"), ("pleasant surprise", "pleasant")):
        s = [m if dd == flag else 0.0 for m, dd in zip(sig["M"], dirs)]
        out.append(f"  AUC(M if {name}, else 0) = {_fmt(auc(s, y))}  stratified {_fmt(stratified_auc(s, y, bands))}")
    out.append("")

    out.append("== A4b: policy replay (counted wins and tight bounds per 100 calls) ==")
    if comp:
        csig = _signal_values(comp, seed, [p for p in comp if p["full"]["probe"]["win"]])
        out += policy_replay(comp, weaves, csig, seed)
    else:
        out.append("  (no completed problems)")
    out.append("")

    out.append("== A4b: replication of run 2 (non-prior win by value surprise) ==")
    by = _bounds_by_problem(weaves)
    cells = {}
    for flag in (True, False):
        ps = [p for p in comp if bool(p.get("surprising")) is flag]
        hit = [p for p in ps if _gain(by[(p["run"], p["id"])]["wide"])[0] > 0]
        cells[flag] = (len(hit), len(ps) - len(hit))
        out.append(f"  {'value-surprising' if flag else 'not surprising':<17} {len(hit)}/{len(ps)}")
    pf = _fisher(cells[True][0], cells[True][1], cells[False][0], cells[False][1])
    out.append(f"  Fisher p = {pf:.3g}")
    out.append("")

    out.append("== A4b: seeds and mutants (strata) ==")
    for st in ("seed", "mutant"):
        cs = [p for p in comp if p.get("stratum") == st]
        ps = [p for p in prim if p.get("stratum") == st]
        ss = _signal_values(ps, seed, prim)
        yy = [bool(p["full"]["Y"]) for p in ps]
        out.append(f"  {st:<7} completed {len(cs):<3} primary {len(ps):<3} Y=1 {sum(yy):<3} "
                   f"stratified AUC(M) {_fmt(stratified_auc(ss['M'], yy, [p['full']['method']['observed'] for p in ps]))}"
                   f"  prior failed {sum(1 for p in cs if not p['full']['probe']['win'])}")
    out.append("")
    out += tight_importance_section(store, done, weaves, run2_stores)
    return "\n".join(out)


def _logistic_lines(prim, sig, y, bands, seed) -> list[str]:
    rows = []
    for i, p in enumerate(prim):
        gf = p["full"]["probe"]["best_g"] - p["full"]["probe"]["f_max"]
        if gf > 0:
            rows.append((i, math.log(gf)))
    levels = sorted({bands[i] for i, _ in rows}, key=method.BANDS.index)

    def design(idx):
        X, yy = [], []
        for i, lg in idx:
            X.append([1.0] + [1.0 if bands[i] == b else 0.0 for b in levels[1:]] + [lg, sig["M"][i]])
            yy.append(y[i])
        return X, yy

    if len(rows) < 3 or len(set(y[i] for i, _ in rows)) < 2:
        return [f"  logistic Y ~ band + log(g - f) + M: too few problems or one outcome only (n = {len(rows)})"]
    X, yy = design(rows)
    beta, ok = logistic(X, yy)
    rng = random.Random(seed + 3)
    coefs, failed = [], 0
    for _ in range(BOOT_SAMPLES):
        smp = [rows[rng.randrange(len(rows))] for _ in rows]
        Xb, yb = design(smp)
        if len(set(yb)) < 2:
            failed += 1
            continue
        b, okb = logistic(Xb, yb)
        if okb:
            coefs.append(b[-1])
        else:
            failed += 1
    return [f"  logistic Y ~ band ({' / '.join(levels)}) + log(g - f) + M, n = {len(rows)}: coefficient of M "
            f"{beta[-1]:+.3f}{'' if ok else ' (did not converge: separation)'}  95% CI "
            f"{_ci(percentile_interval(coefs))}  ({len(coefs)} usable samples, {failed} dropped)"]


# ---- reuse weighted by tightness ---------------------------------------------------
def _pair(w: dict) -> tuple:
    return (w.get("bridge_id"), tuple(sorted(w.get("lemma_ids") or [])))


def _truth_lookup(done: list[dict]) -> dict[tuple, dict]:
    return {(p["run"], p["id"]): p.get("truth") or {} for p in done}


def _f_over_g(w: dict, truths: dict) -> float | None:
    g = method.g_at_nmax(w)
    if g is None:
        return None
    f = (truths.get((w["run"], w["problem"])) or {}).get(str(w["n_max"]))
    if f is None:
        return None
    if g <= 0:
        return 1.0 if f == 0 else None
    return f / g


def importance(weaves: list[dict], truths: dict) -> dict[tuple, dict]:
    """Per pair: raw (families with a win, problems with a win) and tight
    (families with a bound tight at n_max, tie-break: mean over the problems where the
    pair was used of its best f(n_max)/g(n_max), 0 for a problem with no consistent
    bound; None below MIN_TIEBREAK_PROBLEMS problems)."""
    d: dict[tuple, dict] = defaultdict(lambda: {"win_fam": set(), "win_prob": set(), "tight_fam": set(),
                                                 "best": defaultdict(float), "used": set()})
    for w in weaves:
        if not w.get("counted", True):
            continue
        x = d[_pair(w)]
        key = (w["run"], w["problem"])
        x["used"].add(key)
        if w.get("win"):
            x["win_fam"].add(w["family"])
            x["win_prob"].add(key)
        if w.get("status") == "consistent" and w.get("n_max") in (w.get("tight_at") or []):
            x["tight_fam"].add(w["family"])
        r = _f_over_g(w, truths)
        if r is not None:
            x["best"][key] = max(x["best"][key], r)
    out = {}
    for k, x in d.items():
        tb = (sum(x["best"].get(u, 0.0) for u in x["used"]) / len(x["used"])
              if len(x["used"]) >= MIN_TIEBREAK_PROBLEMS else None)
        out[k] = {"raw": (len(x["win_fam"]), len(x["win_prob"])), "tight": (len(x["tight_fam"]), tb),
                  "used": len(x["used"])}
    return out


def _raw_scalar(v):
    return v[0] * 1000 + v[1]


def _tight_scalar(v):
    return v[0] + 0.999 * (v[1] or 0.0)


def _rank_key_tight(v):
    return (-v[0], -(v[1] if v[1] is not None else -1.0))


def tight_importance_section(store: Store, done: list[dict], weaves: list[dict], run2_stores) -> list[str]:
    truths = _truth_lookup(done)
    out = ["== Tight importance (pair = bridge + sorted lemmas; A4b counted bounds) =="]
    imp = importance(weaves, truths)
    ranked = sorted(imp.items(), key=lambda kv: (_rank_key_tight(kv[1]["tight"]), kv[0]))
    for (b, lem), v in ranked[:12]:
        out.append(f"  {b:<20} {'+'.join(lem) or '-':<40} tight families={v['tight'][0]} "
                   f"tie-break={_fmt(v['tight'][1])} raw families={v['raw'][0]} problems={v['raw'][1]} used={v['used']}")
    if not ranked:
        out.append("  (no bounds)")
    wide = [w for w in weaves if w.get("step") == "wide"]
    wimp = importance(wide, truths)
    top3 = sorted(wimp.items(), key=lambda kv: (_rank_key_tight(kv[1]["tight"]), kv[0]))[:3]
    out.append("  top 3 on A4b's wide weaves: " + ("; ".join(f"{b}+{'+'.join(lem)}" for (b, lem), _ in top3) or "--"))
    out.append("  double_counting pair in that top 3: "
               + ("yes" if any(b == "double_counting" for (b, _), _ in top3) else "no" if top3 else "--"))
    out.append("")

    out.append("== Tight importance: out-of-sample Spearman against run 2's A4 stores ==")
    r2_done, r2_weaves, used = [], [], []
    for p in run2_stores:
        p = Path(p)
        if (p / "weaves.jsonl").exists():
            st = Store(p, "report")
            r2_done += _done(st)
            r2_weaves += st.rows("weaves")
            used.append(p.name)
    if not r2_weaves:
        out.append("  (run 2 stores not found)")
        return out
    r2 = importance(r2_weaves, _truth_lookup(r2_done))
    score: dict[tuple, list[float]] = defaultdict(list)
    for w in wide:
        if w.get("counted", True):
            r = _f_over_g(w, truths)
            if r is not None:
                score[_pair(w)].append(r)
    pairs = sorted(k for k, v in score.items() if len(v) >= MIN_OOS_BOUNDS)
    out.append(f"  ranked on {', '.join(used)} ({len(r2)} pairs); scored on A4b wide weaves: {len(pairs)} pairs with "
               f">= {MIN_OOS_BOUNDS} counted, consistent bounds")
    ys = [sum(score[k]) / len(score[k]) for k in pairs]
    zero = {"raw": (0, 0), "tight": (0, None)}
    for name, fn in (("raw", _raw_scalar), ("tight", _tight_scalar)):
        xs = [fn(r2.get(k, zero)[name]) for k in pairs]
        rho = spearman(xs, ys)
        out.append(f"  Spearman({name} importance on run 2, mean f/g on A4b wide) = {_fmt(rho, '{:+.3f}')}  n = {len(pairs)}")
    return out


# ---- A5b ------------------------------------------------------------------------------
def a5b_report(store: Store) -> str:
    done = [p for p in _done(store) if p.get("condition") and p.get("predict_only")]
    calls = store.rows("calls")
    out = [f"store {store.dir}", f"A5b: {len(done)} problem rows, {len(calls)} calls "
           f"({sum(1 for c in calls if c.get('parse_ok'))} parsed)",
           "models returned: " + (", ".join(f"{k} x{v}" for k, v in
                                           Counter(c.get("model") for c in calls).items()) or "--"), ""]
    bits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    info: dict[str, dict] = {}
    for p in done:
        if p.get("surprise") is not None:
            bits[p["condition"]][p["id"]].append(p["surprise"])
        info.setdefault(p["id"], p)  # overlap and position are the same in every condition
    mean_bits = {c: {k: sum(v) / len(v) for k, v in d.items()} for c, d in bits.items()}
    children = sorted(k for k, p in info.items() if p.get("cluster_role") == "child")
    clusters = {p.get("cluster") for p in info.values()}
    out.append("== A5b: clusters ==")
    out.append(f"  {len(clusters)} clusters, {len(info)} problems, {len(children)} children; conditions run: "
               + ", ".join(f"{c} ({len(mean_bits.get(c, {}))} scored)" for c in ("N0", "N1", "N2")))
    exp = [p for p in done if p["condition"] == "N2" and p.get("explanation")]
    out.append(f"  N2 explanations: {len(exp)} written, {sum(1 for p in exp if p.get('explanation_truncated'))} cut "
               f"to 80 words, explain calls {sum(1 for c in calls if c.get('step') == 'explain')}")
    out.append("")

    def paired(a: str, b: str, ids: list[str]) -> str:
        common = [k for k in ids if k in mean_bits.get(a, {}) and k in mean_bits.get(b, {})]
        diffs = [mean_bits[a][k] - mean_bits[b][k] for k in common]
        if not diffs:
            return f"{a} - {b}: no paired children"
        p = sign_test(diffs)
        w, pw, nw = wilcoxon(diffs)
        return (f"{a} - {b} over {len(diffs)} children: mean {sum(diffs) / len(diffs):+.3f} median "
                f"{statistics.median(diffs):+.3f} bits  (+{sum(1 for d in diffs if d > 0)} / "
                f"-{sum(1 for d in diffs if d < 0)} / 0 {sum(1 for d in diffs if d == 0)})  sign test p = "
                f"{_fmt(p, '{:.3g}')}  Wilcoxon W+ = {_fmt(w, '{:.1f}')} p = {_fmt(pw, '{:.3g}')} (n = {nw})")

    out.append("== A5b: paired N2 - N1 (primary; bits per child) ==")
    out.append("  " + paired("N2", "N1", children))
    out.append("")
    out.append(f"== A5b: overlap split (low overlap < {LOW_OVERLAP}: transfer; high: lookup) ==")
    low = [k for k in children if info[k].get("overlap") is not None and info[k]["overlap"] < LOW_OVERLAP]
    high = [k for k in children if info[k].get("overlap") is not None and info[k]["overlap"] >= LOW_OVERLAP]
    out.append(f"  low overlap {len(low)} children, high overlap {len(high)}")
    for label, ids in (("low ", low), ("high", high)):
        for a, b in (("N1", "N0"), ("N2", "N0"), ("N2", "N1")):
            out.append(f"  {label} {paired(a, b, ids)}")
    out.append("")
    out.append("== A5b: position (mean bits by position in the cluster; 1 = parent) ==")
    positions = sorted({p.get("cluster_pos") for p in info.values() if p.get("cluster_pos")})
    out.append("  " + f"{'cond':<5}" + "".join(f"{f'pos {q}':>16}" for q in positions))
    for c in ("N0", "N1", "N2"):
        cells = []
        for q in positions:
            v = [mean_bits.get(c, {}).get(k) for k, p in info.items() if p.get("cluster_pos") == q]
            v = [x for x in v if x is not None]
            cells.append(f"{sum(v) / len(v):.3f} (n={len(v)})" if v else "--")
        out.append("  " + f"{c:<5}" + "".join(f"{x:>16}" for x in cells))
    return "\n".join(out)
