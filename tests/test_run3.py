"""Run 3: clusters, method_band, breadth full (A4b), cluster notes (A5b), the reports and
the cloud entry point -- all against the mock backend."""

import json
import math

import pytest

from mathexp import clusters as cl
from mathexp import method, prompts, report3
from mathexp.__main__ import main
from mathexp.llm import MockBackend
from mathexp.loop import Config, Loop, overlap, prior_full
from mathexp.problems import get_family
from mathexp.store import Refusal, Store
from mathexp.truth import TruthCache

A, B = get_family("setfam"), get_family("addfree")


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _done(store_dir):
    return [r for r in _rows(store_dir / "problems.jsonl") if r["event"] == "done"]


# ---- clusters ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cache = TruthCache(tmp_path_factory.mktemp("c") / "truth.json")
    return cl.build_clusters(cache=cache, parents=2, children=4, draws=50, seed=0, families=("setfam",),
                             log=lambda *_: None), cache


def test_cluster_rules(built):
    data, cache = built
    run2 = cl.run2_ids()
    assert len(run2) == 40 and len(data["clusters"]) == 2
    elig = [r["id"] for r in cl.eligible("setfam", exclude=run2)]
    ids = [m["id"] for c in data["clusters"] for m in c["members"]]
    assert len(ids) == len(set(ids)) == 10  # 2 clusters of 5, no member in two clusters
    assert not set(ids) & run2
    for c in data["clusters"]:
        parent, kids = c["members"][0], c["members"][1:]
        assert parent["role"] == "parent" and parent["id"] in elig[:2 + len(data["replacements"])]
        assert [m["cluster_pos"] for m in c["members"]] == [1, 2, 3, 4, 5]
        for k in kids:
            assert k["role"] == "child" and len(k["changed"]) == 1
            assert len(k["truth"]) >= 4 and k["truth"] != k["trivial"]
            assert A.problem_id(k["spec"]) == k["id"]
            # every value and a witness at n_max went through the cache
            v, w = cache.get_witness(A, k["spec"], k["ns"][-1])
            assert v == k["truth"][-1] and len(w) == v and A.check_witness(k["spec"], k["ns"][-1], w) == []
        for i, a in enumerate(c["members"]):
            for b in c["members"][i + 1:]:
                assert not cl.same_sequence(a, b)
    # parents are the top eligible specs by scan score, in order, outside run 2's 40
    parents = sorted((c["members"][0] for c in data["clusters"]), key=lambda m: m["scan_rank"])
    assert [p["id"] for p in parents] == elig[:2]


def test_cluster_order_is_seeded(built, tmp_path):
    data, _ = built
    again = cl.build_clusters(cache=TruthCache(tmp_path / "t.json"), parents=2, seed=0, families=("setfam",),
                              log=lambda *_: None)
    assert [c["cluster"] for c in again["clusters"]] == [c["cluster"] for c in data["clusters"]]
    assert [[m["id"] for m in c["members"]] for c in again["clusters"]] == \
           [[m["id"] for m in c["members"]] for c in data["clusters"]]
    assert [c["order"] for c in again["clusters"]] == [1, 2]


def test_cluster_replacement_is_listed(tmp_path):
    # one draw cannot yield 4 qualifying children: every parent is replaced and listed
    data = cl.build_clusters(cache=TruthCache(tmp_path / "t.json"), parents=1, children=4, draws=1, seed=0,
                             families=("setfam",), log=lambda *_: None)
    assert not data["clusters"]
    assert len(data["replacements"]) == len(cl.eligible("setfam", exclude=cl.run2_ids()))
    assert all("1 draws" in r["reason"] or "only" in r["reason"] for r in data["replacements"])
    assert "replacements: 19" in cl.summary_text(data)


def test_cluster_child_excluded_when_run2(built, tmp_path, monkeypatch):
    data, _ = built
    child = data["clusters"][0]["members"][1]["id"]
    orig = cl.run2_ids()
    monkeypatch.setattr(cl, "run2_ids", lambda scan_dir=None: orig | {child})
    again = cl.build_clusters(cache=TruthCache(tmp_path / "t.json"), parents=2, seed=0, families=("setfam",),
                              log=lambda *_: None)
    assert child not in {m["id"] for c in again["clusters"] for m in c["members"]}


def test_a4b_seeds():
    seeds = cl.a4b_seeds()
    run2 = cl.run2_ids()
    fams = [s["family"] for s in seeds]
    assert fams == ["setfam"] * 19 + ["addfree"] * 20
    assert not {s["id"] for s in seeds} & run2
    add = [s for s in seeds if s["family"] == "addfree"]
    assert [s["family_rank"] for s in add] == list(range(21, 41))  # the next 20 after run 2's 20
    assert [s["rank"] for s in seeds] == list(range(1, 40))


# ---- method_band ------------------------------------------------------------------
def test_bands_and_bits():
    assert method.BAND_CUTS == (1.0, 1.25, 2.0) and method.METHOD_SURPRISE_BITS == 3.0
    assert [method.band_of(r) for r in (1.0, 1.1, 1.25, 1.26, 2.0, 2.01, math.inf, None)] == \
           ["tight", "close", "close", "moderate", "moderate", "loose", "loose", "failed"]
    d = {"tight": 0.5, "close": 0.25, "moderate": 0.125, "loose": 0.125, "failed": 0.0}
    assert method.band_bits(d, "moderate")[0] == pytest.approx(3.0)
    assert method.band_bits(d, "failed")[0] == method.MAX_BITS
    assert method.median_band(d) == "tight"
    assert method.direction(d, "loose") == "disappointment" and method.direction(d, "tight") == "as_expected"
    assert method.direction({**d, "tight": 0.0, "close": 0.75}, "tight") == "pleasant"
    # uniform forecast: 2.32 bits on every band
    assert method.band_bits({b: 0.2 for b in method.BANDS}, "loose")[0] == pytest.approx(2.3219, abs=1e-3)
    # floored ratio: 7.5 bounds an integer 7 tightly
    assert method.g_at_nmax({"status": "consistent", "n_max": 5, "values": {"5": 7.5}}) == 7
    assert method.ratio(0, 0) == 1.0 and method.ratio(1, 0) == math.inf


def test_bridge_groups():
    assert method.excluded_bridges(["multilinear_poly"]) == {"charvec_fp", "multilinear_poly", "incidence_matrix"}
    assert method.excluded_bridges(["slice_rank", "translates"]) == {"fourier_indicator", "slice_rank", "translates"}
    assert method.bridge_group("double_counting") == {"double_counting"}


def test_parse_method():
    ok = '{"bridge_id": null, "bands": {"tight": 0.2, "close": 0.2, "moderate": 0.2, "loose": 0.2, "failed": 0.4}}'
    p = prompts.parse_method(ok)
    assert sum(p["bands"].values()) == pytest.approx(1.0) and p["normalized"] and p["bridge_id"] is None
    with pytest.raises(prompts.ParseError):
        prompts.parse_method('{"bridge_id": null, "bands": {"tight": 1.0}}')
    with pytest.raises(prompts.ParseError):  # P empty: the bridge must be on the menu
        prompts.parse_method(ok.replace("null", '"made_up"'), choose_from=["charvec_fp"])
    assert prompts.parse_method(ok.replace("null", '"charvec_fp"'), choose_from=["charvec_fp"])["bridge_id"] == \
           "charvec_fp"


def test_parse_explain_cuts_to_80_words():
    p = prompts.parse_explain(json.dumps({"explanation": " ".join(["w"] * 100)}))
    assert p["truncated"] and p["words"] == 100 and len(p["explanation"].split()) == 80


# ---- breadth full (A4b) ----------------------------------------------------------------
def _seed_file(tmp_path, n_setfam=3, n_addfree=2):
    seeds = [s for s in cl.a4b_seeds() if s["family"] == "setfam"][:n_setfam] + \
            [s for s in cl.a4b_seeds() if s["family"] == "addfree" and s["ns"][-1] <= 20][:n_addfree]
    p = tmp_path / "seeds.jsonl"
    p.write_text("".join(json.dumps(s) + "\n" for s in seeds), encoding="utf-8")
    return p


def test_prior_full_rules(tmp_path):
    st = Store(tmp_path / "s", "mock")

    def w(problem, bridge, g, family="setfam", win=True, counted=True):
        st.append("weaves", {"run": "r", "problem": problem, "bridge_id": bridge, "family": family, "win": win,
                             "by": "m", "status": "consistent", "n_max": 6, "values": {"6": g}, "counted": counted})

    w("par", "double_counting", 9)
    w("par", "charvec_fp", 7.5)
    w("par", "translates", 8)
    w("par", "probabilistic", 1, counted=False)  # an excluded wide label never enters P
    w("par", "not_a_menu_id", 1)                  # nor does a free-form label
    assert prior_full(st, "setfam", "par", "m") == (["charvec_fp", "translates"], "parent")
    assert prior_full(st, "setfam", "nobody", "m")[1] == "family"
    assert len(prior_full(st, "setfam", "nobody", "m")[0]) <= 2
    assert prior_full(Store(tmp_path / "e", "mock"), "setfam", None, "m") == ([], "none")


def test_prior_full_fills_from_the_family_ranking(tmp_path):
    """PROTOCOL.md A4b step 2 is an ordered fill up to 2: the parent's winning bridges,
    then the family's ranking (run 2's rule), skipping bridges already in P."""
    st = Store(tmp_path / "s", "mock")

    def w(problem, bridge, g, family="setfam", win=True):
        st.append("weaves", {"run": "r", "problem": problem, "bridge_id": bridge, "family": family, "win": win,
                             "by": "m", "status": "consistent", "n_max": 6, "values": {"6": g}, "counted": True})

    # family ranking: double_counting wins on 3 problems, translates on 2, charvec_fp on 1
    for i in range(3):
        w(f"x{i}", "double_counting", 9)
    for i in range(2):
        w(f"y{i}", "translates", 9)
    w("par", "charvec_fp", 7)            # the parent's only win
    w("par", "probabilistic", 5, win=False)
    assert prior_full(st, "setfam", "par", "m") == (["charvec_fp", "double_counting"], "parent+family")
    # the parent's win is the family's top bridge: the fill takes the next one
    w("par2", "double_counting", 8)
    assert prior_full(st, "setfam", "par2", "m") == (["double_counting", "translates"], "parent+family")
    # a seed (no parent) gets the family's top 2
    assert prior_full(st, "setfam", None, "m") == (["double_counting", "translates"], "family")
    # a parent with no win falls through to the family's ranking
    assert prior_full(st, "setfam", "x-none", "m") == (["double_counting", "translates"], "family")
    # another family's wins never fill this family's P
    assert prior_full(st, "addfree", None, "m") == ([], "none")


def test_full_breadth_end_to_end(tmp_path, capsys):
    seeds = _seed_file(tmp_path)
    store = tmp_path / "a4b"
    assert main(["run", "--backend", "mock", "--breadth", "full", "--seeds-from", str(seeds), "--exclude-run2",
                 "--max-calls", "60", "--max-problems", "8", "--store", str(store), "--seed", "1"]) == 0
    done = _done(store)
    comp = [p for p in done if p.get("completed")]
    assert len(comp) >= 5 and all(p["policy"] == "random" for p in done)
    weaves = _rows(store / "weaves.jsonl")
    preds = _rows(store / "predictions.jsonl")
    calls = _rows(store / "calls.jsonl")
    assert all(c["problem"] and c["model"] == "mock-1" for c in calls)
    for p in comp:
        f = p["full"]
        assert set(p["calls_by_step"]) >= {"predict", "method", "probe", "reprobe", "wide"}
        assert 1 <= len(f["P"]) <= method.P_CAP
        ws = [w for w in weaves if w["problem"] == p["id"]]
        probes = [w for w in ws if w["step"] in ("probe", "reprobe")]
        wide = [w for w in ws if w["step"] == "wide"]
        assert len([w for w in ws if w["step"] == "probe"]) == 1 and len(wide) <= 3
        assert all(w["bridge_id"] in f["P"] for w in probes)  # menu restricted to P
        excl = method.excluded_bridges(f["P"])
        assert set(f["wide_menu"]).isdisjoint(excl)
        for w in wide:
            assert w["counted"] == (w["bridge_id"] not in excl)
        # the method row precedes the probe's bound row in the ledger
        ids = [r.get("kind") for r in preds if r.get("problem") == p["id"]]
        assert ids.index("method_band") < ids.index("bound_holds")
        assert ids.count("method_band") == ids.count("method_band_top") == 1
        m = f["method"]
        assert m["observed"] == f["probe"]["band"] and m["M"] == pytest.approx(-math.log2(m["bands"][m["observed"]]))
        assert isinstance(f["Y"], bool)
    # the first problem of each family has no prior: the method call chose P
    assert any(p["full"]["p_from_method"] and p["prior_source"] == "method" for p in comp)
    assert any(w["counted"] is False for w in weaves)
    # outcome rows score the method forecast with bits; the top band is a binary row
    outs = {r["ref"]: r for r in preds if r.get("kind") == "outcome"}
    mb = [r for r in preds if r.get("kind") == "method_band"]
    assert mb and all(outs[r["id"]]["bits"] is not None for r in mb if outs[r["id"]]["outcome"] == "right")
    top = [r for r in preds if r.get("kind") == "method_band_top"]
    assert {outs[r["id"]]["outcome"] for r in top} <= {"right", "wrong"}
    capsys.readouterr()
    assert main(["report3", "--a4b", str(store)]) == 0
    out = capsys.readouterr().out
    for section in ("A4b: primary population", "band-stratified AUC of M", "A4b: secondary measures",
                    "logistic Y ~ band", "AUC(M0)", "direction of method surprise",
                    "policy replay", "M q=25%", "Random q=50%", "replication of run 2", "seeds and mutants",
                    "Tight importance", "out-of-sample Spearman"):
        assert section in out, section
    assert main(["report", "--store", str(store)]) == 0
    assert "method_band_top" in capsys.readouterr().out


def test_probe_refused_without_method_row(tmp_path, monkeypatch):
    seeds = _seed_file(tmp_path, 1, 0)
    store = Store(tmp_path / "s", "mock")
    real = store.predict

    def drop_method(**kw):
        if kw["kind"] == "method_band":
            return {"id": "never-written"}
        return real(**kw)

    monkeypatch.setattr(store, "predict", drop_method)
    loop = Loop(Config(backend=MockBackend(), store=store, breadth="full", policy="random", seeds_from=str(seeds),
                       max_calls=10, log=lambda *_: None))
    with pytest.raises(Refusal, match="method_band"):
        loop.run_loop()
    assert not (tmp_path / "s" / "weaves.jsonl").exists()


def test_full_breadth_refuses_surprise_policy(tmp_path):
    assert main(["run", "--backend", "mock", "--breadth", "full", "--policy", "surprise",
                 "--store", str(tmp_path / "s")]) == 2


def test_run2_mutants_excluded(tmp_path):
    seeds = _seed_file(tmp_path, 2, 0)
    s1 = Store(tmp_path / "s1", "mock")
    Loop(Config(backend=MockBackend(seed=4), store=s1, breadth="full", policy="random", seeds_from=str(seeds),
                max_calls=14, max_problems=2, log=lambda *_: None)).run_loop()
    kids = {r["id"] for r in s1.rows("problems") if r["event"] == "queued" and r.get("parent")}
    assert kids
    s2 = Store(tmp_path / "s2", "mock")
    Loop(Config(backend=MockBackend(seed=4), store=s2, breadth="full", policy="random", seeds_from=str(seeds),
                max_calls=14, max_problems=2, exclude=frozenset(kids), log=lambda *_: None)).run_loop()
    ev = [r for r in s2.rows("problems") if r["event"] == "excluded"]
    assert {r["id"] for r in ev} == kids
    assert not {r["id"] for r in s2.rows("problems") if r["event"] == "queued" and r.get("parent")} & kids


# ---- A5b ---------------------------------------------------------------------------
@pytest.fixture
def cluster_file(built, tmp_path):
    data, _ = built
    p = tmp_path / "clusters.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_overlap():
    assert overlap({3: 1, 4: 2, 5: 3}, [{3: 1, 4: 9}, {4: 2, 5: 3, 9: 9}]) == 1.0
    assert overlap({3: 1, 4: 2}, [{3: 2, 4: 2}]) == 0.5
    assert overlap({3: 1}, [{4: 1}]) is None


def test_a5b_conditions_end_to_end(cluster_file, tmp_path, capsys):
    store = tmp_path / "a5b"
    for label, extra in (("N0", []), ("N1", ["--notes", "on", "--notes-scope", "cluster"]),
                         ("N2", ["--notes", "on", "--notes-scope", "cluster", "--explain", "on"])):
        assert main(["run", "--backend", "mock", "--predict-only", "--clusters", str(cluster_file),
                     "--store", str(store), "--run-id", f"a5b-{label}", "--max-calls", "40", *extra]) == 0
    done = _done(store)
    by = {c: [p for p in done if p["condition"] == c] for c in ("N0", "N1", "N2")}
    assert all(len(v) == 10 for v in by.values())
    order = [p["id"] for p in by["N0"]]
    assert order == [p["id"] for p in by["N1"]] == [p["id"] for p in by["N2"]]
    assert [p["cluster_pos"] for p in by["N0"]] == [1, 2, 3, 4, 5] * 2
    # notes: only the same cluster's earlier members
    for p in by["N1"]:
        assert p["notes_shown"] == (p["cluster_pos"] - 1)
    assert all(p["notes_shown"] == 0 for p in by["N0"])
    calls = _rows(store / "calls.jsonl")
    exp = [c for c in calls if c["step"] == "explain"]
    assert len(exp) == 2 * 4  # every member but each cluster's last child
    last = [p for p in by["N2"] if p["cluster_pos"] == p["cluster_size"]]
    assert all(p["explanation"] is None and not p.get("explain_call") for p in last)
    assert all(len(p["explanation"].split()) <= 80 for p in by["N2"] if p["explanation"])
    assert any(p.get("explanation_truncated") for p in by["N2"])
    # the explain prompt shows statement, true values and the witness at n_max
    assert all("extremal family F at n = " in c["prompt"] for c in exp)
    # N2's notes carry the explanations; N1's do not
    n2_prompts = [c["prompt"] for c in calls if c["step"] == "predict" and c["run"] == "a5b-N2"]
    n1_prompts = [c["prompt"] for c in calls if c["step"] == "predict" and c["run"] == "a5b-N1"]
    assert sum("Explanation:" in t for t in n2_prompts) == 8 and not any("Explanation:" in t for t in n1_prompts)
    # overlap is the same in every condition, and defined for children only
    for c in ("N1", "N2"):
        assert [p["overlap"] for p in by[c]] == [p["overlap"] for p in by["N0"]]
    assert all((p["overlap"] is None) == (p["cluster_role"] == "parent") for p in by["N0"])
    capsys.readouterr()
    assert main(["report3", "--a5b", str(store)]) == 0
    out = capsys.readouterr().out
    for section in ("A5b: clusters", "paired N2 - N1", "sign test p", "Wilcoxon", "overlap split",
                    "low  N1 - N0", "high N2 - N1", "position", "pos 5"):
        assert section in out, section


# ---- statistics ---------------------------------------------------------------------------
def test_auc_and_stratified():
    assert report3.auc([3, 2, 1], [True, False, False]) == 1.0
    assert report3.auc([1, 1], [True, False]) == 0.5
    # a pair across strata is never compared
    s, y, b = [5, 1, 4, 0], [True, False, False, True], ["loose", "loose", "close", "close"]
    assert report3.stratified_auc(s, y, b) == pytest.approx(0.5)  # loose 1.0, close 0.0
    assert report3.auc(s, y) == pytest.approx(0.5)
    assert report3.stratified_auc([1, 2], [True, True], ["a", "a"]) is None
    boots = report3.boot_stratified_auc([5, 1, 6, 0, 3, 2], [True, False, True, False, True, False],
                                        ["a", "a", "a", "b", "b", "b"], seed=0, samples=200)
    assert boots == report3.boot_stratified_auc([5, 1, 6, 0, 3, 2], [True, False, True, False, True, False],
                                                ["a", "a", "a", "b", "b", "b"], seed=0, samples=200)
    assert all(0 <= x <= 1 for x in boots)


def test_wilcoxon_exact():
    w, p, n = report3.wilcoxon([1, 2, 3, 4, 5])
    assert (w, n) == (15, 5) and p == pytest.approx(2 / 32)
    w, p, n = report3.wilcoxon([1, -2, 3, 0])
    assert n == 3 and w == 4 and p == pytest.approx(0.75)  # subset sums 0,1,2,3,3,4,5,6: P(W >= 4) = 3/8


def test_logistic_recovers_direction():
    import random as _r

    rng = _r.Random(0)
    X, y = [], []
    for _ in range(300):
        x = rng.uniform(-2, 2)
        X.append([1.0, x])
        y.append(rng.random() < 1 / (1 + math.exp(-2 * x)))
    beta, ok = report3.logistic(X, y)
    assert ok and 1.4 < beta[1] < 2.8


def test_tight_importance():
    truths = {("r", "p1"): {"6": 4}, ("r", "p2"): {"6": 4}, ("r", "q1"): {"10": 3}}

    def w(problem, family, bridge, g, nm, tight, win=True):
        return {"run": "r", "problem": problem, "family": family, "bridge_id": bridge, "lemma_ids": ["b", "a"],
                "status": "consistent", "n_max": nm, "values": {str(nm): g}, "tight_at": [nm] if tight else [],
                "win": win}

    ws = [w("p1", "setfam", "x", 4, 6, True), w("q1", "addfree", "x", 3, 10, True), w("p2", "setfam", "x", 8, 6, False),
          w("p1", "setfam", "dc", 5, 6, False), w("q1", "addfree", "dc", 6, 10, False)]
    imp = report3.importance(ws, truths)
    assert imp[("x", ("a", "b"))]["tight"][0] == 2 and imp[("x", ("a", "b"))]["tight"][1] == pytest.approx((1 + 1 + 0.5) / 3)
    assert imp[("dc", ("a", "b"))]["tight"] == (0, None)  # used on 2 problems: no tie-break
    assert imp[("dc", ("a", "b"))]["raw"] == (2, 2)


# ---- the cloud entry point ----------------------------------------------------------------
def test_run3_mock_end_to_end(tmp_path, capsys):
    out = tmp_path / "run3"
    rc = main(["run3", "--backend", "mock", "--out", str(out), "--parents", "1", "--children", "2", "--draws", "20",
               "--budget-a5b", "30", "--budget-a4b", "25", "--max-problems", "3", "--no-precompute-seeds",
               "--cache", str(tmp_path / "cache" / "truth.json")])
    assert rc == 0
    for f in ("clusters.json", "a4b_seeds.jsonl", "build.txt", "probe_call.json", "a5b_report.txt",
              "a4b_report.txt", "run3_summary.json", "run.log", "a5b/report.txt", "a4b/report.txt"):
        assert (out / f).exists(), f
    a5b = (out / "a5b_report.txt").read_text(encoding="utf-8")
    a4b = (out / "a4b_report.txt").read_text(encoding="utf-8")
    for s in ("A5b: clusters", "paired N2 - N1", "overlap split", "position"):
        assert s in a5b
    for s in ("primary population", "band-stratified AUC", "secondary measures", "direction", "policy replay",
              "replication of run 2", "seeds and mutants", "Tight importance", "out-of-sample Spearman"):
        assert s in a4b
    summ = json.loads((out / "run3_summary.json").read_text(encoding="utf-8"))
    assert summ["a5b"]["clusters"] == 2 and summ["a5b"]["calls"] <= 30
    assert summ["a4b"]["calls"] <= 25 and summ["probe"]["ok"]


def test_run3_refuses_alias_and_aborts_on_failed_probe(tmp_path, monkeypatch, capsys):
    from mathexp import run3 as r3

    with pytest.raises(r3.Run3Abort):
        r3.check_model_id("claude", "sonnet")
    r3.check_model_id("claude", "claude-sonnet-5-5")

    class Broken(MockBackend):
        def complete(self, *a, **k):
            from mathexp.llm import LLMError
            raise LLMError("not logged in")

    monkeypatch.setattr(r3, "make_backend", lambda *a, **k: Broken())
    monkeypatch.setattr(r3, "build", lambda *a, **k: {})
    rc = main(["run3", "--backend", "mock", "--out", str(tmp_path / "o")])
    assert rc == 3
    assert "probe call to the mock backend failed" in capsys.readouterr().err
    assert not (tmp_path / "o" / "a5b").exists()


# ---- review fixes (judge, before the cloud run) -----------------------------------------
def test_a5b_plan_counts_four_s_minus_c():
    """N0 s + N1 s + N2 (s predicts + s - c explains) = 4s - c; a budget below that for
    all clusters must run fewer clusters, never starve N0 or N2."""
    from mathexp.run3 import a5b_plan

    data = {"clusters": [{"order": i, "members": [0] * 5} for i in range(12)]}
    assert a5b_plan(data, 228, None) == (12, [60, 60, 108])
    assert a5b_plan(data, 285, None) == (12, [60, 60, 108])
    assert a5b_plan(data, 227, None)[0] == 11
    assert a5b_plan(data, 200, None) == (10, [50, 50, 90])
    for b in range(0, 300, 7):
        c, nom = a5b_plan(data, b, None)
        assert sum(nom) <= b


def _run3_args(out, cache, *extra):
    return ["run3", "--backend", "mock", "--out", str(out), "--parents", "1", "--children", "2", "--draws", "20",
            "--budget-a5b", "30", "--budget-a4b", "25", "--max-problems", "3", "--no-precompute-seeds",
            "--cache", str(cache), *extra]


def test_run3_refuses_a_rerun_into_a_used_out_dir(tmp_path, capsys):
    out, cache = tmp_path / "run3", tmp_path / "cache" / "truth.json"
    assert main(_run3_args(out, cache)) == 0
    before = {p.name: p.read_bytes() for p in (out / "a5b").glob("*.jsonl")}
    summary = (out / "run3_summary.json").read_bytes()
    capsys.readouterr()
    assert main(_run3_args(out, cache)) == 3
    assert "does not resume" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in (out / "a5b").glob("*.jsonl")} == before
    assert (out / "run3_summary.json").read_bytes() == summary


def test_run3_phases_build_only_what_they_need(tmp_path):
    cache = tmp_path / "cache" / "truth.json"
    a = tmp_path / "a5b-only"
    assert main(_run3_args(a, cache, "--phase", "a5b")) == 0
    assert (a / "clusters.json").exists() and not (a / "a4b_seeds.jsonl").exists()
    assert (a / "a5b_report.txt").exists() and not (a / "a4b").exists()
    assert json.loads((a / "run3_summary-a5b.json").read_text(encoding="utf-8"))["a5b"]["clusters"] == 2
    b = tmp_path / "a4b-only"
    assert main(_run3_args(b, cache, "--phase", "a4b")) == 0
    assert (b / "a4b_seeds.jsonl").exists() and not (b / "clusters.json").exists()
    assert (b / "a4b_report.txt").exists() and not (b / "a5b").exists()
    # the A4b phase can then run in the A5b phase's directory without touching A5b's files
    a5b_rows = (a / "a5b" / "calls.jsonl").read_bytes()
    assert main(_run3_args(a, cache, "--phase", "a4b")) == 0
    assert (a / "a5b" / "calls.jsonl").read_bytes() == a5b_rows and (a / "a4b_report.txt").exists()
    assert main(_run3_args(a, cache, "--phase", "a4b")) == 3  # but not twice


def test_run3_stops_when_the_backend_is_down(tmp_path, monkeypatch, capsys):
    """After the probe call, a backend that keeps failing stops the run after
    MAX_BACKEND_ERRORS calls instead of spending the budget; the reports still exist."""
    from mathexp import run3 as r3
    from mathexp.llm import LLMError

    class DiesAfterPing(MockBackend):
        def complete(self, system, prompt, json_schema=None, context=None):
            if context and context.get("step") != "ping":
                raise LLMError("usage limit reached")
            return super().complete(system, prompt, json_schema, context)

    monkeypatch.setattr(r3, "make_backend", lambda *a, **k: DiesAfterPing())
    out = tmp_path / "run3"
    assert main(_run3_args(out, tmp_path / "c" / "truth.json")) == 4
    assert "backend errors in a row" in capsys.readouterr().err
    calls = _rows(out / "a5b" / "calls.jsonl")
    assert len(calls) == r3.MAX_BACKEND_ERRORS and not any(c["parse_ok"] for c in calls)
    assert (out / "a5b_report.txt").exists() and not (out / "a4b").exists()
    assert "aborted" in json.loads((out / "run3_summary.json").read_text(encoding="utf-8"))


def test_get_witness_never_overwrites_a_known_value(tmp_path):
    """A value cached without a witness (an older cache) whose witness recompute times
    out under a smaller limit keeps its value."""
    cache = TruthCache(tmp_path / "truth.json")
    spec = {"p": 2, "size_mod": [1], "int_mod": [0], "ns": [3, 4, 5, 6]}
    key = f"{A.problem_id(spec)}:6"
    cache.data[key] = {"value": 999, "seconds": 1.0, "time_limit": 20.0, "family": "setfam"}

    class Slow(type(A)):
        def truth_witness(self, spec, n, time_limit):
            return None, None

    v, w = cache.get_witness(Slow(), spec, 6, 0.001)
    assert (v, w) == (999, None) and cache.data[key]["value"] == 999


def test_store_reads_past_a_truncated_last_line(tmp_path):
    st = Store(tmp_path / "s", "mock")
    st.append("calls", {"a": 1})
    with st.path("calls").open("a", encoding="utf-8") as fh:
        fh.write('{"a": 2, "b": ')  # killed mid-append
    assert st.rows("calls") == [{"a": 1}]
