import json

from mathexp.__main__ import main
from mathexp.store import FILES


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_mock_run_then_report(tmp_path, capsys):
    store = tmp_path / "store"
    assert main(["run", "--backend", "mock", "--max-calls", "9", "--store", str(store),
                 "--seeds", "oddtown,cap_set,sidon_int", "--breadth", "fixed"]) == 0
    for name in FILES:
        assert (store / f"{name}.jsonl").exists(), name
    calls = _rows(store / "calls.jsonl")
    assert len(calls) == 9 and all(c["parse_ok"] for c in calls)
    weaves = _rows(store / "weaves.jsonl")
    statuses = {w["status"] for w in weaves}
    assert {"consistent", "violated", "error"} <= statuses
    assert any(w["win"] for w in weaves)
    assert any(w["status"] == "consistent" and not w["win"] for w in weaves)  # valid but loose
    preds = _rows(store / "predictions.jsonl")
    kinds = {p["kind"] for p in preds}
    assert {"interval", "bound_holds", "mutation_effect", "recognition", "outcome"} <= kinds
    # every prediction row was written before its outcome row
    seen = set()
    for p in preds:
        if p["kind"] == "outcome":
            assert p["ref"] in seen
        else:
            assert p["predicted_first"] and p["outcome"] is None
            seen.add(p["id"])
    capsys.readouterr()
    assert main(["report", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    for section in ("Calibration", "Top surprises", "Weave status", "importance", "A4", "What-if tree"):
        assert section in out


def test_adaptive_paths(tmp_path, capsys):
    """Mock: problem 1 is predicted exactly (unsurprising) -> narrow -> no win ->
    escalate; problem 2 is overestimated 4x (surprising) -> wide; problem 3
    (unsurprising, prior = the family's top bridges) -> narrow, and the prior holds."""
    store = tmp_path / "store"
    assert main(["run", "--backend", "mock", "--max-calls", "40", "--max-problems", "3",
                 "--store", str(store), "--seeds", "oddtown,eventown,fisher_one"]) == 0
    done = [r for r in _rows(store / "problems.jsonl") if r["event"] == "done"]
    assert len(done) == 3
    p1, p2, p3 = done
    for p in done:
        for key in ("breadth", "mode", "surprising", "surprise_max", "prior_bridges", "escalated"):
            assert key in p
    assert p1["breadth"] == "adaptive"
    assert (p1["surprising"], p1["mode"], p1["escalated"]) == (False, "narrow", True)
    assert (p2["surprising"], p2["mode"], p2["escalated"]) == (True, "wide", False)
    assert p2["surprise_max"] >= 1
    weaves = _rows(store / "weaves.jsonl")
    w1 = [w for w in weaves if w["problem"] == p1["id"]]
    assert [w["weave_call"] for w in w1] == [1, 2, 2, 2, 2, 2]
    assert all("in_prior" in w for w in weaves)
    assert sum(1 for w in weaves if w["problem"] == p2["id"]) == 5
    # problem 3 (a seed, no parent) takes the family's top bridges as its prior,
    # and the narrow weave wins with one of them
    assert p3["mode"] == "narrow" and p3["prior_bridges"] and p3["prior_source"] == "family"
    w3 = [w for w in weaves if w["problem"] == p3["id"]]
    assert len(w3) == 1 and w3[0]["in_prior"] and w3[0]["weave_call"] == 1 and w3[0]["win"]
    assert not p3["escalated"]
    calls = _rows(store / "calls.jsonl")
    # narrow 1-call weaves: p1 2 weave calls (escalated), p2 1, p3 1
    assert sum(1 for c in calls if c["step"] == "weave") == 4
    capsys.readouterr()
    assert main(["report", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "prior held 1/2" in out


def test_prior_bridges_order(tmp_path):
    from mathexp.loop import prior_bridges
    from mathexp.store import Store

    st = Store(tmp_path / "s", "mock")
    assert prior_bridges(st, "setfam", None, "m") == ([], "none")

    def w(problem, bridge, family="setfam", win=True, by="m"):
        st.append("weaves", {"problem": problem, "bridge_id": bridge, "family": family, "win": win, "by": by})

    w("p1", "a")
    w("p2", "b")
    w("p3", "b")
    w("q1", "b", family="addfree")
    w("p4", "c", win=False)
    w("p5", "d", by="other-model")
    # family top 2 by importance (distinct families, then problems), same model only
    assert prior_bridges(st, "setfam", None, "m") == (["b", "a"], "family")
    # the parent's wins come first
    assert prior_bridges(st, "setfam", "p1", "m") == (["a"], "parent")
    # a parent without wins falls back to the family
    assert prior_bridges(st, "setfam", "p4", "m") == (["b", "a"], "family")
