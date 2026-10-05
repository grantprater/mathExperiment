"""Run 2: the scan, scan-seeded predict-only runs (arm A5) and the mutation refusal."""

import json

import pytest

from mathexp import scan
from mathexp.__main__ import main
from mathexp.llm import MockBackend
from mathexp.loop import Config, Loop
from mathexp.problems import get_family
from mathexp.store import Store


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_tiny_scan_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "INTCYC_NS", list(range(4, 12)))
    monkeypatch.setattr(scan, "SETFAM_NS", list(range(3, 8)))
    out = tmp_path / "scan"
    cache = scan.ScanTruthCache(tmp_path / "scan_truth.jsonl")
    summ = scan.run_scan(["setfam", "addfree"], {"setfam": 3, "addfree": 4}, 5.0, top=6, quota=4,
                         out_dir=out, log=lambda *_: None, golf_cap=6, cache=cache,
                         golf_cache_dir=tmp_path / "golf")
    a, b = summ["setfam"], summ["addfree"]
    assert a["enumerated"] == 16 and b["enumerated"] == 6  # L <= 3 set families; int/cyclic named eqs
    assert a["scored"] + a["refused_uncheckable"] + a["no_interaction_dropped"] == a["enumerated"]
    assert a["no_interaction_dropped"] > 0  # e.g. |A| = 1 alone: f(n) = n = trivial(n)
    assert b["refused_uncheckable"] == 0
    rows = _rows(out / "setfam.jsonl") + _rows(out / "addfree.jsonl")
    for r in rows:
        assert r["J"] == r["G_full"] - r["G_prefix"] >= 0
        assert r["score"] == pytest.approx(r["S"] / r["L"])
        assert r["S"] == pytest.approx(sum(b["cost"] for b in r["breaks"]))
        assert r["truth"] != r["trivial"]  # no-interaction specs are dropped
        assert len(r["truth"]) >= scan.MIN_VALUES
    # truth-sequence dedup keeps the shortest statement
    for r in rows:
        if not r["kept"]:
            keeper = next(k for k in rows if k["id"] == r["dup_of"])
            assert keeper["truth"] == r["truth"] and keeper["L"] <= r["L"]
    sel = _rows(out / "selected.jsonl")
    assert [s["rank"] for s in sel] == list(range(1, len(sel) + 1))
    assert all(s["S"] > 0 for s in sel)  # no zero-score fillers
    assert len(sel) <= 6 and all(sum(1 for s in sel if s["family"] == f) <= 4 for f in ("setfam", "addfree"))
    assert all(sel[i]["score"] >= sel[i + 1]["score"] for i in range(len(sel) - 1))
    # resumable: a second pass computes nothing new
    cache2 = scan.ScanTruthCache(tmp_path / "scan_truth.jsonl")
    scan.run_scan(["setfam"], 3, 5.0, top=6, quota=4, out_dir=out, log=lambda *_: None, golf_cap=6,
                  cache=cache2, golf_cache_dir=tmp_path / "golf")
    assert cache2.computed == 0


def _selected(tmp_path):
    A, B = get_family("setfam"), get_family("addfree")
    specs = [(A, {"p": 2, "size_mod": [1], "int_mod": [0]}, [3, 4, 5, 6]),
             (A, {"p": 2, "int_abs": [1]}, [3, 4, 5, 6]),
             (A, {"p": 3, "size_mod": [2], "int_mod": [0, 1]}, [3, 4, 5, 6]),
             (B, {"group": "int", "equation": "3ap"}, [4, 5, 6, 7, 8]),
             (B, {"group": "cyclic", "equation": "sumfree", "equation2": "3ap"}, [4, 5, 6, 7, 8])]
    path = tmp_path / "selected.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, (fam, spec, ns) in enumerate(specs, 1):
            fh.write(json.dumps({"rank": i, "family": fam.name, "id": fam.problem_id(spec),
                                 "score": 1.0 / i, "S": 2.0 / i, "J": 5 - i, "L": fam.statement_length(spec),
                                 "ns": ns, "truth": [], "spec": spec}) + "\n")
    return path


def test_predict_only_notes_arms_then_report(tmp_path, capsys):
    sel = _selected(tmp_path)
    s_on, s_off = tmp_path / "on", tmp_path / "off"
    for store, notes in ((s_on, "on"), (s_off, "off")):
        assert main(["run", "--backend", "mock", "--seeds-from", str(sel), "--predict-only",
                     "--notes", notes, "--store", str(store), "--seed", "7", "--max-calls", "50"]) == 0
    on = [r for r in _rows(s_on / "problems.jsonl") if r["event"] == "done"]
    off = [r for r in _rows(s_off / "problems.jsonl") if r["event"] == "done"]
    assert len(on) == len(off) == 5
    # one call per problem, nothing woven or mutated
    for store in (s_on, s_off):
        calls = _rows(store / "calls.jsonl")
        assert len(calls) == 5 and {c["step"] for c in calls} == {"predict"}
        assert not (store / "weaves.jsonl").exists() or not _rows(store / "weaves.jsonl")
    # the same seeded order in both arms, and it is a shuffle of the ranks
    assert [r["id"] for r in on] == [r["id"] for r in off]
    assert [r["position"] for r in on] == [1, 2, 3, 4, 5]
    assert all(r["provenance"] == "scan" and r["scan_rank"] and r["predict_only"] for r in on)
    # notes: the same family's earlier problems, most recent first
    calls_on = _rows(s_on / "calls.jsonl")
    seen: dict[str, list[str]] = {}
    for r, c in zip(on, calls_on):
        prev = seen.setdefault(r["family"], [])
        assert r["notes_shown"] == len(prev)
        if prev:
            assert c["prompt"].index(prev[-1]) < c["prompt"].index("Nothing has been computed")
            if len(prev) > 1:
                assert c["prompt"].index(prev[-1]) < c["prompt"].index(prev[-2])
        prev.append(r["statement"])
    assert all(r["notes_shown"] == 0 for r in off)
    assert all("Notes:" not in c["prompt"] for c in _rows(s_off / "calls.jsonl"))
    # a different seed gives a different order (5! orders; these two differ)
    s3 = tmp_path / "seed3"
    main(["run", "--backend", "mock", "--seeds-from", str(sel), "--predict-only", "--store", str(s3),
          "--seed", "3", "--max-calls", "50"])
    assert [r["id"] for r in _rows(s3 / "problems.jsonl") if r["event"] == "done"] != [r["id"] for r in on]
    # ledger: distribution and top1 rows, scored in bits
    preds = Store(s_on, "mock").joined_predictions()
    kinds = {p["kind"] for p in preds}
    assert {"distribution", "top1"} <= kinds and "interval" not in kinds
    assert all(p["bits"] is not None and p["p_true"] is not None for p in preds if p["kind"] == "distribution")
    # report over both arms (a merged store)
    merged = tmp_path / "merged"
    merged.mkdir()
    for name in ("problems", "predictions", "calls"):
        text = (s_on / f"{name}.jsonl").read_text(encoding="utf-8") + (s_off / f"{name}.jsonl").read_text(encoding="utf-8")
        (merged / f"{name}.jsonl").write_text(text, encoding="utf-8")
    capsys.readouterr()
    assert main(["report", "--store", str(merged)]) == 0
    out = capsys.readouterr().out
    assert "A5: notes" in out and "paired (notes - no notes) over 5 problems" in out
    assert "sign test p" in out and "tercile" in out
    assert "Spearman(scan score S/L, mean bits)" in out and "n = 5" in out
    assert "top1" in out and "distribution" in out


def test_uncheckable_mutations_refused(tmp_path):
    # cap_set lives on F_3^k with ns [1, 2, 3]: every same-regime child has 3 values
    store = Store(tmp_path / "s", "mock")
    loop = Loop(Config(backend=MockBackend(seed=2), store=store, seeds=["cap_set"], breadth="fixed",
                       max_calls=3, log=lambda *_: None))
    loop.run_loop()
    rows = _rows(tmp_path / "s" / "problems.jsonl")
    refused = [r for r in rows if r["event"] == "refused_uncheckable"]
    queued_children = [r for r in rows if r["event"] == "queued" and r["parent"]]
    assert refused and all(r["values_computable"] < 4 for r in refused)
    for r in queued_children:
        assert loop.checkable_values(get_family("addfree"), r["spec"]) >= 4
    # mutation-effect predictions are written only for queued children
    preds = [p for p in _rows(tmp_path / "s" / "predictions.jsonl") if p.get("kind") == "mutation_effect"]
    assert {p["problem"] for p in preds} == {r["id"] for r in queued_children}


def test_checkable_counts_until_first_timeout(tmp_path):
    loop = Loop(Config(backend=MockBackend(), store=Store(tmp_path / "s", "mock"), time_limit=0.001,
                       log=lambda *_: None))
    B = get_family("addfree")
    # Sidon in [1..34+] needs far more than the time limit: no value computes
    assert loop.checkable_values(B, {"group": "int", "equation": "sidon", "ns": [34, 35, 36, 37]}) == 0
    # fewer than 4 n values: refused without computing anything
    assert loop.checkable_values(B, {"group": "int", "equation": "3ap", "ns": [4, 5, 6]}) == 3
    loop.cfg.time_limit = 20.0
    assert loop.checkable_values(B, {"group": "int", "equation": "3ap", "ns": [4, 5, 6, 7, 8]}) == 4


def test_scan_seeded_weave_run_reports(tmp_path, capsys):
    """A4 again: scan problems as seeds with weaving; the report handles scan wins."""
    sel = _selected(tmp_path)
    store = tmp_path / "a4"
    assert main(["run", "--backend", "mock", "--seeds-from", str(sel), "--top", "2", "--breadth", "fixed",
                 "--max-calls", "6", "--store", str(store)]) == 0
    weaves = _rows(store / "weaves.jsonl")
    assert weaves and any(w["win"] and w["provenance"] == "scan" for w in weaves)
    capsys.readouterr()
    assert main(["report", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "Weave headline" in out and "scan     bounds=" in out
    assert out.index("Weave headline") < out.index("Weave status counts")
