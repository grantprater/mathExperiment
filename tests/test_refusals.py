"""The three rules the protocol enforces in code."""

import json

import pytest

from mathexp import REPO
from mathexp.llm import MockBackend
from mathexp.loop import Config, Loop
from mathexp.store import Refusal, Store


def test_a_truth_needs_interval_predictions_on_disk(tmp_path):
    store = Store(tmp_path / "s", "mock")
    with pytest.raises(Refusal):
        store.require_interval_predictions("abc", 3)
    store.predict(run="r", by="m", domain="setfam", kind="interval", level="value", claim="x",
                  confidence=0.8, problem="abc", call=None, prompt_version="v", n=3, point=1, lo=0, hi=2)
    assert store.require_interval_predictions("abc", 3)["n"] == 3
    with pytest.raises(Refusal):
        store.require_interval_predictions("abc", 4)


def test_a_loop_refuses_when_intervals_were_not_written(tmp_path):
    class DroppingStore(Store):
        def predict(self, **kw):  # simulate a bug that loses the interval rows
            if kw["kind"] == "interval":
                return {"id": "lost"}
            return super().predict(**kw)

    store = DroppingStore(tmp_path / "s", "mock")
    loop = Loop(Config(backend=MockBackend(), store=store, seeds=["oddtown"], max_calls=3,
                       log=lambda *_: None))
    with pytest.raises(Refusal):
        loop.run_loop()
    assert not (tmp_path / "truth.json").exists()  # nothing was computed


def test_b_mock_may_not_write_the_real_store(tmp_path):
    with pytest.raises(Refusal):
        Store(REPO / "store", "mock")
    with pytest.raises(Refusal):
        Store(REPO / "store" / "sub", "mock")
    Store(tmp_path / "ok", "mock")  # anywhere else is fine


def test_c_two_field_mutation_rejected_and_logged(tmp_path):
    store = Store(tmp_path / "s", "mock")
    loop = Loop(Config(backend=MockBackend(), store=store, seeds=["eventown"], breadth="fixed",
                       max_calls=3, log=lambda *_: None))
    loop.run_loop()
    rows = [json.loads(x) for x in (tmp_path / "s" / "problems.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected = [r for r in rows if r["event"] == "rejected"]
    assert len(rejected) == 1
    assert any("exactly one" in e for e in rejected[0]["reasons"])
    queued = [r for r in rows if r["event"] == "queued" and r["parent"]]
    assert len(queued) == 2  # m = 3 proposals, one rejected
