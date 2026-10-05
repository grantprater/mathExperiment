import math
import random

import pytest

from mathexp import calibration as C
from mathexp.loop import SURPRISE_TRIGGER_LOG2, is_surprising, surprise_stats


def _row(kind, conf, outcome, **kw):
    return {"domain": "setfam", "kind": kind, "level": "value", "by": "m", "confidence": conf,
            "outcome": outcome, **kw}


def test_summary_math():
    rows = [_row("interval", 0.8, "right", log2_err=0.5), _row("interval", 0.8, "wrong", log2_err=-1.0),
            _row("interval", 0.8, "right", log2_err=2.0), _row("interval", 0.8, "couldnt-tell")]
    m = C.summarize(rows)
    assert m["n"] == 4 and m["n_scored"] == 3
    assert m["acc"] == pytest.approx(2 / 3)
    assert m["mean_conf"] == pytest.approx(0.8)
    assert m["gap"] == pytest.approx(2 / 3 - 0.8)
    assert m["brier"] == pytest.approx((0.04 + 0.64 + 0.04) / 3)
    assert m["coverage80"] == pytest.approx(2 / 3)
    assert m["median_log2_err"] == pytest.approx(0.5)


def test_table_groups_by_domain_kind_level_model():
    rows = [_row("interval", 0.8, "right"), {**_row("interval", 0.8, "wrong"), "by": "other"},
            {**_row("bound_holds", 0.6, "right"), "level": "mechanism"}]
    t = C.table(rows)
    assert len(t) == 3
    assert t[("setfam", "bound_holds", "mechanism", "m")]["brier"] == pytest.approx(0.16)


def test_runs_and_change_point_ported():
    r = C.runs([1, 0] * 10)
    assert r[2] > C.RUNS_Z  # alternating
    r2 = C.runs([1] * 10 + [0] * 10)
    assert r2[2] < -C.RUNS_Z  # clustered
    c = C.change_point([1] * 15 + [0] * 15)
    assert c and c["changed"] and c["k"] == 15
    assert C.change_point([1] * 4 + [0] * 4) is None  # underpowered
    # Bonferroni: the corrected p is the raw p times the number of splits
    assert c["p"] == pytest.approx(min(1.0, c["p_raw"] * c["n_splits"]))


def test_change_point_quiet_on_coin_flips():
    rng = random.Random(3)
    fired = 0
    for _ in range(100):
        cp = C.change_point([rng.randint(0, 1) for _ in range(40)])
        fired += bool(cp and cp["changed"])
    assert fired <= 10


def test_surprise_trigger():
    assert SURPRISE_TRIGGER_LOG2 == 1.0
    # point 7 vs f 3: log2(8/4) = 1 -> surprising (the threshold is inclusive)
    mean, mx = surprise_stats({3: {"point": 7}, 4: {"point": 4}}, {3: 3, 4: 4})
    assert mx == pytest.approx(1.0) and mean == pytest.approx(0.5)
    assert is_surprising(mx)
    # only n with known f count
    mean, mx = surprise_stats({3: {"point": 3}, 4: {"point": 100}}, {3: 3, 4: None})
    assert mx == 0 and not is_surprising(mx)
    assert surprise_stats({3: {"point": 3}}, {3: None}) == (None, None)
    assert not is_surprising(None)
    _, mx = surprise_stats({5: {"point": 10}}, {5: 6})  # log2(11/7) < 1
    assert mx == pytest.approx(math.log2(11 / 7)) and not is_surprising(mx)
    _, mx = surprise_stats({5: {"point": 1}}, {5: 6})  # underestimates count too
    assert is_surprising(mx)


def test_feedback_text():
    rows = [_row("interval", 0.8, "right", log2_err=1.0), _row("interval", 0.8, "wrong", log2_err=2.0)]
    text = C.feedback_text(rows, "setfam", "m")
    assert "coverage 50%" in text and "run high" in text
    assert "No scored" in C.feedback_text(rows, "addfree", "m")
