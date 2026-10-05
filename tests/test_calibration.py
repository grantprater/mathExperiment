import math
import random

import pytest

from mathexp import calibration as C
from mathexp.loop import MAX_BITS, SURPRISING_BITS, bits, is_surprising, surprise_stats
from mathexp.prompts import ParseError, normalize_distribution


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


def test_bits_rule():
    """PROTOCOL.md "Model surprise in bits"."""
    d = {"candidates": [[4, 0.5], [5, 0.25]], "other": 0.25}
    # a listed value scores its own mass
    assert bits(d, 4, 10) == (pytest.approx(1.0), 0.5)
    assert bits(d, 5, 10)[0] == pytest.approx(2.0)
    # an unlisted value shares the residual equally over [0, trivial] minus the listed
    # values: 11 values 0..10, 2 listed, so 9 share 0.25
    b, p = bits(d, 7, 10)
    assert p == pytest.approx(0.25 / 9) and b == pytest.approx(-math.log2(0.25 / 9))
    # listed values outside [0, trivial] do not shrink the share
    d2 = {"candidates": [[40, 0.5]], "other": 0.5}
    assert bits(d2, 3, 3)[1] == pytest.approx(0.5 / 4)
    # zero mass on the truth scores the 30-bit cap, as does a tiny mass
    assert bits({"candidates": [[1, 1.0]], "other": 0.0}, 2, 10) == (MAX_BITS, 0.0)
    assert bits({"candidates": [[1, 1 - 1e-12]], "other": 1e-12}, 2, 10)[0] == MAX_BITS
    # a certain, correct prediction scores 0 bits
    assert bits({"candidates": [[2, 1.0]], "other": 0.0}, 2, 10)[0] == 0.0


def test_distribution_normalisation():
    # masses summing to more than 1 are divided by the sum
    d = normalize_distribution([[3, 0.8], [4, 0.6]], 0.2)
    assert d["normalized"] and d["other"] == pytest.approx(0.2 / 1.6)
    assert d["candidates"] == [[3, pytest.approx(0.5)], [4, pytest.approx(0.375)]]
    # a shortfall goes to the residual; repeated values merge; sorted by mass
    d = normalize_distribution([[5, 0.2], [3, 0.3], [5, 0.1]], 0.1)
    assert not d["normalized"] and d["other"] == pytest.approx(0.4)
    assert {v: p for v, p in d["candidates"]} == {5: pytest.approx(0.3), 3: pytest.approx(0.3)}
    assert normalize_distribution([[1, 0.1], [2, 0.6]], 0.3)["candidates"][0][0] == 2
    with pytest.raises(ParseError):
        normalize_distribution([[1, 0.1]] * 1 + [[i, 0.1] for i in range(2, 8)], 0.0)  # 7 values
    with pytest.raises(ParseError):
        normalize_distribution([[1.5, 0.5]], 0.5)  # not an integer
    with pytest.raises(ParseError):
        normalize_distribution([[1, 1.5]], 0.0)  # mass out of range


def test_surprise_trigger_in_bits():
    assert SURPRISING_BITS == 4.0
    mean, mx = surprise_stats({3: 1.0, 4: 4.0})
    assert mean == pytest.approx(2.5) and mx == 4.0
    assert is_surprising(mx)  # inclusive: P(f) = 1/16
    assert not is_surprising(3.99)
    assert surprise_stats({}) == (None, None)
    assert not is_surprising(None)


def test_feedback_text():
    rows = [_row("distribution", 0.9, "right", bits=1.0), _row("distribution", 0.9, "wrong", bits=6.0),
            _row("top1", 0.8, "right"), _row("top1", 0.6, "wrong")]
    text = C.feedback_text(rows, "setfam", "m")
    assert "mean surprise 3.50 bits" in text and "candidates 50%" in text
    assert "- top1: 2 scored, stated mean confidence 0.70, actual accuracy 0.50" in text
    assert C.summarize(rows[:2])["mean_bits"] == pytest.approx(3.5)
    assert "No scored" in C.feedback_text(rows, "addfree", "m")
