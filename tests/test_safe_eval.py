import pytest

from mathexp.safe_eval import SafeEvalError, evaluate


@pytest.mark.parametrize("expr,expected", [
    ("n", 7), ("2**floor(n/2)", 8), ("n^2 - 1", 48), ("comb(n, 2)", 21), ("binom_sum(n, 2)", 29),
    ("factorial(4) // 5", 4), ("7 % 3", 1), ("-n + 10", 3), ("min(n, 3) + max(1, 2)", 5),
    ("ceil(sqrt(n))", 3), ("log2(8)", 3.0), ("p**n", 128), ("floor(log(exp(2)))", 2),
])
def test_allowlist(expr, expected):
    assert evaluate(expr, {"n": 7, "p": 2}) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "n.__class__", "__import__('os')", "(lambda: 1)()", "foo + 1", "open('x')", "2**10000",
    "10**50", "[1, 2]", "n if n else 1", "'a'", "n < 3", "eval('1')", "comb(n, k=2)",
    "factorial(10000)", "exp(1000)", "1/0", "n +* 2", "getattr(n, 'x')", "{}.keys()",
    "import os",
])
def test_rejects(expr):
    with pytest.raises(SafeEvalError):
        evaluate(expr, {"n": 7, "p": 2})
