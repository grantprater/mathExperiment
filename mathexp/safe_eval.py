"""AST-allowlist evaluator for model-written bound expressions.

Model output is never passed to ``eval`` or ``exec``. The expression is parsed
with :func:`ast.parse` and walked node by node; any node type, name or function
not on the allowlist is an error. Exponents and intermediate results are capped.

The single textual normalisation is ``^`` -> ``**`` (models write powers that way).
"""

from __future__ import annotations

import ast
import math
import operator

MAX_EXPONENT = 200
MAX_ABS = 10 ** 40
MAX_COMB_ARG = 10_000
MAX_FACTORIAL_ARG = 300
MAX_LEN = 500


class SafeEvalError(ValueError):
    pass


def _check(x):
    if isinstance(x, bool):
        raise SafeEvalError("booleans are not numbers here")
    if isinstance(x, complex):
        raise SafeEvalError("complex result")
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        raise SafeEvalError("non-finite result")
    if abs(x) > MAX_ABS:
        raise SafeEvalError("result too large")
    return x


def _int_arg(x, name, cap):
    if isinstance(x, float):
        if not x.is_integer():
            raise SafeEvalError(f"{name} needs integer arguments")
        x = int(x)
    if x < 0 or x > cap:
        raise SafeEvalError(f"{name} argument out of range")
    return x


def _comb(a, b):
    a = _int_arg(a, "comb", MAX_COMB_ARG)
    if isinstance(b, float) and b.is_integer():
        b = int(b)
    if not isinstance(b, int):
        raise SafeEvalError("comb needs integer arguments")
    if b < 0 or b > a:
        return 0
    return math.comb(a, b)


def _binom_sum(n, s):
    """sum_{i=0}^{s} C(n, i)."""
    n = _int_arg(n, "binom_sum", MAX_COMB_ARG)
    if isinstance(s, float):
        s = math.floor(s)
    if s < 0:
        return 0
    return sum(math.comb(n, i) for i in range(0, min(int(s), n) + 1))


def _factorial(x):
    return math.factorial(_int_arg(x, "factorial", MAX_FACTORIAL_ARG))


def _log(x, base=None):
    if x <= 0 or (base is not None and (base <= 0 or base == 1)):
        raise SafeEvalError("log domain")
    return math.log(x) if base is None else math.log(x, base)


def _log2(x):
    if x <= 0:
        raise SafeEvalError("log domain")
    return math.log2(x)


def _sqrt(x):
    if x < 0:
        raise SafeEvalError("sqrt domain")
    return math.sqrt(x)


def _exp(x):
    if x > 90:
        raise SafeEvalError("exp too large")
    return math.exp(x)


FUNCS = {
    "comb": _comb,
    "factorial": _factorial,
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": _sqrt,
    "log": _log,
    "log2": _log2,
    "exp": _exp,
    "min": min,
    "max": max,
    "binom_sum": _binom_sum,
}


def _pow(a, b):
    if abs(b) > MAX_EXPONENT:
        raise SafeEvalError("exponent too large")
    if a == 0 and b < 0:
        raise SafeEvalError("division by zero")
    if isinstance(b, int) and abs(a) >= 2 and b * math.log2(abs(a)) > 140:
        raise SafeEvalError("result too large")
    return a ** b


def _div(a, b):
    if b == 0:
        raise SafeEvalError("division by zero")
    return a / b


def _floordiv(a, b):
    if b == 0:
        raise SafeEvalError("division by zero")
    return a // b


def _mod(a, b):
    if b == 0:
        raise SafeEvalError("modulo by zero")
    return a % b


BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _div,
    ast.FloorDiv: _floordiv,
    ast.Mod: _mod,
    ast.Pow: _pow,
}
UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def normalize(expr: str) -> str:
    return expr.replace("^", "**").strip()


def evaluate(expr: str, variables: dict[str, float]):
    """Evaluate ``expr`` with the given variables. Raises :class:`SafeEvalError`."""
    if not isinstance(expr, str):
        raise SafeEvalError("expression must be a string")
    if len(expr) > MAX_LEN:
        raise SafeEvalError("expression too long")
    try:
        tree = ast.parse(normalize(expr), mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"syntax error: {e.msg}") from None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise SafeEvalError(f"constant {node.value!r} not allowed")
            return _check(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return _check(variables[node.id])
            raise SafeEvalError(f"unknown name {node.id!r}")
        if isinstance(node, ast.BinOp):
            op = BINOPS.get(type(node.op))
            if op is None:
                raise SafeEvalError(f"operator {type(node.op).__name__} not allowed")
            return _check(op(ev(node.left), ev(node.right)))
        if isinstance(node, ast.UnaryOp):
            op = UNARY.get(type(node.op))
            if op is None:
                raise SafeEvalError(f"operator {type(node.op).__name__} not allowed")
            return _check(op(ev(node.operand)))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                raise SafeEvalError("only allowlisted functions may be called")
            if node.keywords:
                raise SafeEvalError("keyword arguments not allowed")
            args = [ev(a) for a in node.args]
            try:
                return _check(FUNCS[node.func.id](*args))
            except SafeEvalError:
                raise
            except (TypeError, ValueError, OverflowError) as e:
                raise SafeEvalError(f"{node.func.id}: {e}") from None
        raise SafeEvalError(f"{type(node).__name__} not allowed")

    try:
        return ev(tree)
    except SafeEvalError:
        raise
    except (OverflowError, ZeroDivisionError, RecursionError, TypeError, ValueError) as e:
        raise SafeEvalError(str(e)) from None
