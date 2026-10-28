"""Unit-aware calculator. The agent does all arithmetic here, never in the model.

Expressions are parsed with `ast` and evaluated against a whitelist, so no arbitrary
Python runs. Quantities carry units through `pint`, so a dimension error (adding volts
to amps, say) is caught instead of producing a plausible-looking wrong number.
"""

import ast
import math
import operator
import re

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 64
PREFERRED_UNITS = ["Hz", "s", "V", "A", "W", "ohm", "F", "H", "J", "m", "K", "C", "V/s", "A/V"]


class CalcError(ValueError):
    pass


def _dimensionless(x) -> float:
    if isinstance(x, Q_):
        try:
            return float(x.to("dimensionless").magnitude)
        except pint.DimensionalityError as exc:
            raise CalcError(f"Expected a dimensionless value, got {x.units:~P}") from exc
    return float(x)


def _parallel(*values):
    if not values:
        raise CalcError("parallel() needs at least one value")
    return 1 / sum(1 / v for v in values)


FUNCTIONS = {
    "sqrt": lambda x: x**0.5,
    "abs": abs,
    "min": min,
    "max": max,
    "log10": lambda x: math.log10(_dimensionless(x)),
    "ln": lambda x: math.log(_dimensionless(x)),
    "exp": lambda x: math.exp(_dimensionless(x)),
    "sin": lambda x: math.sin(_dimensionless(x)),
    "cos": lambda x: math.cos(_dimensionless(x)),
    "tan": lambda x: math.tan(_dimensionless(x)),
    "atan": lambda x: math.atan(_dimensionless(x)),
    "db": lambda ratio: 10 * math.log10(_dimensionless(ratio)),  # power ratio -> dB
    "db20": lambda ratio: 20 * math.log10(_dimensionless(ratio)),  # amplitude ratio -> dB
    "from_db": lambda d: 10 ** (_dimensionless(d) / 10),
    "from_db20": lambda d: 10 ** (_dimensionless(d) / 20),
    "parallel": _parallel,
}

CONSTANTS = {
    "pi": math.pi,
    "k_B": Q_(1, "boltzmann_constant"),
    "q_e": Q_(1, "elementary_charge"),
    "c0": Q_(1, "speed_of_light"),
    "h_planck": Q_(1, "planck_constant"),
}

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


# "0.7 MHz" or "12pF" written inline -> "(0.7*MHz)", so models can write units naturally.
# The atomic group stops backtracking into the exponent, so "1e3" is not read as 1 * e3.
_INLINE_UNIT = re.compile(r"(?<![\w.])((?>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))\s*([A-Za-zµΩ]\w*)")


def _inline_units(expression: str) -> str:
    return _INLINE_UNIT.sub(r"(\1*\2)", expression)


def _unit_name(name: str):
    """Resolve a bare name as a unit (e.g. MHz, kohm). Single letters are refused: 'C' or 'R'
    is far more likely an undefined variable than coulomb or a gas constant."""
    if len(name) < 2:
        return None
    try:
        return Q_(1, name)
    except Exception:  # pint raises UndefinedUnitError and others for non-units
        return None


def parse_quantity(text: str):
    try:
        return Q_(str(text).strip())
    except Exception as exc:  # pint raises several unrelated error types on bad input
        raise CalcError(f"Could not read {text!r} as a number with units") from exc


def _evaluate(node, names: dict):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise CalcError("Booleans are not allowed")
        if isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node.value, str):
            return parse_quantity(node.value)
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        unit = _unit_name(node.id)
        if unit is not None:
            return unit
        raise CalcError(f"Unknown name '{node.id}'. Pass it in variables, e.g. {{\"{node.id}\": \"10 kohm\"}}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = _evaluate(node.left, names)
        right = _evaluate(node.right, names)
        if isinstance(node.op, ast.Pow):
            right = _dimensionless(right)
            if abs(right) > MAX_EXPONENT:
                raise CalcError(f"Exponent {right} is too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except pint.DimensionalityError as exc:
            raise CalcError(f"Unit mismatch: {exc}") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = FUNCTIONS.get(node.func.id)
        if func is None:
            raise CalcError(f"Unknown function '{node.func.id}'. Available: {', '.join(FUNCTIONS)}")
        args = [_evaluate(a, names) for a in node.args]
        try:
            return func(*args)
        except pint.DimensionalityError as exc:
            raise CalcError(f"Unit mismatch in {node.func.id}(): {exc}") from exc
    raise CalcError(f"Unsupported syntax: {type(node).__name__}")


def _format(value, result_unit: str | None) -> dict:
    if not isinstance(value, Q_):
        value = float(value)
        return {"value": value, "unit": "", "display": f"{value:.4g}"}
    if result_unit:
        try:
            value = value.to(result_unit)
        except pint.DimensionalityError as exc:
            raise CalcError(f"Result has units {value.units:~P}, which cannot be shown in {result_unit}") from exc
        except Exception as exc:
            raise CalcError(f"Unknown result unit {result_unit!r}") from exc
    else:
        value = value.to_reduced_units()
        if value.dimensionless:
            value = value.to("dimensionless")
        else:
            for unit in PREFERRED_UNITS:
                if value.dimensionality == ureg.Unit(unit).dimensionality:
                    value = value.to(unit).to_compact()
                    break
    magnitude = float(value.magnitude)
    unit = f"{value.units:~P}"
    return {"value": magnitude, "unit": unit, "display": f"{magnitude:.4g} {unit}".strip()}


def calculate(expression: str, variables: dict[str, str] | None = None, result_unit: str | None = None) -> dict:
    """Evaluate `expression` with unit-carrying `variables` such as {"R": "10 kohm"}."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalcError("Expression is too long")
    names = dict(CONSTANTS)
    inputs = {}
    for name, text in (variables or {}).items():
        if not name.isidentifier():
            raise CalcError(f"Variable name {name!r} is not a valid identifier")
        names[name] = parse_quantity(text)
        inputs[name] = str(text)
    try:
        tree = ast.parse(_inline_units(expression.strip()), mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"Could not parse expression: {exc.msg}") from exc
    try:
        value = _evaluate(tree, names)
    except ZeroDivisionError as exc:
        raise CalcError("Division by zero") from exc
    except OverflowError as exc:
        raise CalcError("Result overflowed") from exc
    return {"expression": expression, "inputs": inputs, **_format(value, result_unit)}
