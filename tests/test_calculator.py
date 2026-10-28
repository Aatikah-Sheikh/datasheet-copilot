import math

import pytest

from copilot.calculator import CalcError, calculate


def test_rc_corner_frequency():
    result = calculate("1/(2*pi*R*C)", {"R": "1 kohm", "C": "1 nF"}, "kHz")
    assert result["value"] == pytest.approx(1e3 / (2 * math.pi), rel=1e-9)
    assert result["unit"] == "kHz"


def test_auto_unit_picks_hertz():
    result = calculate("1/(2*pi*R*C)", {"R": "1 kohm", "C": "1 nF"})
    assert result["unit"] == "kHz"
    assert result["value"] == pytest.approx(159.15, rel=1e-3)


def test_rise_time_from_bandwidth():
    result = calculate("0.35/f", {"f": "150 MHz"}, "ns")
    assert result["value"] == pytest.approx(2.333, rel=1e-3)


def test_decibels_and_parallel():
    assert calculate("db20(10)")["value"] == pytest.approx(20)
    assert calculate("from_db(3)")["value"] == pytest.approx(1.995, rel=1e-3)
    assert calculate("parallel(R1, R2)", {"R1": "100 ohm", "R2": "100 ohm"}, "ohm")["value"] == pytest.approx(50)


def test_thermal_noise_voltage():
    result = calculate("sqrt(4*k_B*T*R*B)", {"T": "300 K", "R": "50 ohm", "B": "1 MHz"}, "uV")
    assert result["value"] == pytest.approx(0.9100, rel=1e-3)


def test_unit_mismatch_is_an_error():
    with pytest.raises(CalcError, match="Unit mismatch"):
        calculate("V + I", {"V": "1 V", "I": "1 A"})


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "(1).__class__", "[x for x in range(3)]", "open('x')", "2**10000"],
)
def test_rejects_unsafe_or_unsupported(expression):
    with pytest.raises(CalcError):
        calculate(expression)


@pytest.mark.parametrize("expression", ["0.7 MHz / 20", "0.7*MHz/20", "700kHz/20"])
def test_inline_units(expression):
    result = calculate(expression, result_unit="kHz")
    assert result["value"] == pytest.approx(35)


def test_inline_units_leave_variables_and_exponents_alone():
    assert calculate("R1*2", {"R1": "5 ohm"}, "ohm")["value"] == pytest.approx(10)
    assert calculate("1e3 / 2")["value"] == pytest.approx(500)


def test_unknown_name_explains_fix():
    with pytest.raises(CalcError, match="Pass it in variables"):
        calculate("R*C")
