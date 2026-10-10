import pytest
from math import sin, pi, exp

from burou_1440_project import taylor_sin, taylor_exp

@pytest.mark.parametrize("x", [0.2, 3, -0.1, 6, 8, 7])
def test_sine_match_math(x):
    taylor_sine = taylor_sin(x, 50)
    assert taylor_sine == pytest.approx(sin(x), abs=5e-14)

@pytest.mark.parametrize("x", [0.2, 3, -0.1, 6, 8, 7])
def test_exp_match_math(x):
    taylor_expon = taylor_exp(x, 50)
    assert taylor_expon == pytest.approx(exp(x), abs=1e-9)