import pytest
from math import sin, pi, exp, log1p

from burou_1440_project import taylor_sin, taylor_exp, taylor_log1p

@pytest.mark.parametrize("x", [0.2, 3, -0.1, 6, 8, 7])
def test_sine_match_math(x):
    taylor_sine = taylor_sin(x, 50)
    assert taylor_sine == pytest.approx(sin(x), abs=5e-14)

@pytest.mark.parametrize("x", [0.2, 3, -0.1, 6, 8, 7])
def test_exp_match_math(x):
    taylor_expon = taylor_exp(x, 50)
    assert taylor_expon == pytest.approx(exp(x), abs=1e-9)

def test_taylor_sin_first_terms():
    x = 0.5
    assert taylor_sin(x, 0) == 0
    assert taylor_sin(x, 1) == pytest.approx(x)
    assert taylor_sin(x, 2) == pytest.approx(x - x**3 / 6)
    assert taylor_sin(x, 3) == pytest.approx(x - x**3 / 6 + x**5 / 120)

def test_taylor_exp_first_terms():
    x = 0.5
    assert taylor_exp(x, 0) == 0
    assert taylor_exp(x, 1) == pytest.approx(1)
    assert taylor_exp(x, 2) == pytest.approx(1 + x)
    assert taylor_exp(x, 3) == pytest.approx(1 + x + x**2 / 2)

def test_taylor_log1p_first_terms():
    x = 0.5
    assert taylor_log1p(x, 0) == 0
    assert taylor_log1p(x, 1) == pytest.approx(x)
    assert taylor_log1p(x, 2) == pytest.approx(x - x**2 / 2)
    assert taylor_log1p(x, 3) == pytest.approx(x - x**2 / 2 + x**3 / 3)

@pytest.mark.parametrize("func, ref, xs", [
    (taylor_sin, sin, [-2.0, -0.5, 0.3, 2.0]),
    (taylor_exp, exp, [-2.0, -0.5, 0.3, 2.0]),
    (taylor_log1p, log1p, [-0.5, -0.1, 0.3, 0.5]),
])
def test_series_matches_math_inside_radius(func, ref, xs):
    for x in xs:
        assert func(x, 200) == pytest.approx(ref(x), rel=1e-12)

@pytest.mark.parametrize("func", [taylor_sin, taylor_exp, taylor_log1p])
def test_negative_n_raises(func):
    with pytest.raises(ValueError):
        func(0.5, -1)

def test_taylor_sin_is_odd():
    assert taylor_sin(-0.7, 6) == pytest.approx(-taylor_sin(0.7, 6))

def test_taylor_log1p_diverges_outside_radius():
    assert abs(taylor_log1p(1.5, 60) - log1p(1.5)) > 1e3
