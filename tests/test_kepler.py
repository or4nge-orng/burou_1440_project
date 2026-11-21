import pytest
import math
import numpy as np
from mini_ballistic.kepler import *
from mini_ballistic.coords import numerical_jacobian

@pytest.mark.parametrize("M, e", [
    (0.0, 0.0),
    (0.01, 0.1),
    (1.0, 0.5),
    (math.pi, 0.9),
    (5.0, 0.99)
])
def test_solve_kepler(M, e):
    E = solve_kepler(M, e)
    assert abs(E - e * math.sin(E) - M) < 1e-10 * max(1., abs(M)) * 1e-3

def test_kepler_circle():
    assert solve_kepler(1.0, 0.0) == pytest.approx(1.0)

def test_kepler_incorrect_eccentricity():
    with pytest.raises(ValueError):
        solve_kepler(1.0, -0.1)
    with pytest.raises(ValueError):
        solve_kepler(1.0, 1.0)
    with pytest.raises(ValueError):
        solve_kepler(1.0, 1.5)

def test_kepler_periodic():
    e1 = solve_kepler(0.5, 0.3)
    e2 = solve_kepler(0.5 + 2 * math.pi, 0.3)
    assert abs(e1 - e2) == pytest.approx(2 * math.pi, abs=1e-10)

def test_dE_dM_circular_orbit():
    assert dE_dM(1.0, 0.0) == pytest.approx(1.0)

@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_dE_dM_special_points(e):
    assert dE_dM(0.0, e) == pytest.approx(1.0 / (1 - e))
    assert dE_dM(math.pi, e) == pytest.approx(1.0 / (1 + e))

@pytest.mark.parametrize("M, e", [
    (0.0, 0.0),
    (0.01, 0.1),
    (1.0, 0.5),
    (math.pi, 0.9),
    (5.0, 0.99)
])
def test_dE_dM_matches_numerical_derivative(M, e):
    J_num = numerical_jacobian(lambda m: np.array([solve_kepler(m, e)]), (M,))[0, 0]
    assert dE_dM(M, e) == pytest.approx(J_num, abs=1e-7)

@pytest.mark.parametrize("e", [
    0.0,
    0.1,
    0.5,
    0.9,
    0.99
])
def test_dE_dM_boundary(e):
    for M in np.linspace(0.0, 2 * math.pi, 50):
        assert 1 / (1 + e) <= dE_dM(M, e) <= 1 / (1 - e)