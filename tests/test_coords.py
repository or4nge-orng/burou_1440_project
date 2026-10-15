import pytest

import math
import numpy as np

from burou_1440_project.coords import *
from burou_1440_project import Vector

def _xyz(r, phi, theta):
    v = spherical_to_cartesian(r, phi, theta)
    return np.array([v.x, v.y, v.z])

def test_known_points():
    assert spherical_to_cartesian(2, 0, 0) == Vector(2, 0, 0)
    p = spherical_to_cartesian(2, math.pi / 2, 0)
    assert (p - Vector(0, 2, 0)).norm() == pytest.approx(0)
    p = spherical_to_cartesian(2, math.pi / 2, math.pi / 2)
    assert (p - Vector(0, 0, 2)).norm() == pytest.approx(0)

@pytest.mark.parametrize("r, phi, theta", [(1, 4, 1), (4, math.pi / 3, 70), (9, 60, 55)])
def test_norm_matches(r, phi, theta):
    assert spherical_to_cartesian(r, phi, theta).norm() == pytest.approx(r)

def test_jacobian_known_point():
    jac = spherical_jacobian(2, math.pi / 2, 0)
    expected = np.array([
        [0, -2, 0],
        [1, 0, 0],
        [0, 0, 2]
    ])
    assert np.allclose(jac, expected, atol=1e-12)

def test_jacobian_det():
    jac = spherical_jacobian(2, math.pi / 2, 0)
    assert np.linalg.det(jac) == pytest.approx(4 * math.cos(0))

def test_jac_small_displacement():
    p = (2.0, 0.5, 0.7)
    d = np.array([1e-6, -2e-6, 1.5e-6])
    J = spherical_jacobian(*p)
    
    
    assert np.allclose(_xyz(*(np.array(p) + d)) - _xyz(*p), J @ d, atol = 1e-10)

def test_num_jac_matches_analytic():
    point = (0.2, 0.4, 0.6)
    J_an = spherical_jacobian(*point)
    assert np.allclose(numerical_jacobian(_xyz, point), J_an, atol=1e-8)
    assert np.allclose(numerical_jacobian(_xyz, point, scheme="forward"), J_an, atol=1e-8)

def test_unknown_scheme():
    with pytest.raises(ValueError): 
        numerical_jacobian(_xyz, None, scheme="not a valid option")

def test_jacobian_matches_linear_function():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    f = lambda x, y, z: A @ np.array([x, y, z])
    assert np.allclose(numerical_jacobian(f, (1, -2, 0.5), h=1e-3), A, atol=1e-9)
