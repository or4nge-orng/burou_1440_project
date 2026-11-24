from mini_ballistic.linalg import *
import pytest
import numpy as np

rng = np.random.default_rng(124687)

@pytest.mark.parametrize("m, n", [(20, 4),(5,2), (100, 10), (18, 6)])
def test_lstsq_orthogonal_to_cols(m, n):
    A = rng.integers(low=0, high=50, size=(m, n))
    b = rng.integers(low=0, high=50, size=A.shape[0])
    x = lstsq_svd(A, b)
    r = A @ x - b
    
    assert A.T @ r == pytest.approx(np.zeros(A.shape[1]), abs=1e-10*np.linalg.norm(A)*np.linalg.norm(b))

@pytest.mark.parametrize("n", [2, 3, 1, 5, 7])
def test_lstsq_line(n):
    t = rng.choice(50, size=n, replace=False)
    coeffs = rng.integers(low=0, high=50, size=n)
    poly = np.polynomial.Polynomial(coeffs)
    y = poly(t)
    A = np.vander(t, N=n, increasing=True)
    x = lstsq_svd(A, y)
    assert x == pytest.approx(coeffs, rel=1e-5, abs=1e-6)

@pytest.mark.parametrize("m, n", [(20, 4),(5,2), (100, 10), (18, 6)])
def test_lstsq_numpy(m, n):
    A = rng.integers(low=0, high=50, size=(m, n))
    b = rng.integers(low=0, high=50, size=A.shape[0])
    x_1 = lstsq_svd(A, b)
    x_2 = np.linalg.lstsq(A, b)[0]
    assert x_1 == pytest.approx(x_2)