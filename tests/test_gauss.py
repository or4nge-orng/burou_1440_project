from mini_ballistic.linalg import *
from mini_ballistic import Vector

import pytest
import numpy as np

def test_gauss():
    A = np.array([[2,1,1],[1,2,1],[1,1,2]])
    b = np.array([3,0,0])
    sol = gauss_solve(A,b)
    assert (Vector(*map(float,sol))-Vector(9/4,-3/4,-3/4)).norm() == pytest.approx(0)

@pytest.mark.parametrize("A,b", [(np.array([[2,1,1],[1,2,1],[1,1,2]]), np.array([3,0,0])),
                                (np.array([[1,2,3],[0,1,4],[5,6,0]]), np.array([14, 8, 32])),
                                (np.array([[3,2],[1,4]]), np.array([5,6]))])
def test_gauss_numpy(A,b):
    assert gauss_solve(A,b) == pytest.approx(np.linalg.solve(A,b))

def test_gauss_singular():
    A = np.array([[1,2,3],[2,4,6],[3,6,9]])
    b = np.array([1,2,3])
    with pytest.raises(ValueError):
        gauss_solve(A,b)

def test_gauss_non_square():
    A = np.array([[1,2,3],[4,5,6]])
    b = np.array([1,2])
    with pytest.raises(ValueError):
        gauss_solve(A,b)

def test_gauss_b_not_size_A():
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    b = np.array([[1,2]])
    with pytest.raises(ValueError):
        gauss_solve(A,b)
    