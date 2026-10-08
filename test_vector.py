import pytest

from vector import Vector

def test_add():
    assert Vector(1, 2, 3) + Vector(2, 4, 6) == Vector(3, 6, 9)

def test_add_type_error():
    with pytest.raises(TypeError):
        Vector(1, 2, 3) + 4

def test_sub():
    assert Vector(4, 5, 6) - Vector(1, 2, 3) == Vector(3, 3, 3)

def test_sub_type_error():
    with pytest.raises(TypeError):
        Vector(1, 2, 3) - 4

def test_mul_scalar():
    v = Vector(4, 6, 8)
    assert v * 0.5 == 0.5 * v == Vector(2., 3., 4.)

def test_mul_not_scalar_left():
    with pytest.raises(TypeError):
        Vector(2, 3, 4) * '2'

def test_mul_not_scalar_right():
    with pytest.raises(TypeError):
        '2' * Vector(2, 3, 4)

def test_dot():
    assert Vector(2, 3, 4).dot(Vector(5, 4, 3)) == 2 * 5 + 3 * 4 + 4 * 3

def test_cross():
    assert Vector(2, 3, 4).cross(Vector(6, 3, 1)) == Vector(-9, 22, -12)

def test_dot_not_vectors():
    with pytest.raises(TypeError):
        Vector(2, 4, 5).dot(2)

def test_cross_not_vectors():
    with pytest.raises(TypeError):
        Vector(2, 4, 5).cross(2)