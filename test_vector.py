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

@pytest.mark.parametrize("a, b", [
    (Vector(2, 3, 4), Vector(6, 3, 1)),
    (Vector(1, 0, 0), Vector(0, 1, 1)),
    (Vector(0.3, -1.2, 5), Vector(2, 0.7, -1))
])
def test_cross(a, b):
    c = a.cross(b)
    assert c.dot(a) == pytest.approx(0)
    assert c.dot(b) == pytest.approx(0)
    assert b.cross(a) == Vector(-c.x, -c.y, -c.z)

def test_dot_not_vectors():
    with pytest.raises(TypeError):
        Vector(2, 4, 5).dot(2)

def test_cross_not_vectors():
    with pytest.raises(TypeError):
        Vector(2, 4, 5).cross(2)

def test_norm():
    assert Vector(3, 4, 0).norm() == pytest.approx(5)
    assert Vector(1, 2, 2).norm() == pytest.approx(3)

def test_norm_zero_vector():
    assert Vector(0, 0, 0).norm() == pytest.approx(0)

@pytest.mark.parametrize("k", [
    2, -3, 0.5
])
def test_norm_scaling(k):
    v = Vector(1, -2, 3)
    assert (k * v).norm() == pytest.approx(abs(k) * v.norm())

def test_lagrange_identity():
    a, b = Vector(1, 2, 3), Vector(-2, 0.5, 4)
    lhs = a.cross(b).norm()**2 + a.dot(b)**2
    assert lhs == pytest.approx(a.norm() ** 2 * b.norm() ** 2)

def test_normalized():
    assert Vector(3, 4, 0).normalized() == Vector(0.6, 0.8, 0)

@pytest.mark.parametrize("v", [
    Vector(1, 2, 3),
    Vector(-5, 0.1, 7),
    Vector(1e-8, 2e-8, -3e-8),
    Vector(1e7, -2e7, 3e6)
])
def test_normilized_len(v):
    assert v.normalized().norm() == pytest.approx(1)

def test_normalize_zero_vector():
    with pytest.raises(ValueError):
        Vector(0, 0, 0).normalized()

def test_norm_keep_direction():
    v = Vector(1, 2, 3)
    n = v.normalized()
    assert v.cross(n).norm() == pytest.approx(0, abs=1E-12)