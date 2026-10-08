import pytest
from math import pi, sqrt, hypot

from vector import Vector
from rotation import Rotation

def test_from_axis_angle():
    r = Rotation.from_axis_angle(Vector(0, 0, 1), pi / 2)
    assert r.w == pytest.approx(sqrt(2) / 2)
    assert r.x == pytest.approx(0)
    assert r.y == pytest.approx(0)
    assert r.z == pytest.approx(sqrt(2) / 2)

def test_from_axis_angle_ignores_length():
    a, b = Rotation.from_axis_angle(Vector(0, 0, 1), 1.0), Rotation.from_axis_angle(Vector(0, 0, 5), 1.0)
    assert a.z == pytest.approx(b.z)

def test_from_axis_angle_zero_vector():
    with pytest.raises(ValueError):
        Rotation.from_axis_angle(Vector(0, 0, 0), 10)

@pytest.mark.parametrize("v", [
    Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1), Vector(4, 3, 4)
])
def test_from_axis_angle_unit_length(v):
    r = Rotation.from_axis_angle(v, 10)
    assert hypot(r.w, r.x, r.y, r.z) == pytest.approx(1)

def test_rotation_applying():
    r = Rotation.from_axis_angle(Vector(0, 0, 1), pi/2)
    res = r.apply(Vector(1, 0, 0))
    assert res.x == pytest.approx(0)
    assert res.y == pytest.approx(1)
    assert res.z == pytest.approx(0)

@pytest.mark.parametrize("phi", [
    0.1, pi/2, 70
])
def test_rotation_keeps_length(phi):
    r = Rotation.from_axis_angle(Vector(1, -2, 0.4), phi)
    v = Vector(4, 3, 5)
    assert r.apply(v).norm() == pytest.approx(v.norm())

def test_inverse():
    r = Rotation.from_axis_angle(Vector(0, 0, 1), pi/2)
    v = Vector(-6, 0, 1)
    assert r.inverse().apply(r.apply(v)) == v

def test_composition():
    r1 = Rotation.from_axis_angle(Vector(1, 2, 3), pi/2)
    r2 = Rotation.from_axis_angle(Vector(0, -1, 1), 0.5)
    v = Vector(1, 2, 4)
    res1 = (r2 @ r1).apply(v)
    res2 = r2.apply(r1.apply(v))
    assert res1.x == pytest.approx(res2.x)
    assert res1.y == pytest.approx(res2.y)
    assert res1.z == pytest.approx(res2.z)

def test_composition_not_com():
    r1 = Rotation.from_axis_angle(Vector(1, 2, 3), pi/2)
    r2 = Rotation.from_axis_angle(Vector(0, -1, 1), 0.5)
    v = Vector(1, 2, 4)
    a = (r2 @ r1).apply(v)
    b = (r1 @ r2).apply(v)
    assert(a - b).norm() != 0

def test_composition_rotation_same_axis():
    r1 = Rotation.from_axis_angle(Vector(0, 0, 3), 0.1)
    r2 = Rotation.from_axis_angle(Vector(0, 0, 1), 0.5)
    expected = Rotation.from_axis_angle(Vector(0, 0, 1), 0.6)
    v = Vector(1, 2, 4)
    assert ((r2 @ r1).apply(v) - expected.apply(v)).norm() == pytest.approx(0)

def test_composition_wrong_type():
    with pytest.raises(TypeError):
        Rotation.from_axis_angle(Vector(1, 2, 3), 0.2) @ 4

def test_composition_inverse():
    r = Rotation.from_axis_angle(Vector(0, 0, 3), 0.1)
    v = Vector(1, 2, 4)
    assert ((r.inverse() @ r).apply(v) - v).norm() == pytest.approx(0)

def test_composition_draif():
    r = Rotation.from_axis_angle(Vector(0, 0, 3), 0.01)
    total = r
    for _ in range(1_000_000):
        total = r @ total
    length = hypot(total.w, total.x, total.y, total.z)
    assert length == pytest.approx(1)