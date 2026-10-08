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