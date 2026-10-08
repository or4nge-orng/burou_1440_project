from dataclasses import dataclass
from .vector import Vector
from math import cos, sin, hypot
import numpy as np
@dataclass(frozen=True)
class Rotation:
    w: float
    x: float
    y: float
    z: float

    @classmethod
    def from_axis_angle(cls, axis: Vector, angle: float):
        n = axis.normalized()
        w = cos(angle/2)
        s = sin(angle/2)
        return cls(w, s * n.x, s * n.y, s * n.z)

    def apply(self, v: Vector):
        u = Vector(self.x, self.y, self.z)
        t = u.cross(v)
        v_new = v + 2 * self.w * t + 2 * u.cross(t)
        return v_new

    def inverse(self):
        return type(self)(self.w, -self.x, -self.y, -self.z)

    def normalized(self):
        n = hypot(self.w, self.x, self.y, self.z)
        if n == 0:
            raise ValueError("Cannot normalize zero quaternion")
        return type(self)(self.w / n, self.x / n, self.y / n, self.z / n)

    def __matmul__(self, other):
        if not isinstance(other, Rotation):
            return NotImplemented

        w1 = self.w
        w2 = other.w
        u1 = Vector(self.x, self.y, self.z)
        u2 = Vector(other.x, other.y, other.z)
        w = w1 * w2 - u1.dot(u2)
        v = w1 * u2 + w2 * u1 + u1.cross(u2)
        return type(self)(w, v.x, v.y, v.z).normalized()

    def to_matrix(self):
        K = np.array([[0, -self.z, self.y],
                      [self.z, 0, -self.x],
                      [-self.y, self.x, 0]])
        return np.eye(3) + 2 * self.w * K + 2 * K @ K
    