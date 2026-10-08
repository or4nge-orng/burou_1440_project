from dataclasses import dataclass
from vector import Vector
from math import cos, sin

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

    
