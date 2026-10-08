from dataclasses import dataclass
from numbers import Real
from math import hypot

@dataclass(frozen=True)
class Vector:
    
    x: float
    y: float
    z: float

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return type(self)(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return type(self)(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k):
        if not isinstance(k, (int, float)):
            return NotImplemented
        return type(self)(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def dot(self, other) -> float:
        if not isinstance(other, Vector):
            raise TypeError(f"Expected Vector, got '{type(Vector)}'")
        return self.x * other. x + self.y * other.y + self.z * other.z

    def cross(self, other):
        if not isinstance(other, Vector):
            raise TypeError(f"Expected Vector, got '{type(Vector)}'")
        return type(self)(self.y * other.z - self.z * other.y, -(self.x * other.z - self.z * other.x), self.x * other.y - self.y * other.x)

    def norm(self) -> float:
        return hypot(self.x, self.y, self.z)

    def normalized(self):
        vec_norm = self.norm()
        if vec_norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return type(self)(self.x / vec_norm, self.y / vec_norm, self.z / vec_norm)