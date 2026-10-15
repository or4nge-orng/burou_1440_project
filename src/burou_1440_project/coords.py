from burou_1440_project import Vector
import math
import numpy as np
from typing import Callable

def spherical_to_cartesian(r: float, phi: float, theta: float) -> Vector:
    """
    Convert spherical coordinates to cartesian coordinates

    Parameters
    ----------
    - r : float
        radius (distance from origin)
    - phi : float
        longotude in the Oxy plane
    - theta : float
        Latitude above the Oxy plane

    Returns
    -------
    Vector
        Vector in Cartesian coordinate system
    """
    x = r * math.cos(theta) * math.cos(phi)
    y = r * math.cos(theta) * math.sin(phi)
    z = r * math.sin(theta)

    return Vector(x, y, z)

def spherical_jacobian(r: float, phi: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    jacobian = np.array([
        [ct * cp, -r * ct * sp, -r * st * cp],
        [ct * sp, r * ct * cp, -r * st * sp],
        [st, 0.0, r * ct]
    ])

    return jacobian

def numerical_jacobian(func: Callable[..., np.ndarray], point, h: float = 1e-6, scheme: str = "central") -> np.ndarray:
    if scheme not in ("central", "forward"): raise ValueError("Unkown scheme: {scheme!r}")
    p = np.asarray(point, dtype=float)
    res = []
    if scheme == "forward":
        f0 = np.asarray(func(*p), dtype=float)

    for j in range(len(p)):
        step = np.zeros_like(p)
        step[j] = h
        if scheme == "central":
            diff = np.asarray(func(*(p + step))) - np.asarray(func(*(p - step)))
            res.append(diff / (2 * h))
        else:
            diff = np.asarray(func(*(p + step))) - f0
            res.append(diff / h)

    return np.column_stack(res)
