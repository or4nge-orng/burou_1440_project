from burou_1440_project import Vector
import math
import numpy as np

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

