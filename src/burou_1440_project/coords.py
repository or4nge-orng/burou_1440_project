from burou_1440_project import Vector
import math

def spherical_to_cartesian(r, phi, theta) -> Vector:
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