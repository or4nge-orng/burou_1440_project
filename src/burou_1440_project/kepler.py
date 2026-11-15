import math

def solve_kepler(M: float, e: float, tol: float = 1e-10, max_inter: int = 50) -> float:

    if e < 0 or e >= 1:
        raise ValueError("Eccentricity must be in the range [0, 1)")

    E_0 = M if e < 0.8 else math.pi

    for _ in range(max_inter):
        delta = (E_0 - e * math.sin(E_0) - M) / (1 - e * math.cos(E_0))
        E_0 -= delta
        if abs(delta) < tol * max(1., abs(e)):
            return E_0 + (M - M % (2 * math.pi))

    raise ValueError("Kepler's equation did not converge within the maximum number of iterations")
