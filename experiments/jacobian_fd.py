from burou_1440_project.coords import *
import math

from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out"
FLOOR = 1e-18
point = (0.2, 0.4, 0.6)
hs = np.logspace(-14, -1, 60)

def _xyz(r, phi, theta):
    v = spherical_to_cartesian(r, phi, theta)
    return np.array([v.x, v.y, v.z])

def error(h, scheme):
    j_an = spherical_jacobian(*point)
    j_num = numerical_jacobian(_xyz, point, h, scheme)
    return np.max(np.abs(j_an - j_num))

def main():
    OUT.mkdir(exist_ok=True)
    err_forward = [error(h, "forward") for h in hs]
    err_central = [error(h, "central") for h in hs]

    plt.loglog(hs, err_forward, "o-", markersize=3, label="forward")
    plt.loglog(hs, err_central, "o-", markersize=3, label="central")
    plt.xlabel("h")
    plt.ylabel("error")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(OUT / "jacobian_fd.png")
    plt.show()

if __name__ == '__main__':
    main()