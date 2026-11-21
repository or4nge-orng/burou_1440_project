from mini_ballistic.kepler import *
from mini_ballistic.coords import numerical_jacobian
import math
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


OUT = Path(__file__).parent / "out"
es = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
STEPS = 2000
Ms=np.linspace(0, 2*math.pi, STEPS)

def deriv_anal(e):
    return [dE_dM(M, e) for M in Ms]
    

def deriv_num(Ms_new, e):
    return [numerical_jacobian(lambda m: np.array([solve_kepler(m, e)]), (M,))[0, 0] for M in Ms_new]

def main():
    OUT.mkdir(exist_ok=True)
    Ms_new = np.linspace(0, 2*math.pi, STEPS // 100)
    for e in es:
        plt.semilogy(Ms, deriv_anal(e), label=f"e={e}")
        plt.semilogy(Ms_new, deriv_num(Ms_new, e), "o-", markersize=4)

    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel("M")
    plt.ylabel(r"$\frac{dE}{dM}$")
    plt.tight_layout()
    plt.savefig(OUT / "dE_dM.png", dpi=150)
    plt.show()

if  __name__ == '__main__':
    main()