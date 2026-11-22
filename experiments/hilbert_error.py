from mini_ballistic.linalg import *

from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out"

def main():
    OUT.mkdir(exist_ok=True)
    sizes = range(2,15)
    errors_x = []
    errors_x_np = []
    errors_b = []
    conds = []
    for size in sizes:
        H = hilbert(size)
        x_true = np.ones(size)
        b = H @ x_true
        x = gauss_solve(H, b, 0.0)
        errors_x.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
        errors_x_np.append(np.linalg.norm(np.linalg.solve(H,b) - x_true) / np.linalg.norm(x_true))
        errors_b.append(np.linalg.norm(H @ x - b))
        conds.append(np.linalg.cond(H) * np.finfo(float).eps)

    plt.figure(figsize=(8, 4))
    plt.semilogy(sizes, errors_x, marker="o", label="ошибка решения gauss_solve")
    plt.semilogy(sizes, errors_x_np, marker="o", label="ошибка решения np.linalg.solve")
    plt.semilogy(sizes, errors_b, marker="o", label="невязка")
    plt.semilogy(sizes, conds, marker="o", label="оценка cond*eps")
    plt.title("Ошибка решения системы с матрицей Гильберта")
    plt.xlabel("Размерность матрицы")
    plt.ylabel("Ошибка")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(OUT / "hilbert_error.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    main()