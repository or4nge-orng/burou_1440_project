import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mini_ballistic.linalg import lstsq_normal, lstsq_svd

OUT = Path(__file__).parent / "out"
N_RUNS = 20
t = np.linspace(0, 1, 50)
ms = range(2, 15)
eps_mach = np.finfo(float).eps
rng = np.random.default_rng(13414)

def error_with_noise(A, m, sigma, solver):
    x_true = np.ones(m)
    errs = []
    for _ in range(N_RUNS):
        b = A @ x_true + sigma * rng.normal(size=A.shape[0])
        x = solver(A, b)
        errs.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))

    return np.median(errs)

def plot(axis, label, sigma):
    err_svd1 = []
    err_svd2 = []
    err_norm = []
    conds = []
    for m in ms:
        A = np.vander(t, N=m, increasing=True)
        err_svd1.append(error_with_noise(A, m, sigma, lambda A, b: lstsq_svd(A, b, rcond=1e-8)))
        err_svd2.append(error_with_noise(A, m, sigma, lambda A, b: lstsq_svd(A, b, rcond=1e-6)))
        err_norm.append(error_with_noise(A, m, sigma,  lambda A, b: lstsq_normal(A, b, eps=0.)))
        conds.append(np.linalg.cond(A))

    conds = np.array(conds)
    est_svd = conds * eps_mach
    est_norm = (conds ** 2) * eps_mach

    axis.set_title(label)

    axis.semilogy(ms, err_svd1, "o-", label="Ошибка SVD (rcond=1e-8)")
    axis.semilogy(ms, err_svd2, "o-", label="Ошибка SVD (rcond=1e-6)")
    axis.semilogy(ms, err_norm, "s-", label="Ошибка нормального уравнения")
    axis.semilogy(ms, est_svd, "--", label=r'Оценка $\mathrm{cond}(A) \cdot \varepsilon$')
    axis.semilogy(ms, est_norm, "o-", label=r'Оценка $\mathrm{cond}^2(A) \cdot \varepsilon$')

    axis.set_xlabel('Число столбцов m (степень полинома m - 1)')
    axis.set_xticks(ms)
    axis.grid(True, which='both', linestyle='--', alpha=0.5)
    axis.legend(loc="best")

def main():
    OUT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15,6), sharey=True)

    plot(axes[0], "С шумом", 1e-6)
    plot(axes[1], "Без шума", 0)
    
    fig.tight_layout()
    fig.savefig(OUT / "lstsq_cond.png", dpi=150)
    fig.show()

if __name__ == '__main__':
    main()