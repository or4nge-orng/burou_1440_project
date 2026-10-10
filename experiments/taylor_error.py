from burou_1440_project import taylor_sin, taylor_exp, taylor_log1p
import math

from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out"
N_MAX = 60
FLOOR = 1e-18

def errors(taylor, ref, x, n_max=N_MAX) -> list[float]:
    return [abs(taylor(x, n) - ref(x)) for n in range(1, n_max + 1)]

def errors_abs(taylor, ref, x, n_max=N_MAX) -> list[float]:
    return [abs(taylor(x, n) - ref(x)) / abs(ref(x)) for n in range(1, n_max + 1)]

def plot_case(ax, title, taylor ,ref, xs):
    n_values = range(1, N_MAX + 1)
    for x in xs:
        errs = [max(e, FLOOR) for e in errors(taylor, ref, x)]
        ax.semilogy(n_values, errs, marker="o", markersize=3, label=f"x = {x}")

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("ошибка")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

def plot_case_abs(ax, title, taylor ,ref, xs):
    n_values = range(1, N_MAX + 1)
    for x in xs:
        errs = [max(e, FLOOR) for e in errors_abs(taylor, ref, x)]
        ax.semilogy(n_values, errs, marker="o", markersize=3, label=f"x = {x}")

    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("ошибка")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()



def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 4))
    plot_case(axes[0][0], "sin(x)", taylor_sin, math.sin, [0.5, 2.0, 20.0])
    plot_case(axes[0][1], "exp(x)", taylor_exp, math.exp, [0.5, 10.0, -10.0])
    plot_case(axes[0][2], "log(1+x)", taylor_log1p, math.log1p, [0.5, 1.0, 1.5])
    plot_case_abs(axes[1][0], "sin(x)", taylor_sin, math.sin, [0.5, 2.0, 20.0])
    plot_case_abs(axes[1][1], "exp(x)", taylor_exp, math.exp, [0.5, 10.0, -10.0])
    plot_case_abs(axes[1][2], "log(1+x)", taylor_log1p, math.log1p, [0.5, 1.0, 1.5])
    fig.tight_layout()
    fig.savefig(OUT / "taylor_error.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    main()
    print(taylor_log1p(0.5, 60), math.log1p(0.5))
    print(taylor_log1p(1.5, 3), math.log1p(1.5))