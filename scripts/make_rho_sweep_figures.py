"""Figures 3-5: DBPS metrics versus rho in [0, 0.5], one figure per gamma.

Reads tables/dbps_sweep.csv written by scripts.make_paper_tables. Each
figure is a 3 x 2 grid: rows SWD / FD / MMD, columns K = 128 / K = 50,
one line per setting.

usage:
  python -m scripts.make_rho_sweep_figures --sweep tables/dbps_sweep.csv --out figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SETTINGS = (("adni", "ADNI-only", "tab:blue", "o"),
            ("union", "Union→Union", "tab:green", "s"),
            ("union_to_adni", "Union→ADNI", "tab:red", "^"))
METRICS = (("swd_mean", "SWD"), ("fd_mean", "FD"), ("mmd", "MMD"))
KS = (128, 50)


def plot_gamma(df: pd.DataFrame, gamma: int, out_dir: Path, rho_max: float = 0.5) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for i, (col, name) in enumerate(METRICS):
        for j, k in enumerate(KS):
            ax = axes[i, j]
            for key, label, color, marker in SETTINGS:
                d = df[(df.setting == key) & (df.gamma == gamma) & (df.K == k) & (df.rho <= rho_max + 1e-9)]
                d = d.sort_values("rho")
                ax.plot(d.rho, d[col], color=color, marker=marker, linewidth=2, label=label)
            ax.set_title(f"{name}, K={k}")
            ax.set_ylabel(name)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if i == len(METRICS) - 1:
                ax.set_xlabel(r"Controller mixing $\rho$")
                ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
                ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"DBPS, rho = 0-0.5, gamma = {gamma}", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"rho_sweep_gamma{gamma}.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="tables/dbps_sweep.csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sweep)
    for gamma in sorted(df.gamma.unique()):
        plot_gamma(df, int(gamma), out)
        print(f"wrote {out}/rho_sweep_gamma{int(gamma)}.pdf")


if __name__ == "__main__":
    main()
