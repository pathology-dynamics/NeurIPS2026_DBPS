"""Evaluate a baseline's generated endpoints with the paper protocol.

Reads <workdir>/generated_endpoints.npy ([n_eval, repeat, d]) and, for each
repeat r (generator seeded with 42 + 7919 r), draws 100 generated endpoints
and 100 targets from the held-out reference Y_val + Y_test, and computes SWD
(256 projections) and FD. MMD is computed once between all generated
endpoints and the full reference pool. Writes <workdir>/baseline_metrics.csv
and the per-repeat SWD to <workdir>/swd_per_repeat.npy (for scripts.run_paired_tests).

usage:
  python -m scripts.eval_baseline --workdir outputs/adni/cfm_ot_normalization --setting adni
  # --config / --data override the workdir's config_snapshot.py and data file
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    frechet_distance,
    maximum_mean_discrepancy,
    sliced_wasserstein_distance,
)
from utils.cli import SETTINGS
from utils.config_utils import apply_workdir_overrides, load_config, resolve_data_path
from utils.seed import make_rng

FIELDS = ["swd_mean", "swd_std", "fd_mean", "fd_std", "mmd", "n_eval", "n_repeat"]


def evaluate_endpoints(ge: np.ndarray, Y_eval: np.ndarray, seed: int = 42,
                       eval_batch_size: int = 100) -> dict:
    n_eval, repeat, d = ge.shape
    n_Y = Y_eval.shape[0]
    eval_batch = min(int(eval_batch_size), n_eval, n_Y)
    swds, fds = [], []
    for r in range(repeat):
        rng_r = make_rng(seed + r * 7919)
        idx_x = rng_r.choice(n_eval, eval_batch, replace=False).astype(np.int64)
        idx_y = rng_r.choice(n_Y, eval_batch, replace=False).astype(np.int64)
        xh_b = ge[idx_x, r, :]
        Y_b = Y_eval[idx_y]
        swds.append(float(sliced_wasserstein_distance(xh_b, Y_b, num_projections=256, rng=rng_r)))
        fds.append(float(frechet_distance(xh_b, Y_b, cov_eps=1e-6)))
    mmd = float(maximum_mean_discrepancy(ge.reshape(-1, d), Y_eval, rng=make_rng(seed + 9001)))

    def _stat(xs):
        a = np.asarray(xs, dtype=np.float64)
        return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)

    swd_m, swd_s = _stat(swds)
    fd_m, fd_s = _stat(fds)
    return {"swd_mean": swd_m, "swd_std": swd_s, "fd_mean": fd_m, "fd_std": fd_s,
            "mmd": mmd, "n_eval": eval_batch, "n_repeat": repeat, "swd_per_repeat": swds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--config", default=None, help="default: <workdir>/config_snapshot.py")
    ap.add_argument("--setting", choices=sorted(SETTINGS), default=None,
                    help="evaluation setting (fixes the reference pool)")
    ap.add_argument("--data", default=None, help="override processed_data.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    config = load_config(args.config or workdir / "config_snapshot.py")
    apply_workdir_overrides(config, workdir)
    if args.setting is not None:
        config.processed_data_path = SETTINGS[args.setting]["data"]
    if args.data is not None:
        config.processed_data_path = Path(args.data)
    z = np.load(resolve_data_path(config))
    Y_eval = np.concatenate([z["Y_val"], z["Y_test"]], axis=0).astype(np.float64)

    ge = np.load(workdir / "generated_endpoints.npy")
    if ge.ndim != 3:
        raise ValueError(f"expected generated_endpoints of shape (n_eval, repeat, d); got {ge.shape}")
    row = evaluate_endpoints(ge, Y_eval, seed=args.seed,
                             eval_batch_size=int(getattr(config, "eval_batch_size", 100) or 100))

    out = workdir / "baseline_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)
    np.save(workdir / "swd_per_repeat.npy", np.asarray(row["swd_per_repeat"]))
    print(f"[eval_baseline] {workdir.name}: SWD {row['swd_mean']:.4f} +/- {row['swd_std']:.4f}  "
          f"FD {row['fd_mean']:.2f}  MMD {row['mmd']:.3e} -> {out}")


if __name__ == "__main__":
    main()
