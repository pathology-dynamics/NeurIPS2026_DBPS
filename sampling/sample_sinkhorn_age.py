"""Sinkhorn-OT-AGE baseline: barycentric Sinkhorn from X_eval to Y_train
using a 1-D age-distance cost matrix.

Identical to sampling/sample_sinkhorn.py except cost matrix replaces
sqeuclidean of 84-d ROI vectors with squared age difference:
    M[i, j] = (age_X[i] - age_Y[j])^2
This makes the OT plan transport mass through age-similar pairs instead
of ROI-similar pairs, using only the age signal (no learned controller).

Per-repeat sampled mode (matching sample_sinkhorn.py).
"""
from __future__ import annotations


import argparse

import numpy as np
import ot

from utils.config_utils import load_config, resolve_data_path
from utils.cli import add_setting_args, apply_setting_args
from utils.io_utils import ensure_dir, save_config_snapshot
from utils.seed import set_global_seed


def _impute(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    fill = float(np.median(finite)) if finite.size > 0 else 70.0
    return np.where(np.isnan(a), fill, a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--use_normalization", action="store_true")
    parser.add_argument("--no_normalization", action="store_true")
    add_setting_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_setting_args(config, args)
    if args.dataset_name is not None:
        config.dataset_name = args.dataset_name
    if args.use_normalization and args.no_normalization:
        raise ValueError("cannot pass both --use_normalization and --no_normalization")
    if args.use_normalization:
        config.use_normalization = True
        config.suffix = "_normalization"
    if args.no_normalization:
        config.use_normalization = False
        config.suffix = "_raw"

    set_global_seed(config.seed)

    data_path = resolve_data_path(config)
    npz = np.load(data_path)
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    eval_split = getattr(config, "eval_split", "merged")
    if eval_split == "merged":
        X_eval = np.concatenate([
            np.asarray(npz["X_val"], dtype=np.float64),
            np.asarray(npz["X_test"], dtype=np.float64),
        ], axis=0)
    else:
        X_eval = np.asarray(npz[f"X_{eval_split}"], dtype=np.float64)
    n_eval, d = X_eval.shape
    n_train = Y_train.shape[0]
    repeat = int(config.repeat)

    Y_train_ages = _impute(npz["Y_train_ages"])
    X_eval_ages = _impute(np.concatenate([
        np.asarray(npz["X_val_ages"], dtype=np.float64),
        np.asarray(npz["X_test_ages"], dtype=np.float64),
    ], axis=0))

    workdir = config.workdir_root / config.make_workdir_name(config)
    ensure_dir(workdir)

    sinkhorn_reg = float(getattr(config, "sinkhorn_reg", 0.05))
    # Age-only cost matrix: [n_eval, n_train]
    M = (X_eval_ages[:, None] - Y_train_ages[None, :]) ** 2
    if M.max() > 0:
        M = M / M.max()                                                # normalize for log-domain stability
    a = np.ones(n_eval, dtype=np.float64) / n_eval
    b = np.ones(n_train, dtype=np.float64) / n_train
    T = ot.sinkhorn(
        a, b, M, reg=sinkhorn_reg, numItermax=2000,
        method="sinkhorn_log",
    )
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    T_norm = T / row_sums

    rng_local = np.random.default_rng(int(config.seed))
    x_1 = np.empty((n_eval, repeat, d), dtype=np.float64)
    for i in range(n_eval):
        idx = rng_local.choice(n_train, size=repeat, p=T_norm[i])
        x_1[i] = Y_train[idx]
    np.save(workdir / "generated_endpoints.npy", x_1)
    np.save(
        workdir / "y_star_indices.npy",
        -np.ones((n_eval, repeat), dtype=np.int64),
    )

    save_config_snapshot(config, workdir)
    with open(workdir / "sample_summary.txt", "w") as f:
        f.write(f"method: {config.method}\n")
        f.write(f"dataset: {config.dataset_name} ({config.suffix})\n")
        f.write(f"eval_split: {eval_split}\n")
        f.write(f"n_eval: {n_eval}, repeat: {repeat}, d: {d}\n")
        f.write(f"sinkhorn_reg: {sinkhorn_reg}\n")
        f.write("cost: (age_X - age_Y)^2 (1-d age distance, no ROI signal)\n")
        f.write(f"workdir: {workdir}\n")
    print(f"Done. Wrote outputs to {workdir}")


if __name__ == "__main__":
    main()
