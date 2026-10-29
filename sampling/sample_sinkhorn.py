"""Sinkhorn-OT baseline: barycentric transport from X_eval to Y_train.

Solve regularized OT plan T between empirical X_eval and Y_train, then
transport each x_eval[i] barycentrically:
    x_1[i] = sum_j (T[i, j] / sum_j T[i, j]) * Y_train[j]
This is the entropy-regularized analogue of OT-Map for unpaired distribution
transfer. Each `repeat` copy is identical (deterministic).

Output schema matches sample_uncontrolled / sample_meanshift so the same
scripts.reeval_baseline pipeline produces baseline_metrics.csv.
"""
import argparse

import numpy as np
import ot

from utils.config_utils import load_config, resolve_data_path
from utils.cli import add_setting_args, apply_setting_args
from utils.io_utils import ensure_dir, save_config_snapshot
from utils.seed import make_rng, set_global_seed


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

    workdir = config.workdir_root / config.make_workdir_name(config)
    ensure_dir(workdir)

    sinkhorn_reg = float(getattr(config, "sinkhorn_reg", 0.05))
    M = ot.dist(X_eval, Y_train, metric="sqeuclidean")                    # [n_eval, n_train]
    # Normalize cost matrix so reg=0.05 isn't pathological at 84-d Euclidean
    # scale (squared distances ~10-100 -> exp(-M/reg) underflows for small
    # reg). Using log-domain Sinkhorn avoids the underflow regardless.
    M = M / M.max()
    a = np.ones(n_eval, dtype=np.float64) / n_eval
    b = np.ones(n_train, dtype=np.float64) / n_train
    T = ot.sinkhorn(
        a, b, M, reg=sinkhorn_reg, numItermax=2000,
        method="sinkhorn_log",
    )                                                                      # [n_eval, n_train]
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)                      # numerical safety
    T_norm = T / row_sums                                                  # row-stochastic
    # Per-repeat sample mode (vs barycentric mean) so SWD/W2/FD see per-
    # subject distributional spread rather than collapsing to barycenters.
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
        f.write(f"workdir: {workdir}\n")
    print(f"Done. Wrote outputs to {workdir}")


if __name__ == "__main__":
    main()
