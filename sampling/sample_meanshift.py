"""Mean-shift baseline: x_1 = X_eval + (Y_train.mean - X_train.mean).

Trivial linear shift baseline for unpaired distribution transfer. No
training, no SDE rollout. Each x_eval[i] gets pushed by the global
population-mean offset between X_train (HC) and Y_train (ADMCI). All
`repeat` copies are identical (deterministic baseline); per-repeat
variance in the SWD/W2/FD metrics comes from per-repeat eval-batch
subsampling inside scripts.reeval_baseline.

Output schema matches sample_uncontrolled / sample_unidb_random so the
same scripts.reeval_baseline pipeline produces baseline_metrics.csv.
"""
import argparse

import numpy as np

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
    rng = make_rng(config.seed)

    data_path = resolve_data_path(config)
    npz = np.load(data_path)
    X_train = np.asarray(npz["X_train"], dtype=np.float64)
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
    repeat = int(config.repeat)

    workdir = config.workdir_root / config.make_workdir_name(config)
    ensure_dir(workdir)

    mean_shift = Y_train.mean(axis=0) - X_train.mean(axis=0)              # [d]
    x_1_det = X_eval + mean_shift[None, :]                                # [n_eval, d]
    # Broadcast across repeats. Deterministic: per-repeat metrics vary
    # only because reeval_baseline subsamples idx_x / idx_y per repeat.
    x_1 = np.broadcast_to(
        x_1_det[:, None, :], (n_eval, repeat, d),
    ).copy()
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
        f.write(
            f"mean_shift_norm: {float(np.linalg.norm(mean_shift)):.6f}\n"
        )
        f.write(f"workdir: {workdir}\n")
    print(f"Done. Wrote outputs to {workdir}")


if __name__ == "__main__":
    main()
