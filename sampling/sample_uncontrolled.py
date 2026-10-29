"""Uncontrolled baseline: rollout of the reference SDE with u = 0.

The evaluation pool is the merged X_val + X_test. y_star indices are filled
with -1 (no anchor). Rollout settings come from configs/base.py.
"""
import argparse

import numpy as np
import torch

from dynamics.rollout import rollout
from dynamics.transition_computation import TransitionComputation
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
    X_eval = np.concatenate([
        np.asarray(npz["X_val"], dtype=np.float64),
        np.asarray(npz["X_test"], dtype=np.float64),
    ], axis=0)
    L = np.asarray(npz["L"], dtype=np.float64)
    n_eval, d = X_eval.shape

    transition = TransitionComputation(
        L=L, beta=config.beta, cov_regularization=config.cov_regularization,
    )

    workdir = config.workdir_root / config.make_workdir_name(config)
    ensure_dir(workdir)

    repeat = int(config.repeat)
    x_0_np = np.repeat(X_eval[:, None, :], repeat, axis=1).reshape(n_eval * repeat, d)
    x_0 = torch.tensor(x_0_np, dtype=torch.float64)

    def controller_fn(x_t, t, y_star, candidates):
        return torch.zeros_like(x_t)

    out = rollout(
        x_0=x_0, controller_fn=controller_fn,
        y_star=None, candidates=None,
        transition=transition, num_steps=int(config.num_rollout_steps),
        g=float(config.g), use_sde=bool(config.use_sde),
        save_trajectories=bool(config.save_trajectories),
        save_controls=bool(config.save_controls), rng=rng,
    )

    x_1 = out["x_1"].reshape(n_eval, repeat, d).cpu().numpy()
    np.save(workdir / "generated_endpoints.npy", x_1)
    if config.save_trajectories:
        traj = out["trajectories"].reshape(
            n_eval, repeat, config.num_rollout_steps + 1, d,
        ).cpu().numpy()
        np.save(workdir / "trajectories.npy", traj)
    if config.save_controls:
        ctrl = out["controls"].reshape(
            n_eval, repeat, config.num_rollout_steps, d,
        ).cpu().numpy()
        np.save(workdir / "controls.npy", ctrl)
    np.save(
        workdir / "y_star_indices.npy",
        -np.ones((n_eval, repeat), dtype=np.int64),
    )

    save_config_snapshot(config, workdir)
    with open(workdir / "sample_summary.txt", "w") as f:
        f.write(f"method: {config.method}\n")
        f.write(f"dataset: {config.dataset_name} ({config.suffix})\n")
        f.write("eval_split: merged (X_val + X_test)\n")
        f.write(f"n_eval: {n_eval}, repeat: {repeat}, d: {d}\n")
        f.write(
            f"num_rollout_steps: {config.num_rollout_steps}, "
            f"use_sde: {config.use_sde}\n"
        )
        f.write(f"beta: {config.beta}, g: {config.g}\n")
        f.write(f"workdir: {workdir}\n")

    print(f"Done. Wrote outputs to {workdir}")


if __name__ == "__main__":
    main()
