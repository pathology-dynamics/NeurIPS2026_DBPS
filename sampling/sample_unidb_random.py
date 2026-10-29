"""UniDB-random baseline: analytic endpoint bridge with random anchors.

Same controller as sampling/sample_unidb_ot.py, but the anchor y* is drawn
uniformly from Y_train per (eval_sample, repeat). This is the endpoint-only
controller that DBPS reduces to at rho = 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from controls.db_control import compute_u_anchor
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
    parser.add_argument("--gamma", type=float, default=None)
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
    if args.gamma is not None:
        config.gamma = (
            int(args.gamma) if float(args.gamma).is_integer() else float(args.gamma)
        )

    set_global_seed(config.seed)
    rng = make_rng(config.seed)

    data_path = resolve_data_path(config)
    npz = np.load(data_path)
    eval_split = getattr(config, "eval_split", "merged")
    if eval_split == "merged":
        # Evaluation pool = X_val + X_test.
        X_eval = np.concatenate([
            np.asarray(npz["X_val"], dtype=np.float64),
            np.asarray(npz["X_test"], dtype=np.float64),
        ], axis=0)
    else:
        X_eval = np.asarray(npz[f"X_{eval_split}"], dtype=np.float64)
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    L = np.asarray(npz["L"], dtype=np.float64)
    n_eval, d = X_eval.shape
    n_train = Y_train.shape[0]

    transition = TransitionComputation(
        L=L, beta=config.beta, cov_regularization=config.cov_regularization,
    )

    workdir = config.workdir_root / config.make_workdir_name(config)
    ensure_dir(workdir)

    repeat = int(config.repeat)
    # Stochastic anchor: per (eval_sample, repeat) draw a single Y_train
    # index uniformly without replacement on a per-row basis (replace=False
    # within each (i, r) is trivial for K=1; we just use random_choice on
    # a fresh rng draw).
    y_star_idx = np.empty((n_eval, repeat), dtype=np.int64)
    for r in range(repeat):
        for i in range(n_eval):
            y_star_idx[i, r] = int(rng.choice(n_train))
    y_star_np = Y_train[y_star_idx]                                         # [n_eval, repeat, d]

    x_0 = np.repeat(X_eval[:, None, :], repeat, axis=1).reshape(n_eval * repeat, d)
    x_0_t = torch.tensor(x_0, dtype=torch.float64)
    y_star_t = torch.tensor(
        y_star_np.reshape(n_eval * repeat, d), dtype=torch.float64,
    )

    def controller_fn(x_t, t, y_star_arg, candidates_arg):
        t_scalar = float(t[0].item())
        return compute_u_anchor(
            x_t, t_scalar, y_star_arg,
            gamma=config.gamma, beta=config.beta, g=config.g,
            transition=transition,
        )

    out = rollout(
        x_0=x_0_t, controller_fn=controller_fn,
        y_star=y_star_t, candidates=None,
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
    np.save(workdir / "y_star_indices.npy", y_star_idx)

    save_config_snapshot(config, workdir)
    with open(workdir / "sample_summary.txt", "w") as f:
        f.write(f"method: {config.method}\n")
        f.write(f"dataset: {config.dataset_name} ({config.suffix})\n")
        f.write(f"eval_split: {eval_split}\n")
        f.write(f"n_eval: {n_eval}, repeat: {repeat}, d: {d}\n")
        f.write(
            f"num_rollout_steps: {config.num_rollout_steps}, "
            f"use_sde: {config.use_sde}\n"
        )
        f.write(f"beta: {config.beta}, g: {config.g}\n")
        f.write(f"gamma: {config.gamma}\n")
        f.write("anchor draw: stochastic uniform from Y_train\n")
        f.write(
            "report_anchor_adherence: "
            f"{getattr(config, 'report_anchor_adherence', True)}\n"
        )
        f.write(f"workdir: {workdir}\n")

    print(f"Done. Wrote outputs to {workdir}")


if __name__ == "__main__":
    main()
