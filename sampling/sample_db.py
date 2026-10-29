"""DBPS sampling and evaluation (Algorithm 2 and the evaluation protocol).

At sampling time the DBPS controller is
    u = (1 - rho) u_end(x_t, t; y*) + rho u_theta(x_t, t, age),
with the anchor y* drawn uniformly from Y_train, integrated with
Euler-Maruyama.

Evaluation protocol (evaluate_checkpoint): for each repeat r, with a
generator seeded by seed + 7919 r, draw eval_batch_size sources from the
source pool and eval_batch_size targets from the reference pool, draw one
anchor per source, roll out the controlled SDE, and compute SWD (256
projections) and FD between the generated endpoints and the target batch.
MMD is computed once between all generated endpoints and the full
reference pool. Two pools are used:
  split="eval":  sources X_val + X_test, reference Y_val + Y_test
                 (reported numbers);
  split="train": sources X_train, reference Y_train
                 (checkpoint selection).

CLI:
  python -m sampling.sample_db --config configs/methods/dbps.py --gamma 100 \
      --checkpoint <run>/checkpoints/ckpt_epoch150.pth --rho 0.1 \
      --split eval --repeat 250 --steps 400 --out metrics.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from controls.db_control import compute_u_anchor
from dynamics.rollout import rollout
from dynamics.transition_computation import TransitionComputation
from evaluation.metrics import (
    frechet_distance,
    maximum_mean_discrepancy,
    sliced_wasserstein_distance,
)
from models.controller_net import ControllerNet
from utils.config_utils import load_config, resolve_data_path
from utils.seed import make_rng, set_global_seed


def _load_eval_data(config):
    """Returns: X_eval, Y_eval, Y_train, L, ot_top_idx_eval,
                X_eval_ages, age_top_idx_eval."""
    data_path = resolve_data_path(config)
    npz = np.load(data_path)
    X_eval = np.concatenate([
        np.asarray(npz["X_val"], dtype=np.float64),
        np.asarray(npz["X_test"], dtype=np.float64),
    ], axis=0)
    Y_eval = np.concatenate([
        np.asarray(npz["Y_val"], dtype=np.float64),
        np.asarray(npz["Y_test"], dtype=np.float64),
    ], axis=0)
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    L = np.asarray(npz["L"], dtype=np.float64)
    ot_top_idx_eval = np.concatenate([
        np.asarray(npz["ot_top_indices_val"], dtype=np.int64),
        np.asarray(npz["ot_top_indices_test"], dtype=np.int64),
    ], axis=0)
    # Source ages (median-imputed) for age conditioning, and age-ranked
    # Y_train indices used by the age-paired baselines.
    if "X_train_ages" in npz.files:
        def _impute(a):
            a = np.asarray(a, dtype=np.float64)
            finite = a[np.isfinite(a)]
            fill = float(np.median(finite)) if finite.size > 0 else 70.0
            return np.where(np.isnan(a), fill, a)
        Y_train_ages = _impute(npz["Y_train_ages"])
        X_eval_ages = _impute(np.concatenate([
            np.asarray(npz["X_val_ages"], dtype=np.float64),
            np.asarray(npz["X_test_ages"], dtype=np.float64),
        ], axis=0))
        K_max = int(getattr(config, "num_candidate_sample", 50))
        age_dist = np.abs(
            X_eval_ages[:, None] - Y_train_ages[None, :],
        )                                                   # [n_X_eval, n_Y_train]
        age_top_idx_eval = np.argsort(age_dist, axis=1)[:, :K_max].astype(np.int64)
    else:
        X_eval_ages = None
        age_top_idx_eval = None
    return (
        X_eval, Y_eval, Y_train, L, ot_top_idx_eval,
        X_eval_ages, age_top_idx_eval,
    )


def _draw_anchors(
    Y_train: np.ndarray, ot_top_idx_eval: np.ndarray,
    n_eval: int, repeat: int, strategy: str,
    rng: np.random.Generator,
    age_top_idx_eval: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor y* per (eval_sample, repeat) per `candidate_strategy`."""
    n_train = Y_train.shape[0]
    d = Y_train.shape[1]
    y_star_idx = np.empty((n_eval, repeat), dtype=np.int64)
    if strategy == "ot_top10":
        for r in range(repeat):
            for i in range(n_eval):
                y_star_idx[i, r] = int(ot_top_idx_eval[i, 0])
    elif strategy == "random10":
        for r in range(repeat):
            for i in range(n_eval):
                y_star_idx[i, r] = int(rng.choice(n_train))
    elif strategy == "age_top50":
        assert age_top_idx_eval is not None, (
            "age_top50 strategy requires age_top_idx_eval"
        )
        for r in range(repeat):
            for i in range(n_eval):
                y_star_idx[i, r] = int(age_top_idx_eval[i, 0])
    else:
        raise ValueError(f"Unknown candidate_strategy: {strategy!r}")
    y_star_np = Y_train[y_star_idx]
    return y_star_idx, y_star_np


def _impute_ages(a: np.ndarray) -> np.ndarray:
    """Replace NaN ages with the array's finite median (or 70 as fallback)."""
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    fill = float(np.median(finite)) if finite.size > 0 else 70.0
    return np.where(np.isnan(a), fill, a)


def load_pool(config, split: str = "eval"):
    """Return (X_source, Y_reference, Y_train, L, X_source_ages) for a split.

    split="eval":  X_val + X_test sources, Y_val + Y_test reference.
    split="train": X_train sources, Y_train reference.
    Ages are median-imputed within the source pool.
    """
    npz = np.load(resolve_data_path(config), allow_pickle=True)
    if split == "eval":
        X = np.concatenate([npz["X_val"], npz["X_test"]], axis=0).astype(np.float64)
        Y_ref = np.concatenate([npz["Y_val"], npz["Y_test"]], axis=0).astype(np.float64)
        ages = (np.concatenate([npz["X_val_ages"], npz["X_test_ages"]], axis=0)
                if "X_val_ages" in npz.files else None)
    elif split == "train":
        X = np.asarray(npz["X_train"], dtype=np.float64)
        Y_ref = np.asarray(npz["Y_train"], dtype=np.float64)
        ages = npz["X_train_ages"] if "X_train_ages" in npz.files else None
    else:
        raise ValueError(f"split must be 'eval' or 'train', got {split!r}")
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    L = np.asarray(npz["L"], dtype=np.float64)
    X_ages = _impute_ages(ages) if ages is not None else None
    return X, Y_ref, Y_train, L, X_ages


def build_controller(config, d: int, device) -> ControllerNet:
    return ControllerNet(
        d=d, hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        cond_on_y_star=False,
        cond_on_age=bool(getattr(config, "cond_on_age", False)),
        age_emb_dim=int(getattr(config, "age_emb_dim", 1)),
        time_emb_dim=int(config.time_emb_dim),
        dropout=0.0,
        activation=str(getattr(config, "activation", "silu")),
        time_embedding_type=str(getattr(config, "time_embedding_type", "sinusoidal")),
        block_type=str(getattr(config, "block_type", "residual")),
    ).to(device)


def evaluate_checkpoint(
    config, checkpoint_path, rho: float,
    X: np.ndarray, Y_ref: np.ndarray, Y_train: np.ndarray, X_ages,
    transition: TransitionComputation, device,
    n_repeat: int, n_steps: int, seed: int = 42, eval_batch_size: int = 100,
) -> dict:
    """Evaluate one DBPS checkpoint at sampling weight rho (see module docstring)."""
    set_global_seed(seed)
    n_X, d = X.shape
    n_Y = Y_ref.shape[0]
    n_train = Y_train.shape[0]
    eval_batch = min(int(eval_batch_size), n_X, n_Y)

    model = build_controller(config, d, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cond_on_age = bool(getattr(config, "cond_on_age", False))
    rho = float(rho)

    def make_controller_fn(age_b_t):
        def controller_fn(x_t, t, y_star_arg, candidates_arg):
            u_anchor = compute_u_anchor(
                x_t, t, y_star_arg,
                gamma=config.gamma, beta=config.beta, g=config.g,
                transition=transition,
            )
            with torch.no_grad():
                u_net = model(
                    x_t.to(dtype=torch.float32, device=device),
                    t.to(dtype=torch.float32, device=device),
                    age=age_b_t,
                ).to(dtype=torch.float64)
            return (1.0 - rho) * u_anchor + rho * u_net
        return controller_fn

    swds, fds, all_x_1 = [], [], []
    for r in range(int(n_repeat)):
        rng_r = make_rng(seed + r * 7919)
        idx_x = rng_r.choice(n_X, eval_batch, replace=False).astype(np.int64)
        idx_y = rng_r.choice(n_Y, eval_batch, replace=False).astype(np.int64)
        y_star_idx = rng_r.choice(n_train, size=eval_batch).astype(np.int64)

        x_0_t = torch.tensor(X[idx_x], dtype=torch.float64, device=device)
        y_star_t = torch.tensor(Y_train[y_star_idx], dtype=torch.float64, device=device)
        age_b_t = None
        if cond_on_age:
            age_b_t = torch.tensor(
                X_ages[idx_x], dtype=torch.float32, device=device,
            ).unsqueeze(-1)

        out = rollout(
            x_0=x_0_t, controller_fn=make_controller_fn(age_b_t),
            y_star=y_star_t, candidates=None,
            transition=transition, num_steps=int(n_steps),
            g=float(config.g), use_sde=bool(config.use_sde),
            save_trajectories=False, save_controls=False, rng=rng_r,
        )
        x_1_b = out["x_1"]
        Y_b_t = torch.from_numpy(Y_ref[idx_y]).to(device=device, dtype=torch.float64)
        swds.append(float(sliced_wasserstein_distance(
            x_1_b, Y_b_t, num_projections=256, rng=rng_r,
        )))
        fds.append(float(frechet_distance(x_1_b, Y_b_t, cov_eps=1e-6)))
        all_x_1.append(x_1_b)

    mmd = float(maximum_mean_discrepancy(
        torch.cat(all_x_1, dim=0),
        torch.from_numpy(Y_ref).to(device=device, dtype=torch.float64),
        rng=make_rng(seed + 9001),
    ))

    def _stat(xs):
        a = np.asarray(xs, dtype=np.float64)
        return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)

    swd_m, swd_s = _stat(swds)
    fd_m, fd_s = _stat(fds)
    return {
        "checkpoint": Path(checkpoint_path).stem.replace("ckpt_", ""),
        "rho": rho,
        "swd_mean": swd_m, "swd_std": swd_s,
        "fd_mean": fd_m, "fd_std": fd_s,
        "mmd": mmd,
        "n_eval": eval_batch, "n_repeat": int(n_repeat), "n_steps": int(n_steps),
        "swd_per_repeat": swds,
    }


FIELDS = ["checkpoint", "rho", "swd_mean", "swd_std", "fd_mean", "fd_std",
          "mmd", "n_eval", "n_repeat", "n_steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--checkpoint", required=True, nargs="+",
                        help="one or more checkpoint files")
    parser.add_argument("--rho", type=float, required=True, nargs="+")
    parser.add_argument("--split", choices=("eval", "train"), default="eval")
    parser.add_argument("--repeat", type=int, default=None,
                        help="default: config.repeat_heavy (eval) / config.select_repeat (train)")
    parser.add_argument("--steps", type=int, default=None,
                        help="default: config.num_rollout_steps_heavy (eval) / config.select_steps (train)")
    parser.add_argument("--data", default=None, help="override processed_data.npz")
    parser.add_argument("--out", required=True, help="output CSV (appended)")
    args = parser.parse_args()

    config = load_config(args.config)
    config.gamma = int(args.gamma) if float(args.gamma).is_integer() else float(args.gamma)
    if args.data is not None:
        config.processed_data_path = Path(args.data)
    if args.split == "eval":
        repeat = args.repeat or int(config.repeat_heavy)
        steps = args.steps or int(config.num_rollout_steps_heavy)
    else:
        repeat = args.repeat or int(config.select_repeat)
        steps = args.steps or int(config.select_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y_ref, Y_train, L, X_ages = load_pool(config, args.split)
    transition = TransitionComputation(
        L=L, beta=config.beta, cov_regularization=config.cov_regularization,
        device=device,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.is_file()
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "gamma"] + FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for ckpt_path in args.checkpoint:
            for rho in args.rho:
                row = evaluate_checkpoint(
                    config, ckpt_path, rho, X, Y_ref, Y_train, X_ages,
                    transition, device, n_repeat=repeat, n_steps=steps,
                    seed=int(config.reeval_seed),
                    eval_batch_size=int(config.eval_batch_size),
                )
                writer.writerow({"split": args.split, "gamma": config.gamma, **row})
                f.flush()
                print(f"{row['checkpoint']} rho={rho}: SWD {row['swd_mean']:.4f} "
                      f"+/- {row['swd_std']:.4f}  FD {row['fd_mean']:.2f}  MMD {row['mmd']:.3e}")


if __name__ == "__main__":
    main()
