"""Conditional Flow Matching (CFM) training for unpaired distribution
transfer baseline.

Two variants (selected by config.fm_strategy):
  - "random":  independent coupling, x_0 ~ X_train, x_1 ~ Y_train each
               drawn independently per batch (Lipman 2023 vanilla CFM).
  - "ot":      mini-batch OT coupling, optimal-transport pairing within
               each minibatch (Tong 2024 OT-CFM, lower-variance training).

Linear path:
    x_t = (1 - t) * x_0 + t * x_1
    target velocity u_target(x_t, t) = x_1 - x_0
    loss = MSE(v_theta(x_t, t), u_target)

Sampling: ODE Euler from x_0 ~ X_eval, dx/dt = v_theta(x_t, t).

Pipeline (parallel to training/train_db.py):
  1. Train config.num_checkpoints checkpoints with per-sample t.
  2. Per-checkpoint cheap evaluation on the training split
     (rollout num_rollout_steps_cheap x repeat_cheap) -> results.csv.
  3. Sample the best checkpoint with the heavy protocol and write
     generated_endpoints.npy (+ y_star_indices.npy = -1) for
     scripts/reeval_baseline.py.

Output schema matches the analytic baselines:
  workdir/generated_endpoints.npy  [n_eval, repeat, d]
  workdir/y_star_indices.npy       [n_eval, repeat] all -1
  workdir/sample_summary.txt
  workdir/config_snapshot.py
plus training artifacts in checkpoints/ + train_log.csv + results.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from evaluation.metrics import (
    frechet_distance,
    match_sample_sizes,
    maximum_mean_discrepancy,
    sinkhorn_w2,
    sliced_wasserstein_distance,
)
from models.controller_net import ControllerNet
from training.losses import mse_controller_loss
from training.train_utils import TrainLogger, make_optimizer
from utils.config_utils import load_config, resolve_data_path
from utils.cli import add_setting_args, apply_setting_args
from utils.io_utils import ensure_dir, save_config_snapshot
from utils.seed import make_rng, set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ckpt_epochs(num_checkpoints: int, total_epochs: int) -> list[int]:
    if num_checkpoints <= 0:
        return []
    if total_epochs <= num_checkpoints:
        return list(range(1, total_epochs + 1))
    step = total_epochs // num_checkpoints
    epochs = [step * k for k in range(1, num_checkpoints + 1)]
    if epochs[-1] != total_epochs:
        epochs[-1] = total_epochs
    return epochs


def _impute_age_nan(ages: np.ndarray) -> np.ndarray:
    ages = np.asarray(ages, dtype=np.float64)
    finite = ages[np.isfinite(ages)]
    fill = float(np.median(finite)) if finite.size > 0 else 70.0
    return np.where(np.isnan(ages), fill, ages)


def _load_data(config):
    data_path = resolve_data_path(config)
    npz = np.load(data_path)
    X_train = np.asarray(npz["X_train"], dtype=np.float64)
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    X_eval = np.concatenate([
        np.asarray(npz["X_val"], dtype=np.float64),
        np.asarray(npz["X_test"], dtype=np.float64),
    ], axis=0)
    Y_eval = np.concatenate([
        np.asarray(npz["Y_val"], dtype=np.float64),
        np.asarray(npz["Y_test"], dtype=np.float64),
    ], axis=0)
    # Optional ages (for cond_on_age=True configs).
    if "X_train_ages" in npz.files:
        X_train_ages = _impute_age_nan(npz["X_train_ages"])
        X_eval_ages = _impute_age_nan(np.concatenate([
            np.asarray(npz["X_val_ages"], dtype=np.float64),
            np.asarray(npz["X_test_ages"], dtype=np.float64),
        ], axis=0))
    else:
        X_train_ages = None
        X_eval_ages = None
    return X_train, Y_train, X_eval, Y_eval, X_train_ages, X_eval_ages


def _ot_minibatch_pair(
    x_batch: torch.Tensor, y_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mini-batch OT pairing via Hungarian algorithm on squared cost.

    Returns (x_perm, y_perm) where x_perm[i] is paired with y_perm[i].
    Uses scipy.optimize.linear_sum_assignment for exact OT on small batches
    (B <= 256 fast enough). For batches with |x| != |y|, drops the larger
    side to size of smaller before matching.
    """
    from scipy.optimize import linear_sum_assignment
    Bx = x_batch.shape[0]
    By = y_batch.shape[0]
    n = min(Bx, By)
    x = x_batch[:n]
    y = y_batch[:n]
    cost = torch.cdist(x, y, p=2.0).pow(2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return x[row_ind], y[col_ind]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_one_epoch_fm(
    model, optimizer, X_train_t, Y_train_t,
    config, rng, logger, epoch, global_step, strategy: str, device,
    X_train_ages_t=None,
) -> tuple[float, int]:
    n_x = X_train_t.shape[0]
    n_y = Y_train_t.shape[0]
    bs = int(config.batch_size)
    n_steps = max(1, n_x // bs)
    perm_x = rng.permutation(n_x)
    t_eps = float(getattr(config, "t_eps", 1e-4))
    epoch_losses = []

    for step in range(n_steps):
        idx_x = perm_x[step * bs : (step + 1) * bs]
        if len(idx_x) == 0:
            continue
        idx_x_orig = idx_x
        x_0 = X_train_t[idx_x]
        # Pair: random shuffle Y_train each batch; OT-CFM = OT inside batch.
        idx_y = rng.choice(n_y, size=len(idx_x), replace=True)
        y_batch = Y_train_t[torch.from_numpy(idx_y).to(device=Y_train_t.device)]

        if strategy == "ot":
            x_0, x_1 = _ot_minibatch_pair(x_0, y_batch)
        elif strategy == "random":
            x_1 = y_batch
        else:
            raise ValueError(f"Unknown fm_strategy: {strategy!r}")

        B = x_0.shape[0]
        t_b = torch.from_numpy(
            rng.uniform(t_eps, 1.0 - t_eps, size=B).astype(np.float64),
        ).to(device=device)
        t_b_d = t_b.unsqueeze(-1)                                     # [B, 1]
        path_mode = str(getattr(config, "path_mode", "linear")).lower()
        if path_mode == "linear":
            # CFM (Lipman 2023) / SOC Matching: linear path + velocity target
            x_t = (1.0 - t_b_d) * x_0 + t_b_d * x_1                   # [B, d]
            u_target = x_1 - x_0                                      # [B, d]
        elif path_mode == "bridge":
            # DSB-lite (Bridge Matching): Brownian bridge interpolation
            sigma_bridge = float(getattr(config, "sigma_bridge", 0.5))
            bridge_std = sigma_bridge * torch.sqrt(
                t_b_d * (1.0 - t_b_d) + 1e-12
            )
            x_t = (1.0 - t_b_d) * x_0 + t_b_d * x_1 + bridge_std * torch.randn_like(x_0)
            # Drift target along Brownian bridge:
            #   E[dx_t/dt | x_t] = (x_1 - x_t) / (1 - t)
            u_target = (x_1 - x_t) / (1.0 - t_b_d + 1e-6)
        else:
            raise ValueError(f"Unknown path_mode: {path_mode!r}")

        x_t_f32 = x_t.to(dtype=torch.float32)
        t_f32 = t_b.to(dtype=torch.float32)
        u_target_f32 = u_target.to(dtype=torch.float32)

        # Optional age conditioning. For OT path, x_0 was permuted by
        # the OT pairing — but we still tag age by the X_train index of
        # the SOURCE row in the original batch (idx_x_orig). FM uses
        # x_0's identity for age, not x_1's.
        age_b = None
        if model.cond_on_age:
            assert X_train_ages_t is not None
            age_b = X_train_ages_t[
                torch.from_numpy(idx_x_orig).to(device=device)
            ].to(dtype=torch.float32).unsqueeze(-1)               # [B, 1]
            # If OT shuffled and reduced batch, slice age_b to match B.
            if age_b.shape[0] != B:
                age_b = age_b[:B]

        v_pred = model(x_t_f32, t_f32, age=age_b)
        loss = mse_controller_loss(v_pred, u_target_f32)
        optimizer.zero_grad()
        loss.backward()
        grad_clip = float(getattr(config, "grad_clip", 0.0) or 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip,
            )
        optimizer.step()

        global_step += 1
        loss_item = float(loss.item())
        epoch_losses.append(loss_item)
        if global_step % 10 == 0 or step == 0:
            logger.log_train(epoch=epoch, step=global_step, loss=loss_item)

    return (
        float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
        global_step,
    )


# ---------------------------------------------------------------------------
# Sampling: ODE Euler integration
# ---------------------------------------------------------------------------


@torch.no_grad()
def _fm_rollout(
    x_0: torch.Tensor, model, num_steps: int, device: torch.device,
    save_trajectories: bool = False,
    use_sde: bool = False, g: float = 1.0,
    rng: "np.random.Generator | None" = None,
    age: "torch.Tensor | None" = None,
) -> dict:
    """ODE Euler (default) or SDE Euler-Maruyama (when use_sde=True).

    ODE:  dx/dt = v_theta(x_t, t)
    SDE:  dx_t  = v_theta(x_t, t) dt + g * sqrt(dt) * eps,  eps ~ N(0, I)

    SDE mode is used by SOC Matching / DSB-lite baselines. ODE mode is
    the standard CFM rollout.
    """
    model.eval()
    B, d = x_0.shape
    dt = 1.0 / float(num_steps)
    x = x_0.to(device=device, dtype=torch.float64).clone()
    if save_trajectories:
        traj = torch.empty(
            (B, num_steps + 1, d), dtype=torch.float64, device=device,
        )
        traj[:, 0] = x
    sqrt_dt = float(np.sqrt(dt))
    for k in range(num_steps):
        t = torch.full(
            (B,), k * dt, dtype=torch.float32, device=device,
        )
        v = model(
            x.to(dtype=torch.float32), t, age=age,
        ).to(dtype=torch.float64)
        x = x + v * dt
        if use_sde:
            if rng is not None:
                noise = torch.from_numpy(
                    rng.standard_normal(x.shape).astype(np.float64),
                ).to(device=device)
            else:
                noise = torch.randn_like(x, dtype=torch.float64)
            x = x + g * sqrt_dt * noise
        if save_trajectories:
            traj[:, k + 1] = x
    out = {"x_1": x}
    if save_trajectories:
        out["trajectories"] = traj
    return out


def _cheap_eval_one_ckpt(
    model, ckpt_label: str, config,
    X_eval: np.ndarray, Y_eval: np.ndarray,
    device: torch.device, rng_seed: int,
    X_eval_ages: np.ndarray = None,
) -> dict:
    """Per-repeat batch protocol: each repeat draws fresh
    eval_batch_size samples from X_eval / Y_eval, rolls out the FM ODE,
    and computes (eval_batch, eval_batch) metrics."""
    n_X_eval, d = X_eval.shape
    n_Y_eval = Y_eval.shape[0]
    # Cheap in-training progress signal. The checkpoint used for reported
    # numbers is selected by scripts/select_ckpt_trainset.py on the training
    # split (select_steps x select_repeat), the same protocol as DBPS.
    repeat = int(config.repeat_cheap)
    num_steps = int(config.num_rollout_steps_cheap)

    eval_batch = int(getattr(config, "eval_batch_size", 0) or 0)
    if eval_batch <= 0:
        eval_batch = min(n_X_eval, n_Y_eval)
    eval_batch = min(eval_batch, n_X_eval, n_Y_eval)

    Y_eval_t_full = torch.from_numpy(Y_eval).to(device=device, dtype=torch.float64)

    swds, w2s, fds = [], [], []
    all_x_1: list[torch.Tensor] = []
    for r in range(repeat):
        rng_r = make_rng(rng_seed + r * 7919)
        idx_x = rng_r.choice(n_X_eval, eval_batch, replace=False).astype(np.int64)
        idx_y = rng_r.choice(n_Y_eval, eval_batch, replace=False).astype(np.int64)

        x_0_b = X_eval[idx_x]
        Y_b = Y_eval[idx_y]

        x_0_t = torch.tensor(x_0_b, dtype=torch.float64, device=device)
        # Age tensor for cheap eval (per-subject age tagged from X_eval).
        age_b_t = None
        if model.cond_on_age:
            assert X_eval_ages is not None
            age_b_t = torch.tensor(
                X_eval_ages[idx_x], dtype=torch.float32, device=device,
            ).unsqueeze(-1)
        out = _fm_rollout(
            x_0_t, model, num_steps, device, save_trajectories=False,
            use_sde=bool(getattr(config, "use_sde_fm", False)),
            g=float(getattr(config, "g", 1.0)),
            rng=rng_r,
            age=age_b_t,
        )
        x_1_b = out["x_1"]                                       # [eval_batch, d]
        Y_b_t = torch.from_numpy(Y_b).to(device=device, dtype=torch.float64)

        swds.append(sliced_wasserstein_distance(
            x_1_b, Y_b_t, num_projections=256, rng=rng_r,
        ))
        w2s.append(sinkhorn_w2(x_1_b, Y_b_t, epsilon=0.05))
        fds.append(frechet_distance(x_1_b, Y_b_t, cov_eps=1e-6))
        all_x_1.append(x_1_b)

    x_full = torch.cat(all_x_1, dim=0)
    mmd = maximum_mean_discrepancy(
        x_full, Y_eval_t_full, rng=make_rng(rng_seed + 9001),
    )

    def _ms(xs):
        a = np.asarray([v for v in xs if np.isfinite(v)], dtype=np.float64)
        if a.size == 0:
            return float("nan"), float("nan")
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

    swd_m, swd_s = _ms(swds)
    w2_m, w2_s = _ms(w2s)
    fd_m, fd_s = _ms(fds)
    return {
        "checkpoint": ckpt_label,
        "sampling_rho": 0.0,                                  # FM has no rho
        "swd_mean": swd_m, "swd_std": swd_s,
        "w2_mean": w2_m, "w2_std": w2_s,
        "fd_mean": fd_m, "fd_std": fd_s,
        "mmd": float(mmd),
        "n_eval": int(eval_batch),
        "n_repeat": int(repeat),
    }


def _append_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "checkpoint", "sampling_rho",
        "swd_mean", "swd_std", "w2_mean", "w2_std", "fd_mean", "fd_std",
        "mmd",
        "n_eval", "n_repeat",
    ]
    new_file = not path.is_file()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        f.flush()


def _heavy_sample_best_ckpt(
    model, ckpt_path: Path, config,
    X_eval: np.ndarray, device: torch.device,
    X_eval_ages: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load best ckpt -> heavy sample (400 step x 80 repeat). Returns
    (generated_endpoints [n_eval, repeat, d], y_star_indices = -1
    placeholder)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_eval, d = X_eval.shape
    # Per-setting repeat count (250 / 80 / 200) comes from the config.
    repeat = int(config.repeat_heavy)
    num_steps = int(config.num_rollout_steps_heavy)
    x_0 = np.repeat(X_eval[:, None, :], repeat, axis=1).reshape(n_eval * repeat, d)
    x_0_t = torch.tensor(x_0, dtype=torch.float64, device=device)
    age_t = None
    if model.cond_on_age:
        assert X_eval_ages is not None
        ages_tiled = np.repeat(
            X_eval_ages[:, None], repeat, axis=1,
        ).reshape(n_eval * repeat)
        age_t = torch.tensor(
            ages_tiled, dtype=torch.float32, device=device,
        ).unsqueeze(-1)
    # REPRODUCIBILITY: pass an explicitly seeded rng. Without it, _fm_rollout
    # falls back to torch.randn_like, i.e. the GLOBAL torch RNG whose state
    # depends on the preceding training -- so a stand-alone re-sample from the
    # same checkpoint produced different endpoints (this is what made the
    # SDE-sampled FM baselines, SOC-Matching / DSB-lite, drift). Deriving the
    # stream from config.seed makes ckpt + seed -> bit-identical endpoints.
    out = _fm_rollout(
        x_0_t, model, num_steps, device, save_trajectories=False,
        use_sde=bool(getattr(config, "use_sde_fm", False)),
        g=float(getattr(config, "g", 1.0)),
        rng=make_rng(int(config.seed) + 12345),
        age=age_t,
    )
    x_1 = out["x_1"].cpu().numpy().reshape(n_eval, repeat, d)
    ysi = np.full((n_eval, repeat), -1, dtype=np.int64)
    return x_1, ysi


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def _resolve_workdir(config) -> Path:
    if hasattr(config, "make_train_workdir_name") and callable(
        config.make_train_workdir_name
    ):
        name = config.make_train_workdir_name(config)
    else:
        name = config.make_workdir_name(config)
    return config.workdir_root / name


def train_fm(config) -> Path:
    set_global_seed(config.seed)
    rng = make_rng(config.seed)

    X_train, Y_train, X_eval, Y_eval, X_train_ages, X_eval_ages = _load_data(config)
    _, d = X_train.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_on_age = bool(getattr(config, "cond_on_age", False))
    age_emb_dim = int(getattr(config, "age_emb_dim", 1))
    model = ControllerNet(
        d=d, hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        cond_on_y_star=False,                                 # FM = u_theta(x, t)
        cond_on_age=cond_on_age,
        age_emb_dim=age_emb_dim,
        time_emb_dim=int(config.time_emb_dim),
        dropout=float(config.dropout),
        activation=str(getattr(config, "activation", "silu")),
        time_embedding_type=str(getattr(config, "time_embedding_type", "sinusoidal")),
        block_type=str(getattr(config, "block_type", "residual")),
    ).to(device)
    optimizer = make_optimizer(
        model,
        lr=float(config.lr),
        weight_decay=float(getattr(config, "weight_decay", 0.0)),
        beta1=float(getattr(config, "adam_beta1", 0.9)),
    )

    workdir = _resolve_workdir(config)
    ensure_dir(workdir)
    ensure_dir(workdir / "checkpoints")
    save_config_snapshot(config, workdir)
    logger = TrainLogger(workdir / "train_log.csv")

    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float64, device=device)
    X_train_ages_t = (
        torch.tensor(X_train_ages, dtype=torch.float64, device=device)
        if X_train_ages is not None else None
    )

    epochs_total = int(config.epochs)
    n_ckpt = int(getattr(config, "num_checkpoints", 25))
    ckpt_epoch_list = _ckpt_epochs(n_ckpt, epochs_total)
    results_csv = workdir / "results.csv"

    strategy = str(getattr(config, "fm_strategy", "random"))
    epoch_loss_history: list[float] = []
    global_step = 0

    for epoch_idx in range(epochs_total):
        epoch = epoch_idx + 1
        avg_loss, global_step = _train_one_epoch_fm(
            model, optimizer, X_train_t, Y_train_t,
            config, rng, logger, epoch, global_step, strategy, device,
            X_train_ages_t=X_train_ages_t,
        )
        epoch_loss_history.append(avg_loss)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "fm_strategy": strategy,
        }
        torch.save(ckpt, workdir / "checkpoints" / "last.pth")

        if epoch in ckpt_epoch_list:
            ckpt_label = f"epoch{epoch:03d}"
            torch.save(ckpt, workdir / "checkpoints" / f"ckpt_{ckpt_label}.pth")

            # Checkpoints are scored on the TRAINING split only. The canonical
            # selector for reported numbers is scripts/select_ckpt_trainset.py
            # (200 rollout steps x 40 repeats, training split).
            row = _cheap_eval_one_ckpt(
                model, ckpt_label, config, X_train, Y_train, device,
                rng_seed=config.seed + epoch,
                X_eval_ages=X_train_ages,
            )
            _append_results_csv(results_csv, [row])
            model.train()

    logger.close()

    # Heavy sample with the best ckpt by cheap SWD.
    import pandas as pd
    df = pd.read_csv(results_csv)
    df = df.dropna(subset=["swd_mean"])
    if df.empty:
        raise RuntimeError(
            f"No finite-swd ckpts in {results_csv}; FM training broke."
        )
    best = df.sort_values("swd_mean").iloc[0]
    best_ckpt_label = str(best["checkpoint"])
    best_ckpt_path = workdir / "checkpoints" / f"ckpt_{best_ckpt_label}.pth"
    print(
        f"[train_fm] best ckpt {best_ckpt_label} (cheap SWD={best['swd_mean']:.4f}); "
        "running heavy sample..."
    )

    ge, ysi = _heavy_sample_best_ckpt(
        model, best_ckpt_path, config, X_eval, device,
        X_eval_ages=X_eval_ages,
    )
    np.save(workdir / "generated_endpoints.npy", ge.astype(np.float64))
    np.save(workdir / "y_star_indices.npy", ysi)
    (workdir / "sample_summary.txt").write_text(
        f"FM ({strategy}) best_ckpt={best_ckpt_label}\n"
        f"n_eval={ge.shape[0]} repeat={ge.shape[1]} d={ge.shape[2]}\n"
    )
    print(f"DONE FM training+sample: {workdir}")
    return workdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--use_normalization", action="store_true")
    parser.add_argument("--no_normalization", action="store_true")
    parser.add_argument("--epochs", default=None, type=int)
    add_setting_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_setting_args(config, args)
    if args.use_normalization:
        config.use_normalization = True
        config.suffix = "_normalization"
    if args.no_normalization:
        config.use_normalization = False
        config.suffix = "_raw"
    if args.epochs is not None:
        config.epochs = int(args.epochs)

    train_fm(config)


if __name__ == "__main__":
    main()
