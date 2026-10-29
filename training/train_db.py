"""Training entry for the DBPS population controller u_theta.

For each minibatch element: draw x_0 ~ X_train and t ~ U(t_eps, 1 - t_eps),
simulate the uncontrolled SDE to x_t, draw K candidates uniformly from
Y_train, and regress u_theta(x_t, t, age) onto the empirical population
label u_pop = sum_k alpha_k u_end(x_t, t; y_k) (Algorithm 1).

Training randomness is re-seeded at every epoch with config.seed + epoch.
config.num_checkpoints checkpoints are saved at evenly spaced epochs; no
evaluation is run during training. At the end, the late-epoch mean loss
is compared with the early-epoch mean and CONVERGENCE_WARNING.txt is
written if the loss did not decrease enough.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from controls.db_control import compute_u_pop_label
from dynamics.transition_computation import TransitionComputation
from models.controller_net import ControllerNet
from training.losses import mse_controller_loss
from training.train_utils import TrainLogger, make_optimizer
from utils.config_utils import load_config, resolve_data_path
from utils.cli import add_setting_args, apply_setting_args
from utils.io_utils import ensure_dir, save_config_snapshot
from utils.seed import make_rng, set_global_seed


# ---------------------------------------------------------------------------
# Workdir helpers
# ---------------------------------------------------------------------------


def _resolve_workdir(config) -> Path:
    if hasattr(config, "make_train_workdir_name") and callable(
        config.make_train_workdir_name
    ):
        name = config.make_train_workdir_name(config)
    else:
        name = config.make_workdir_name(config)
    return config.workdir_root / name


def _ckpt_epochs(num_checkpoints: int, total_epochs: int) -> list[int]:
    """Return list of 1-indexed epoch numbers to save checkpoints at.

    e.g. (25, 200) -> [8, 16, 24, ..., 200].
    Degenerate cases:
      total <= num: save at every epoch [1..total]
      num <= 0:    return []
    """
    if num_checkpoints <= 0:
        return []
    if total_epochs <= num_checkpoints:
        return list(range(1, total_epochs + 1))
    step = total_epochs // num_checkpoints
    epochs = [step * k for k in range(1, num_checkpoints + 1)]
    if epochs[-1] != total_epochs:
        epochs[-1] = total_epochs
    return epochs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _impute_age_nan(ages: np.ndarray) -> np.ndarray:
    """Replace NaN ages with the array's finite median (or 70 as fallback).

    Missing ages would otherwise propagate as NaN through the network
    input and the training loss.
    """
    ages = np.asarray(ages, dtype=np.float64)
    finite = ages[np.isfinite(ages)]
    fill = float(np.median(finite)) if finite.size > 0 else 70.0
    return np.where(np.isnan(ages), fill, ages)


def _load_data(config):
    """Load X_train, Y_train, the graph Laplacian L and (if present) the
    median-imputed source ages for the training split."""
    data_path = resolve_data_path(config)
    npz = np.load(data_path)
    X_train = np.asarray(npz["X_train"], dtype=np.float64)
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    L = np.asarray(npz["L"], dtype=np.float64)
    if "X_train_ages" in npz.files:
        X_train_ages = _impute_age_nan(npz["X_train_ages"])
    else:
        X_train_ages = None
    return X_train, Y_train, L, X_train_ages


# ---------------------------------------------------------------------------
# Candidate drawing
# ---------------------------------------------------------------------------


def _draw_candidates_random(
    Y_train_t: torch.Tensor, n_y: int, batch_size: int, K: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    cand_idx = np.empty((batch_size, K), dtype=np.int64)
    for b in range(batch_size):
        cand_idx[b] = rng.choice(n_y, size=K, replace=False)
    idx_t = torch.from_numpy(cand_idx).to(device=Y_train_t.device)
    return Y_train_t[idx_t]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model, optimizer, X_train_t, Y_train_t,
    X_train_ages_t,                                        # [n_x] or None
    transition, config, rng, logger,
    epoch, global_step,
    label_fn, device,
) -> tuple[float, int]:
    n_x = X_train_t.shape[0]
    n_y = Y_train_t.shape[0]
    n_steps = max(1, n_x // config.batch_size)
    perm = rng.permutation(n_x)
    K = int(config.num_candidate_sample)
    t_eps = float(getattr(config, "t_eps", 1e-4))
    strategy = getattr(config, "candidate_strategy", "random10")
    if strategy != "random10":
        raise ValueError(f"Unknown candidate_strategy: {strategy!r}")
    epoch_losses = []

    for step in range(n_steps):
        batch_abs_idx = perm[step * config.batch_size : (step + 1) * config.batch_size]
        if len(batch_abs_idx) == 0:
            continue
        x_0 = X_train_t[batch_abs_idx]                          # already on device
        B = x_0.shape[0]

        # Per-sample t: each batch element gets its own time stamp.
        t_b_np = rng.uniform(t_eps, 1.0 - t_eps, size=B).astype(np.float64)
        t_b = torch.from_numpy(t_b_np).to(device=device)        # [B] float64 on device
        x_t = transition.sample_x_t_uncontrolled(x_0, t_b, rng=rng)  # device

        candidates = _draw_candidates_random(Y_train_t, n_y, B, K, rng)

        y_star = candidates[:, 0, :]                            # device
        u_label = label_fn(x_t, t_b, candidates, y_star)        # device

        x_t_f32 = x_t.to(dtype=torch.float32)
        t_batch = t_b.to(dtype=torch.float32)
        u_label_f32 = u_label.to(dtype=torch.float32)

        # Age conditioning (only if the model expects it).
        age_b = None
        if model.cond_on_age:
            assert X_train_ages_t is not None
            age_b = X_train_ages_t[
                torch.from_numpy(batch_abs_idx).to(device=device)
            ].to(dtype=torch.float32).unsqueeze(-1)             # [B, 1]

        u_pred = model(x_t_f32, t_batch, y_star=None, age=age_b)

        loss = mse_controller_loss(u_pred, u_label_f32)
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

    avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    return avg, global_step


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------


def _check_convergence(loss_history: Sequence[float]) -> tuple[str, str]:
    """Return (status, message). status in {OK, WARN_WEAK, WARN_NOT_CONVERGED}.

    Early window = first 20% of epochs, late window = last 20%, with a
    floor of 5 epochs each. We compare the late mean to the early mean.
    """
    n = len(loss_history)
    if n < 6:
        return "OK", f"too few epochs ({n}) to assess"
    win = max(5, n // 5)
    early = float(np.mean(loss_history[:win]))
    late = float(np.mean(loss_history[-win:]))
    if not np.isfinite(early) or early <= 1e-12:
        return "OK", f"early avg = {early}; cannot compute ratio"
    ratio = late / early
    if ratio > 1.0:
        return "WARN_NOT_CONVERGED", (
            f"final/early loss ratio = {ratio:.3f} > 1.0 "
            f"(loss not decreasing; window={win} epochs)"
        )
    if ratio > 0.7:
        return "WARN_WEAK", (
            f"final/early loss ratio = {ratio:.3f} "
            f"(weak convergence; window={win} epochs)"
        )
    return "OK", f"final/early loss ratio = {ratio:.3f} (window={win} epochs)"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_loop(
    config,
    label_fn_factory: Callable[
        [TransitionComputation],
        Callable[[torch.Tensor, float, torch.Tensor, torch.Tensor], torch.Tensor],
    ],
) -> Path:
    # Seed model initialization and deterministic backend settings once.
    # Training stochasticity is re-seeded per epoch below with
    # config.seed + epoch.
    set_global_seed(int(config.seed))

    X_train, Y_train, L, X_train_ages = _load_data(config)
    n_train, d = X_train.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transition = TransitionComputation(
        L=L, beta=config.beta, cov_regularization=config.cov_regularization,
        device=device,
    )

    cond_on_age = bool(getattr(config, "cond_on_age", False))
    age_emb_dim = int(getattr(config, "age_emb_dim", 1))
    model = ControllerNet(
        d=d, hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        cond_on_y_star=False,
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

    # All training tensors live on the transition device.
    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float64, device=device)
    X_train_ages_t = (
        torch.tensor(X_train_ages, dtype=torch.float64, device=device)
        if X_train_ages is not None else None
    )

    label_fn = label_fn_factory(transition)

    epochs_total = int(config.epochs)
    n_ckpt = int(getattr(config, "num_checkpoints", 25))
    ckpt_epoch_list = _ckpt_epochs(n_ckpt, epochs_total)

    epoch_loss_history: list[float] = []
    global_step = 0

    for epoch_idx in range(epochs_total):
        epoch = epoch_idx + 1                                 # 1-indexed
        epoch_seed = int(config.seed) + epoch
        set_global_seed(epoch_seed)
        epoch_rng = make_rng(epoch_seed)
        avg_loss, global_step = _train_one_epoch(
            model, optimizer, X_train_t, Y_train_t,
            X_train_ages_t,
            transition, config, epoch_rng, logger, epoch, global_step,
            label_fn, device,
        )
        epoch_loss_history.append(avg_loss)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "epoch_seed": epoch_seed,
            "loss": avg_loss,
            "controller_type": config.controller_type,
            "gamma": config.gamma,
            "rho": getattr(config, "rho", None),
        }
        torch.save(ckpt, workdir / "checkpoints" / "last.pth")

        if epoch in ckpt_epoch_list:
            ckpt_label = f"epoch{epoch:03d}"
            torch.save(ckpt, workdir / "checkpoints" / f"ckpt_{ckpt_label}.pth")

    logger.close()

    status, msg = _check_convergence(epoch_loss_history)
    print(f"[convergence] {status}: {msg}")
    if status != "OK":
        (workdir / "CONVERGENCE_WARNING.txt").write_text(
            f"{status}: {msg}\nloss history (per epoch avg, last 30):\n"
            + "\n".join(f"{i}: {x:.6e}" for i, x in enumerate(epoch_loss_history[-30:]))
            + "\n"
        )

    print(f"DONE training: {workdir}")
    return workdir


def train_dbps(config) -> Path:
    """Train u_theta(x_t, t, age) against the population label u_pop.

    rho is not used during training; it is applied at sampling time.
    """
    def factory(transition):
        def label_fn(x_t, t, candidates, y_star):
            return compute_u_pop_label(
                x_t, t, candidates,
                gamma=config.gamma, beta=config.beta, g=config.g,
                transition=transition,
            )
        return label_fn

    return _train_loop(config, label_fn_factory=factory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--use_normalization", action="store_true")
    parser.add_argument("--no_normalization", action="store_true")
    parser.add_argument("--gamma", default=None, type=float)
    parser.add_argument("--epochs", default=None, type=int)
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
    if args.epochs is not None:
        config.epochs = int(args.epochs)

    if config.controller_type != "unconditional_pop":
        raise ValueError(
            f"Unsupported controller_type: {config.controller_type!r}"
        )
    train_dbps(config)


if __name__ == "__main__":
    main()
