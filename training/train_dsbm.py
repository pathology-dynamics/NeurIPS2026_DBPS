"""Full DSBM -- Diffusion Schroedinger Bridge Matching (Shi et al., NeurIPS 2023),
implemented as a controlled upgrade from the existing `dsb_lite` baseline.

WHAT DSB-lite IS (training/train_fm.py, path_mode="bridge")
    One pass: minibatch Hungarian OT coupling (drawn fresh per batch but never
    *refined*), Brownian-bridge interpolation, a single forward drift regression
    on (x_1 - x_t)/(1 - t). No backward process. No IPF/IMF iteration. It is one
    half of one outer iteration of DSBM.

WHAT THIS FILE ADDS (and nothing else)
    1. A second, backward drift network of identical architecture.
    2. Iterated Markovian Fitting: alternating forward / backward bridge-matching
       fits (the Markovian projection).
    3. The endpoint coupling is REGENERATED every outer iteration by simulating
       the *current learned* process (the reciprocal projection) -- it is NOT the
       frozen minibatch-OT coupling. Only iteration 1 uses the OT coupling, so
       that iteration 1 is a faithful reproduction of DSB-lite's starting point.
    4. Multiple outer iterations with per-iteration checkpoints, so the existing
       leak-free selector (scripts/select_ckpt_trainset.py) selects the outer
       iteration exactly the way it selects a checkpoint for every other
       learned baseline.

HELD IDENTICAL TO DSB-lite (do not touch)
    Brownian reference bridge with sigma = 0.5; initial coupling = minibatch
    Hungarian exact OT; ControllerNet architecture (hidden_dim / num_layers /
    activation / time_embedding_type / block_type / dropout); 84-dim inputs and
    the same processed_data.npz; optimizer family, lr, batch_size, weight_decay,
    adam_beta1, grad_clip, t_eps; the same splits and seed.

THE ONE DELIBERATE DEVIATION
    DSB-lite trains on a sigma = 0.5 bridge but *samples* with diffusion
    g = 1.0 (configs.base default) -- the path measure it simulates is not the one
    it regressed. That mismatch is harmless for a one-pass method but is fatal
    for IMF: the reciprocal projection must simulate the same diffusion the
    Markovian projection regressed, otherwise the iteration has no fixed point.
    So DSBM sets g = sigma_bridge = 0.5 (config field `g`). This is a
    correctness requirement of the algorithm, not a tuning choice.

BACKWARD DIRECTION VIA BRIDGE SYMMETRY
    A Brownian bridge is symmetric under t -> 1-t with the endpoints swapped:
        x_t = (1-t) a + t b + sigma sqrt(t(1-t)) eps
    with s = 1-t reads as the same bridge from b to a. So the backward drift is
    fit and simulated with exactly the same code as the forward one, just called
    with (start, end) = (x_1, x_0) and its own time variable s in [0, 1]. No
    separate reverse-time integrator, and no extra approximation.

OUTPUTS (schema-compatible with every other baseline)
    workdir/checkpoints/ckpt_iter%03d.pth   one per outer iteration;
                                            "model_state_dict" = FORWARD net, so
                                            select_ckpt_trainset.py works
                                            unmodified
    workdir/dsbm_trajectory.csv             per-iteration log
    workdir/results.csv                     checkpoint + TRAIN-side cheap SWD
                                            (leak-free; see note in _write_results)
    workdir/train_log.csv, config_snapshot.py
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import (
    frechet_distance,
    maximum_mean_discrepancy,
    sliced_wasserstein_distance,
)
from controls.db_control import U_STAR_CLAMP
from models.controller_net import ControllerNet
from training.losses import mse_controller_loss
from training.train_fm import _fm_rollout, _impute_age_nan, _ot_minibatch_pair
from training.train_utils import TrainLogger, make_optimizer
from utils.config_utils import load_config, resolve_data_path
from utils.cli import add_setting_args, apply_setting_args
from utils.io_utils import ensure_dir, save_config_snapshot
from utils.seed import make_rng, set_global_seed


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _resolve_npz(config):
    return resolve_data_path(config)


def _load_data(config):
    npz = np.load(_resolve_npz(config))
    X_train = np.asarray(npz["X_train"], dtype=np.float64)
    Y_train = np.asarray(npz["Y_train"], dtype=np.float64)
    X_eval = np.concatenate([np.asarray(npz["X_val"], np.float64),
                             np.asarray(npz["X_test"], np.float64)], axis=0)
    Y_eval = np.concatenate([np.asarray(npz["Y_val"], np.float64),
                             np.asarray(npz["Y_test"], np.float64)], axis=0)
    if "X_train_ages" in npz.files:
        X_train_ages = _impute_age_nan(npz["X_train_ages"])
        X_eval_ages = _impute_age_nan(np.concatenate(
            [np.asarray(npz["X_val_ages"], np.float64),
             np.asarray(npz["X_test_ages"], np.float64)], axis=0))
    else:
        X_train_ages = X_eval_ages = None
    return X_train, Y_train, X_eval, Y_eval, X_train_ages, X_eval_ages


# ---------------------------------------------------------------------------
# Bridge matching (the Markovian projection), shared by both directions
# ---------------------------------------------------------------------------


def _bridge_sample(start, end, t_col, sigma, rng, device):
    """Brownian bridge from `start` (at time 0) to `end` (at time 1), and the
    corresponding drift target. Identical algebra to train_fm's path_mode
    'bridge' -- this is the piece that must NOT drift from DSB-lite."""
    noise = torch.from_numpy(
        rng.standard_normal(tuple(start.shape)).astype(np.float64)
    ).to(device=device)
    bridge_std = sigma * torch.sqrt(t_col * (1.0 - t_col) + 1e-12)
    x_t = (1.0 - t_col) * start + t_col * end + bridge_std * noise
    u_target = (end - x_t) / (1.0 - t_col + 1e-6)
    return x_t, u_target


@torch.no_grad()
def _dsbm_rollout(x_0, model, num_steps, device, g, rng, age=None,
                  parameterization="endpoint", drift_clamp=None, stats=None):
    """Euler-Maruyama for the bridge-matched SDE.

    parameterization="endpoint" (default): the network predicts the ENDPOINT
    x1_hat and the drift is formed analytically as (x1_hat - x_t)/(1 - t).
    Because the grid is t_k = k/N, at the last step 1 - t = dt exactly, so
    v*dt = x1_hat - x and the trajectory lands ON the predicted endpoint with
    no extrapolation. The drift-parameterised version instead needs the network
    to *emit* a value of magnitude N*||displacement|| at t = 1 - 1/N -- a
    regime training never covers (t <= 1 - t_eps) -- which under-shoots, leaves
    the injected noise uncompensated, and made IMF diverge.

    No noise is injected on the final step: the reference bridge has zero
    variance at t = 1, so perturbing after the terminal landing would add
    sigma^2*dt of pure, uncompensated variance every outer iteration.

    `drift_clamp` bounds the reconstructed drift per element. The endpoint
    parameterisation's drift is UNBOUNDED: once the injected noise pushes x_t
    off the bridge manifold, x1_hat is an extrapolation and the 1/(1-t) factor
    amplifies it by up to `num_steps`, which is what blew up the real-data runs
    (training loss fell monotonically while the rollout oscillated between 2 and
    31). The bound is not a DSBM-specific hyperparameter: it is
    controls.db_control.U_STAR_CLAMP = 20.0, the same per-element control clamp
    this repo already applies to DBPS's own u* at db_control.py:194 and :275.
    Pass `stats` (a dict) to record how often it actually binds.
    """
    model.eval()
    B = x_0.shape[0]
    dt = 1.0 / float(num_steps)
    sqrt_dt = float(np.sqrt(dt))
    x = x_0.to(device=device, dtype=torch.float64).clone()
    endpoint = (str(parameterization).lower() == "endpoint")
    cl = None if drift_clamp is None else float(drift_clamp)
    for k in range(num_steps):
        t_val = k * dt
        t = torch.full((B,), t_val, dtype=torch.float32, device=device)
        out = model(x.to(torch.float32), t, age=age).to(torch.float64)
        v = (out - x) / (1.0 - t_val) if endpoint else out
        if cl is not None and cl > 0:
            if stats is not None:
                stats["clamped"] = stats.get("clamped", 0) + int(
                    (v.abs() > cl).sum().item())
                stats["total"] = stats.get("total", 0) + v.numel()
            v = torch.clamp(v, -cl, cl)
        x = x + v * dt
        if k < num_steps - 1:
            noise = torch.from_numpy(
                rng.standard_normal(tuple(x.shape)).astype(np.float64)
            ).to(device=device)
            x = x + g * sqrt_dt * noise
    return x


def _fit_direction(model, optimizer, S, E, ages, config, rng, device,
                   epochs, logger, tag, global_step):
    """Regress one direction's Markovian projection on the CURRENT coupling
    (S[i], E[i]). Optimizer settings are DSB-lite's, unchanged."""
    n = S.shape[0]
    bs = int(config.batch_size)
    n_steps = max(1, n // bs)
    t_eps = float(getattr(config, "t_eps", 1e-4))
    sigma = float(getattr(config, "sigma_bridge", 0.5))
    grad_clip = float(getattr(config, "grad_clip", 0.0) or 0.0)
    cond_age = bool(model.cond_on_age)
    endpoint = (str(getattr(config, "dsbm_parameterization", "endpoint")).lower()
                == "endpoint")
    model.train()
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for step in range(n_steps):
            idx = perm[step * bs:(step + 1) * bs]
            if len(idx) == 0:
                continue
            idx_t = torch.from_numpy(idx).to(device=device)
            s_b, e_b = S[idx_t], E[idx_t]
            B = s_b.shape[0]
            t_b = torch.from_numpy(
                rng.uniform(t_eps, 1.0 - t_eps, size=B).astype(np.float64)
            ).to(device=device)
            x_t, u_drift = _bridge_sample(
                s_b, e_b, t_b.unsqueeze(-1), sigma, rng, device)
            # endpoint parameterisation: regress x1 itself; the drift is
            # reconstructed analytically at rollout time. Same network, same
            # loss family -- only the target changes.
            target = e_b if endpoint else u_drift
            age_b = ages[idx_t].to(torch.float32).unsqueeze(-1) if cond_age else None
            pred = model(x_t.to(torch.float32), t_b.to(torch.float32), age=age_b)
            loss = mse_controller_loss(pred, target.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            global_step += 1
            losses.append(float(loss.item()))
            if global_step % 50 == 0:
                logger.log_train(epoch=0, step=global_step, loss=float(loss.item()))
    return (float(np.mean(losses)) if losses else float("nan")), global_step


# ---------------------------------------------------------------------------
# Reciprocal projection: regenerate the coupling by simulating the learned SDE
# ---------------------------------------------------------------------------


@torch.no_grad()
def _simulate(model, x0_t, ages_t, config, device, rng, steps, chunk=20000,
              stats=None):
    """Euler-Maruyama with diffusion g == sigma_bridge (see module docstring)."""
    g = float(getattr(config, "g", 1.0))
    param = str(getattr(config, "dsbm_parameterization", "endpoint"))
    clamp = getattr(config, "dsbm_drift_clamp", U_STAR_CLAMP)
    outs = []
    n = x0_t.shape[0]
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        age = None
        if ages_t is not None and bool(model.cond_on_age):
            age = ages_t[lo:hi].to(torch.float32).unsqueeze(-1)
        outs.append(_dsbm_rollout(x0_t[lo:hi], model, steps, device, g, rng,
                                  age=age, parameterization=param,
                                  drift_clamp=clamp, stats=stats))
    return torch.cat(outs, dim=0)


def _heavy_sample(model, ckpt_path, config, X_eval, X_eval_ages, device):
    """Heavy sampling of the FORWARD process from a per-outer-iteration ckpt.

    Mirrors train_fm._heavy_sample_best_ckpt exactly -- same tiling of X_eval
    across the repeat axis, same per-setting repeat count, same 400 rollout
    steps, same explicitly seeded rng (config.seed + 12345) -- so the endpoints
    handed to scripts.reeval_baseline are produced under the identical protocol
    as every other baseline. The only difference is the rollout itself, which
    must honour the endpoint parameterisation.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])       # forward drift net
    model.eval()
    n_eval, d = X_eval.shape
    repeat = int(config.repeat_heavy)
    steps = int(config.num_rollout_steps_heavy)
    x0 = np.repeat(X_eval[:, None, :], repeat, axis=1).reshape(n_eval * repeat, d)
    x0_t = torch.tensor(x0, dtype=torch.float64, device=device)
    ages_t = None
    if model.cond_on_age:
        assert X_eval_ages is not None
        a = np.repeat(X_eval_ages[:, None], repeat, axis=1).reshape(n_eval * repeat)
        ages_t = torch.tensor(a, dtype=torch.float64, device=device)
    out = _simulate(model, x0_t, ages_t, config, device,
                    make_rng(int(config.seed) + 12345), steps)
    x_1 = out.cpu().numpy().reshape(n_eval, repeat, d)
    ysi = np.full((n_eval, repeat), -1, dtype=np.int64)
    return x_1, ysi


def _init_coupling_ot(X_train_t, Y_train_t, config, rng, device):
    """Iteration-1 coupling: minibatch Hungarian exact OT, chunked at
    batch_size -- the same estimator DSB-lite uses inside each training batch,
    just materialised over the whole train set so it can be *replaced* later."""
    n_x = X_train_t.shape[0]
    n_y = Y_train_t.shape[0]
    bs = int(config.batch_size)
    perm = rng.permutation(n_x)
    S_parts, E_parts, idx_parts = [], [], []
    for lo in range(0, n_x, bs):
        idx = perm[lo:lo + bs]
        if len(idx) == 0:
            continue
        x_c = X_train_t[torch.from_numpy(idx).to(device)]
        j = rng.choice(n_y, size=len(idx), replace=True)
        y_c = Y_train_t[torch.from_numpy(j).to(device)]
        x_p, y_p = _ot_minibatch_pair(x_c, y_c)
        S_parts.append(x_p)
        E_parts.append(y_p)
        idx_parts.append(idx[:x_p.shape[0]])
    return (torch.cat(S_parts, 0), torch.cat(E_parts, 0),
            np.concatenate(idx_parts, 0))


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------


def _cheap_metrics(gen_t, ref_np, rng_seed, repeat, batch, device, want_fd_mmd):
    """Same estimator family as the heavy path (SWD 256 projections, FD
    cov_eps=1e-6, pooled unbiased MMD) at a cheaper repeat count. Used for the
    trajectory table only; final numbers come from scripts/reeval_baseline.py."""
    gen = gen_t.detach().cpu().numpy()
    n_g, n_r = gen.shape[0], ref_np.shape[0]
    eb = min(batch, n_g, n_r)
    swds, fds = [], []
    for r in range(repeat):
        rr = make_rng(rng_seed + r * 7919)
        ig = rr.choice(n_g, eb, replace=False).astype(np.int64)
        ir = rr.choice(n_r, eb, replace=False).astype(np.int64)
        a, b = gen[ig], ref_np[ir]
        swds.append(float(sliced_wasserstein_distance(a, b, num_projections=256, rng=rr)))
        if want_fd_mmd:
            fds.append(float(frechet_distance(a, b, cov_eps=1e-6)))
    swd = np.asarray([v for v in swds if np.isfinite(v)], dtype=np.float64)
    out = {
        "swd": float(swd.mean()) if swd.size else float("nan"),
        "swd_std": float(swd.std(ddof=1)) if swd.size > 1 else 0.0,
    }
    if want_fd_mmd:
        f = np.asarray([v for v in fds if np.isfinite(v)], dtype=np.float64)
        out["fd"] = float(f.mean()) if f.size else float("nan")
        out["mmd"] = float(maximum_mean_discrepancy(
            gen, ref_np, rng=make_rng(rng_seed + 9001)))
    return out


TRAJ_FIELDS = [
    "outer_iter",
    "fwd_swd", "fwd_swd_std", "fwd_fd", "fwd_mmd",
    "bwd_swd", "bwd_swd_std",
    "train_swd_cheap",
    "fwd_loss", "bwd_loss",
    "grad_updates_cum", "wall_clock_s_cum",
    "coupling_source_swd", "coupling_mean_pair_dist",
    "coupling_checksum", "coupling_is_ot_init", "n_pairs",
    "clamp_frac_fwd",
]


def _write_traj(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAJ_FIELDS})


def _write_results(path: Path, rows):
    """`results.csv` for interface compatibility; holds the TRAIN-side
    cheap SWD of each outer iteration."""
    fields = ["checkpoint", "sampling_rho", "swd_mean", "swd_std",
              "n_eval", "n_repeat", "note"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main IMF loop
# ---------------------------------------------------------------------------


def _make_net(d, config, device):
    return ControllerNet(
        d=d, hidden_dim=int(config.hidden_dim),
        num_layers=int(config.num_layers),
        cond_on_y_star=False,
        cond_on_age=bool(getattr(config, "cond_on_age", False)),
        age_emb_dim=int(getattr(config, "age_emb_dim", 32)),
        time_emb_dim=int(config.time_emb_dim),
        dropout=float(config.dropout),
        activation=str(getattr(config, "activation", "silu")),
        time_embedding_type=str(getattr(config, "time_embedding_type", "sinusoidal")),
        block_type=str(getattr(config, "block_type", "residual")),
    ).to(device)


def _resolve_workdir(config) -> Path:
    if hasattr(config, "make_train_workdir_name") and callable(
            config.make_train_workdir_name):
        name = config.make_train_workdir_name(config)
    else:
        name = config.make_workdir_name(config)
    return config.workdir_root / name


def train_dsbm(config) -> Path:
    set_global_seed(config.seed)
    rng = make_rng(config.seed)

    X_train, Y_train, X_eval, Y_eval, X_train_ages, X_eval_ages = _load_data(config)
    d = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_age = bool(getattr(config, "cond_on_age", False))
    if cond_age and X_train_ages is None:
        raise RuntimeError("cond_on_age=True but the npz carries no ages.")

    n_outer = int(getattr(config, "dsbm_outer_iters", 20))
    inner_epochs = int(getattr(config, "dsbm_inner_epochs", 30))
    coupling_steps = int(getattr(config, "dsbm_coupling_steps",
                                 config.num_rollout_steps_heavy))
    traj_steps = int(getattr(config, "dsbm_traj_steps", 200))
    traj_repeat = int(getattr(config, "dsbm_traj_repeat", 10))
    sigma = float(getattr(config, "sigma_bridge", 0.5))
    if abs(float(getattr(config, "g", 1.0)) - sigma) > 1e-12:
        raise RuntimeError(
            f"DSBM requires the simulated diffusion g ({config.g}) to equal "
            f"sigma_bridge ({sigma}); otherwise the reciprocal projection "
            "simulates a different path measure than the Markovian projection "
            "regressed and IMF has no fixed point. Set g = sigma_bridge.")

    workdir = _resolve_workdir(config)
    ensure_dir(workdir)
    ensure_dir(workdir / "checkpoints")
    save_config_snapshot(config, workdir)
    logger = TrainLogger(workdir / "train_log.csv")

    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float64, device=device)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float64, device=device)
    Y_eval_t = torch.tensor(Y_eval, dtype=torch.float64, device=device)
    X_train_ages_t = (torch.tensor(X_train_ages, dtype=torch.float64, device=device)
                      if X_train_ages is not None else None)
    X_eval_ages_t = (torch.tensor(X_eval_ages, dtype=torch.float64, device=device)
                     if X_eval_ages is not None else None)

    fwd = _make_net(d, config, device)
    bwd = _make_net(d, config, device)
    opt_kw = dict(lr=float(config.lr),
                  weight_decay=float(getattr(config, "weight_decay", 0.0)),
                  beta1=float(getattr(config, "adam_beta1", 0.9)))
    opt_f = make_optimizer(fwd, **opt_kw)
    opt_b = make_optimizer(bwd, **opt_kw)

    # ---- iteration-1 coupling = DSB-lite's minibatch Hungarian OT ----------
    S, E, src_idx = _init_coupling_ot(X_train_t, Y_train_t, config, rng, device)
    A = (X_train_ages_t[torch.from_numpy(src_idx).to(device)]
         if cond_age else None)
    coupling_is_ot = True

    print(f"[dsbm] workdir={workdir}")
    print(f"[dsbm] n_outer={n_outer} inner_epochs={inner_epochs} "
          f"sigma={sigma} g={config.g} coupling_steps={coupling_steps} "
          f"cond_on_age={cond_age}")
    print(f"[dsbm] X_train={X_train.shape} Y_train={Y_train.shape} "
          f"X_eval={X_eval.shape} Y_eval={Y_eval.shape}")

    traj_rows, res_rows = [], []
    gs_f = gs_b = 0
    t0 = time.time()

    for it in range(1, n_outer + 1):
        # ---- Markovian projection, FORWARD: fit on the current coupling ----
        fwd_loss, gs_f = _fit_direction(
            fwd, opt_f, S, E, A, config, rng, device, inner_epochs,
            logger, "fwd", gs_f)

        # ---- reciprocal projection: simulate FORWARD from the true source --
        cl_stats = {}
        y_hat = _simulate(fwd, X_train_t, X_train_ages_t, config, device,
                          make_rng(config.seed + 1000 * it + 1), coupling_steps,
                          stats=cl_stats)
        clamp_frac = (cl_stats.get("clamped", 0) / cl_stats["total"]
                      if cl_stats.get("total") else 0.0)

        # ---- Markovian projection, BACKWARD: bridge (y_hat -> X_train) -----
        # Bridge symmetry: calling the same routine with the endpoints swapped
        # fits the time-reversed drift in the s = 1 - t variable.
        # y_hat and X_train are both in X_train's natural row order, so the age
        # vector here is X_train_ages_t directly (NOT the coupling's `A`, which
        # is permuted at iteration 1 and has n_y rows from iteration 2 on).
        bwd_loss, gs_b = _fit_direction(
            bwd, opt_b, y_hat, X_train_t,
            X_train_ages_t if cond_age else None,
            config, rng, device, inner_epochs, logger, "bwd", gs_b)

        # ---- reciprocal projection: simulate BACKWARD from the true target -
        # The generated x0_hat is a synthetic source subject; for the AGE
        # variant it carries an age drawn from the source age distribution,
        # keeping the repo convention that conditioning is on the SOURCE age.
        n_y = Y_train_t.shape[0]
        if cond_age:
            a_draw = rng.choice(X_train_ages, size=n_y, replace=True)
            A_new = torch.tensor(a_draw, dtype=torch.float64, device=device)
        else:
            A_new = None
        x0_hat = _simulate(bwd, Y_train_t, A_new, config, device,
                           make_rng(config.seed + 1000 * it + 2), coupling_steps)

        # ---- diagnostics BEFORE swapping in the new coupling ---------------
        wall = time.time() - t0
        fwd_eval = _simulate(fwd, X_eval_t, X_eval_ages_t, config, device,
                             make_rng(config.seed + 7000 + it), traj_steps)
        bwd_eval = _simulate(bwd, Y_eval_t,
                             (torch.tensor(
                                 rng.choice(X_train_ages, size=Y_eval.shape[0],
                                            replace=True),
                                 dtype=torch.float64, device=device)
                              if cond_age else None),
                             config, device,
                             make_rng(config.seed + 8000 + it), traj_steps)
        train_gen = _simulate(fwd, X_train_t, X_train_ages_t, config, device,
                              make_rng(config.seed + 9000 + it), traj_steps)

        m_f = _cheap_metrics(fwd_eval, Y_eval, 42, traj_repeat, 100, device, True)
        m_b = _cheap_metrics(bwd_eval, X_eval, 42, traj_repeat, 100, device, False)
        m_t = _cheap_metrics(train_gen, Y_train, 42, traj_repeat, 100, device, False)

        # ---- coupling-refresh evidence ----------------------------------------
        # These describe the coupling the forward net was JUST fit on, i.e. the
        # OT one at iteration 1 and a regenerated one from iteration 2 onward.
        # `coupling_checksum` changing across rows is the direct proof that the
        # reciprocal projection is running.
        # SWD in this repo requires equal sample sizes, and |X_train| != |Y_train|,
        # so use the same balanced-subsample estimator as everywhere else.
        src_swd = _cheap_metrics(x0_hat, X_train, 1234, traj_repeat, 100,
                                 device, False)["swd"]
        pair_dist = float(torch.linalg.norm(E - S, dim=-1).mean().item())
        checksum = float(S.sum().item())

        row = {
            "outer_iter": it,
            "fwd_swd": m_f["swd"], "fwd_swd_std": m_f["swd_std"],
            "fwd_fd": m_f["fd"], "fwd_mmd": m_f["mmd"],
            "bwd_swd": m_b["swd"], "bwd_swd_std": m_b["swd_std"],
            "train_swd_cheap": m_t["swd"],
            "fwd_loss": fwd_loss, "bwd_loss": bwd_loss,
            "grad_updates_cum": gs_f + gs_b,
            "wall_clock_s_cum": round(wall, 1),
            "coupling_source_swd": src_swd,
            "coupling_mean_pair_dist": pair_dist,
            "coupling_checksum": checksum,
            "coupling_is_ot_init": int(coupling_is_ot),
            "n_pairs": int(S.shape[0]),
            "clamp_frac_fwd": clamp_frac,
        }
        traj_rows.append(row)
        _write_traj(workdir / "dsbm_trajectory.csv", traj_rows)

        label = f"iter{it:03d}"
        torch.save({
            "model_state_dict": fwd.state_dict(),   # FORWARD -> selector/transfer
            "bwd_state_dict": bwd.state_dict(),
            "outer_iter": it,
            "fwd_loss": fwd_loss, "bwd_loss": bwd_loss,
            "grad_updates": gs_f + gs_b,
        }, workdir / "checkpoints" / f"ckpt_{label}.pth")
        res_rows.append({
            "checkpoint": label, "sampling_rho": 0.0,
            "swd_mean": m_t["swd"], "swd_std": m_t["swd_std"],
            "n_eval": 100, "n_repeat": traj_repeat,
            "note": "TRAIN-side cheap SWD (leak-free)",
        })
        _write_results(workdir / "results.csv", res_rows)

        print(f"[dsbm] iter {it:3d}/{n_outer}  fwd_swd={m_f['swd']:.4f} "
              f"fwd_fd={m_f['fd']:.1f} fwd_mmd={m_f['mmd']:.3e}  "
              f"bwd_swd={m_b['swd']:.4f}  train_swd={m_t['swd']:.4f}  "
              f"src_swd={src_swd:.4f} pair={pair_dist:.3f} clamp={clamp_frac:.2%} "
              f"upd={gs_f + gs_b} {wall:.0f}s", flush=True)

        # ---- swap in the regenerated coupling for the next outer iteration -
        prev_checksum = checksum
        S, E, A = x0_hat.detach(), Y_train_t, A_new
        coupling_is_ot = False
        new_checksum = float(S.sum().item())
        if abs(new_checksum - prev_checksum) < 1e-12:
            raise RuntimeError(
                f"Outer iteration {it}: the endpoint coupling did not change "
                f"(checksum {prev_checksum!r} -> {new_checksum!r}). The "
                "reciprocal projection is not running, which would make this "
                "DSB-lite repeated N times, not DSBM. Refusing to continue.")

    logger.close()
    (workdir / "dsbm_summary.txt").write_text(
        f"DSBM outer_iters={n_outer} inner_epochs={inner_epochs} sigma={sigma} "
        f"g={config.g} coupling_steps={coupling_steps} cond_on_age={cond_age}\n"
        f"total_grad_updates={gs_f + gs_b} (fwd={gs_f} bwd={gs_b})\n"
        f"wall_clock_s={time.time() - t0:.1f}\n")
    print(f"DONE DSBM: {workdir}")
    return workdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_normalization", action="store_true")
    ap.add_argument("--no_normalization", action="store_true")
    ap.add_argument("--outer_iters", type=int, default=None)
    ap.add_argument("--inner_epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    add_setting_args(ap)
    args = ap.parse_args()

    config = load_config(args.config)
    apply_setting_args(config, args)
    if args.use_normalization:
        config.use_normalization = True
        config.suffix = "_normalization"
    if args.no_normalization:
        config.use_normalization = False
        config.suffix = "_raw"
    if args.outer_iters is not None:
        config.dsbm_outer_iters = int(args.outer_iters)
    if args.inner_epochs is not None:
        config.dsbm_inner_epochs = int(args.inner_epochs)
    if args.seed is not None:
        config.seed = int(args.seed)

    train_dsbm(config)


if __name__ == "__main__":
    main()
