"""Table 3: endpoint-anchor dispersion and control energy of a DBPS checkpoint
versus the closed-form endpoint-only anchor (UniDB, rho = 0).

Objective convention: Anchor = gamma/2 ||X_1 - y*||^2, Energy = 1/2 int ||u_t||^2 dt.
For UniDB both terms are computed in closed form at t = 0 for the sampled
(x_0, y*) pairs (total value V = -eta log psi; the anchor term under the
tilted Gaussian endpoint law; energy = total - anchor). For DBPS the
checkpoint is replayed with Euler-Maruyama from the same (x_0, y*) pairs
and both terms are accumulated empirically. Repeats follow the evaluation
protocol (generator seeded with seed + 7919 r).

usage:
  python -m scripts.compute_control_costs --config configs/methods/dbps.py --gamma 100 --rho 0.1 \
      --checkpoint outputs/adni/dbps_gamma100_normalization/checkpoints/ckpt_epoch150.pth \
      --repeat 250 --out tables/table3_adni.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from controls.db_control import _solve_A_inv_v, compute_log_psi, compute_u_anchor
from dynamics.transition_computation import TransitionComputation
from sampling.sample_db import build_controller, load_pool
from utils.config_utils import load_config
from utils.seed import make_rng, set_global_seed


def closed_form_anchor_costs(x0, y_star, config, transition):
    """Closed-form anchor/energy decomposition of the analytic endpoint bridge at t = 0."""
    device = transition.U.device
    x0_t = torch.tensor(x0, dtype=torch.float64, device=device)
    y_t = torch.tensor(y_star, dtype=torch.float64, device=device)
    t0 = torch.zeros((x0_t.shape[0],), dtype=torch.float64, device=device)
    eta = float(config.beta) / (float(config.g) ** 2)
    gamma = float(config.gamma)
    kappa = gamma / eta

    log_psi = compute_log_psi(x0_t, t0, y_t, gamma=gamma, beta=float(config.beta),
                              g=float(config.g), transition=transition)
    total = (-eta * log_psi).detach().cpu().numpy()

    m = transition.M_apply(torch.ones((), dtype=torch.float64, device=device), x0_t)
    a_inv_residual = _solve_A_inv_v(transition, t0, kappa, y_t - m)
    sq_mean_offset = torch.sum(a_inv_residual * a_inv_residual, dim=1)
    d_i = transition._d_eigvals(torch.ones((), dtype=torch.float64, device=device))
    posterior_trace = torch.sum(d_i / (1.0 + kappa * d_i))
    terminal = (0.5 * gamma * (sq_mean_offset + posterior_trace)).detach().cpu().numpy()
    return terminal, total - terminal, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repeat", type=int, default=None, help="default: config.repeat_heavy")
    ap.add_argument("--data", default=None, help="override processed_data.npz (evaluation pool)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    config.gamma = float(args.gamma)
    if args.data is not None:
        config.processed_data_path = Path(args.data)
    repeat = int(args.repeat or config.repeat_heavy)
    steps = int(config.num_rollout_steps_heavy)
    seed = int(config.reeval_seed)
    gamma, rho = float(args.gamma), float(args.rho)

    X_eval, Y_eval, Y_train, L, ages = load_pool(config, "eval")
    n_x, d = X_eval.shape
    n_train = Y_train.shape[0]
    eval_batch = min(int(config.eval_batch_size), n_x, Y_eval.shape[0])

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_controller(config, d, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    transition = TransitionComputation(L=L, beta=float(config.beta),
                                       cov_regularization=float(config.cov_regularization), device=device)
    L_full = (transition.U * transition.sigma.unsqueeze(0)) @ transition.U.T
    L_full = 0.5 * (L_full + L_full.T)
    dt = 1.0 / float(steps)
    sqrt_beta_dt = float(math.sqrt(float(config.beta) * dt))
    cond_on_age = bool(getattr(config, "cond_on_age", False))

    rows = []
    for r in range(repeat):
        rng_r = make_rng(seed + r * 7919)
        idx_x = rng_r.choice(n_x, eval_batch, replace=False).astype(np.int64)
        rng_r.choice(Y_eval.shape[0], eval_batch, replace=False)      # target batch (keeps the RNG stream aligned)
        idx_anchor = rng_r.choice(n_train, size=eval_batch).astype(np.int64)
        x0, y_star = X_eval[idx_x], Y_train[idx_anchor]

        cf_terminal, cf_running, cf_total = closed_form_anchor_costs(x0, y_star, config, transition)

        x = torch.tensor(x0, dtype=torch.float64, device=device)
        y_star_t = torch.tensor(y_star, dtype=torch.float64, device=device)
        age_t = (torch.tensor(ages[idx_x], dtype=torch.float32, device=device).unsqueeze(-1)
                 if cond_on_age else None)
        energy = 0.0
        with torch.no_grad():
            for k in range(steps):
                t_batch = torch.full((eval_batch,), k * dt, dtype=torch.float64, device=device)
                u_anchor = compute_u_anchor(x, t_batch, y_star_t, gamma=gamma, beta=float(config.beta),
                                            g=float(config.g), transition=transition)
                u_theta = model(x.to(torch.float32), t_batch.to(torch.float32), age=age_t).to(torch.float64)
                u = (1.0 - rho) * u_anchor + rho * u_theta
                energy += float(torch.mean(torch.sum(u * u, dim=1)).item()) * dt
                drift = -float(config.beta) * (x @ L_full.T) + float(config.g) * u
                z = torch.from_numpy(rng_r.standard_normal((eval_batch, d)).astype(np.float64)).to(device)
                x = x + drift * dt + sqrt_beta_dt * z
        anchor = 0.5 * gamma * float(torch.mean(torch.sum((x - y_star_t) ** 2, dim=1)).item())
        rows.append({"unidb_anchor": float(np.mean(cf_terminal)), "unidb_energy": float(np.mean(cf_running)),
                     "unidb_sum": float(np.mean(cf_total)), "dbps_anchor": anchor,
                     "dbps_energy": 0.5 * energy, "dbps_sum": anchor + 0.5 * energy})
        if (r + 1) % 10 == 0 or r + 1 == repeat:
            print(f"finished {r + 1}/{repeat}", flush=True)

    summary = {"gamma": gamma, "rho": rho, "checkpoint": str(args.checkpoint), "repeat": repeat,
               "n_eval": eval_batch, "steps": steps}
    for key in rows[0]:
        v = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(v.mean())
        summary[f"{key}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    for m in ("unidb", "dbps"):
        print(f"{m:5s} anchor {summary[f'{m}_anchor_mean']:.1f} ± {summary[f'{m}_anchor_std']:.1f}  "
              f"energy {summary[f'{m}_energy_mean']:.1f} ± {summary[f'{m}_energy_std']:.1f}  "
              f"sum {summary[f'{m}_sum_mean']:.1f} ± {summary[f'{m}_sum_std']:.1f}")


if __name__ == "__main__":
    main()
