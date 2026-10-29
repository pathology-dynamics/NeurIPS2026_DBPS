"""Section 5.2: recovery of empirical disease-effect ranks.

Model effect: per ROI, the mean endpoint shift X_1 - X_0 of DBPS rollouts
over the held-out sources (repeat r draws eval_batch_size sources and
anchors with a generator seeded by seed + 7919 r). Empirical effect: mean
impaired minus mean CN over the held-out pools. Reports Spearman
correlations of the two effects (84 ROIs) and of each effect with Braak
priority (72 Braak-mapped ROIs; earlier stage = higher priority).

usage:
  python -m scripts.rank_alignment --config configs/methods/dbps.py --gamma 100 --rho 0.1 \
      --checkpoint <run>/checkpoints/ckpt_epoch150.pth --repeat 250 --out results/rank_adni
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from controls.db_control import compute_u_anchor
from dynamics.rollout import rollout
from dynamics.transition_computation import TransitionComputation
from evaluation.braak import braak_priority, braak_stage
from sampling.sample_db import build_controller, load_pool
from utils.config_utils import load_config, resolve_data_path
from utils.seed import make_rng, set_global_seed


def spearman(x, y):
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n_roi": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repeat", type=int, default=None, help="default: config.repeat_heavy")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    config.gamma = int(args.gamma) if float(args.gamma).is_integer() else float(args.gamma)
    if args.data is not None:
        config.processed_data_path = Path(args.data)
    seed, steps = int(config.reeval_seed), int(config.num_rollout_steps_heavy)
    repeat, rho = int(args.repeat or config.repeat_heavy), float(args.rho)

    X, Y_ref, Y_train, L, ages = load_pool(config, "eval")
    roi_names = [str(r) for r in np.load(resolve_data_path(config), allow_pickle=True)["roi_names"]]
    stages = np.array([braak_stage(r) for r in roi_names], dtype=np.int64)
    braak = stages > 0
    priority = np.array([braak_priority(int(s)) for s in stages])
    n_x, d = X.shape
    eval_batch = min(int(config.eval_batch_size), n_x, Y_ref.shape[0])

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_controller(config, d, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    tr = TransitionComputation(L=L, beta=config.beta, cov_regularization=config.cov_regularization, device=device)

    deltas = []
    for r in range(repeat):
        rng = make_rng(seed + 7919 * r)
        idx_x = rng.choice(n_x, eval_batch, replace=False).astype(np.int64)
        y_idx = rng.choice(Y_train.shape[0], size=eval_batch).astype(np.int64)
        age_t = (torch.tensor(ages[idx_x], dtype=torch.float32, device=device).unsqueeze(-1)
                 if bool(getattr(config, "cond_on_age", False)) else None)

        def controller_fn(x_t, t, y_star, candidates, age_t=age_t):
            ua = compute_u_anchor(x_t, t, y_star, gamma=config.gamma, beta=config.beta, g=config.g, transition=tr)
            with torch.no_grad():
                un = model(x_t.to(torch.float32), t.to(torch.float32), age=age_t).to(dtype=torch.float64)
            return (1.0 - rho) * ua + rho * un

        out = rollout(x_0=torch.tensor(X[idx_x], dtype=torch.float64, device=device), controller_fn=controller_fn,
                      y_star=torch.tensor(Y_train[y_idx], dtype=torch.float64, device=device), candidates=None,
                      transition=tr, num_steps=steps, g=float(config.g), use_sde=bool(config.use_sde),
                      save_trajectories=False, save_controls=False, rng=rng)
        deltas.append(out["x_1"].detach().cpu().numpy().astype(np.float64) - X[idx_x])
        if (r + 1) % 10 == 0 or r + 1 == repeat:
            print(f"rollout {r + 1}/{repeat}", flush=True)

    model_effect = np.concatenate(deltas).mean(axis=0)
    real_effect = Y_ref.mean(axis=0) - X.mean(axis=0)
    summary = {
        "model_vs_empirical_all84": spearman(model_effect, real_effect),
        "model_vs_empirical_braak72": spearman(model_effect[braak], real_effect[braak]),
        "model_vs_braak_priority": spearman(model_effect[braak], priority[braak]),
        "empirical_vs_braak_priority": spearman(real_effect[braak], priority[braak]),
        "repeat": repeat, "eval_batch": eval_batch, "gamma": config.gamma, "rho": rho,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "effects.npz", model_effect=model_effect, real_effect=real_effect, roi_names=np.array(roi_names))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
