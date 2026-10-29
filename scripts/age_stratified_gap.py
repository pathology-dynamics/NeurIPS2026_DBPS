"""Table 5: age-stratified DBPS transport signatures.

For each evaluation repeat (generator seeded with seed + 7919 r) the same
source batch, anchors and SDE noise are rolled out twice: with the full DBPS
controller and with u = 0. The endpoint gap (full - uncontrolled) is
recorded for every sampled source; records are split into young / old at the
median of the recorded source ages (records without an age are excluded),
and each ROI's young - old difference is reported with a Welch test and
Benjamini-Hochberg q-values over the 84 ROIs. The early-minus-late summary
is the mean gap over Braak stages I-III minus the mean over stages V-VI.

usage:
  python -m scripts.age_stratified_gap --config configs/methods/dbps.py --gamma 100 --rho 0.1 \
      --checkpoint <run>/checkpoints/ckpt_epoch150.pth --repeat 250 --out results/table5_adni
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from controls.db_control import compute_u_anchor
from dynamics.rollout import rollout
from dynamics.transition_computation import TransitionComputation
from evaluation.braak import BRAAK_LABELS, bh_qvalues, braak_stage
from sampling.sample_db import build_controller, load_pool
from utils.config_utils import load_config, resolve_data_path
from utils.seed import make_rng, set_global_seed


def stage_means(values: np.ndarray, stages: np.ndarray) -> dict:
    means = {s: float(values[stages == s].mean()) for s in (2, 3, 4, 5, 6)}
    means["early_minus_late"] = float(np.mean([means[2], means[3]]) - np.mean([means[5], means[6]]))
    return means


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
    repeat = int(args.repeat or config.repeat_heavy)
    rho = float(args.rho)

    X, Y_ref, Y_train, L, ages_imputed = load_pool(config, "eval")
    z = np.load(resolve_data_path(config), allow_pickle=True)
    ages_raw = np.concatenate([z["X_val_ages"], z["X_test_ages"]]).astype(np.float64)
    roi_names = [str(r) for r in z["roi_names"]]
    stages = np.array([braak_stage(r) for r in roi_names], dtype=np.int64)
    n_x, d = X.shape
    eval_batch = min(int(config.eval_batch_size), n_x, Y_ref.shape[0])

    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_controller(config, d, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    tr = TransitionComputation(L=L, beta=config.beta, cov_regularization=config.cov_regularization, device=device)

    def make_controller(full: bool, age_t):
        def controller_fn(x_t, t, y_star, candidates):
            if not full:
                return torch.zeros_like(x_t, dtype=torch.float64, device=x_t.device)
            ua = compute_u_anchor(x_t, t, y_star, gamma=config.gamma, beta=config.beta, g=config.g, transition=tr)
            with torch.no_grad():
                un = model(x_t.to(torch.float32), t.to(torch.float32), age=age_t).to(dtype=torch.float64)
            return (1.0 - rho) * ua + rho * un
        return controller_fn

    gaps, rec_ages, rec_idx = [], [], []
    for r in range(repeat):
        rng = make_rng(seed + 7919 * r)
        idx_x = rng.choice(n_x, eval_batch, replace=False).astype(np.int64)
        rng.choice(Y_ref.shape[0], eval_batch, replace=False)
        y_idx = rng.choice(Y_train.shape[0], size=eval_batch).astype(np.int64)
        age_t = (torch.tensor(ages_imputed[idx_x], dtype=torch.float32, device=device).unsqueeze(-1)
                 if bool(getattr(config, "cond_on_age", False)) else None)
        ends = {}
        for full in (False, True):
            rng_mode = make_rng(seed + 7919 * r)        # identical noise for both rollouts
            rng_mode.choice(n_x, eval_batch, replace=False)
            rng_mode.choice(Y_ref.shape[0], eval_batch, replace=False)
            rng_mode.choice(Y_train.shape[0], size=eval_batch)
            out = rollout(x_0=torch.tensor(X[idx_x], dtype=torch.float64, device=device),
                          controller_fn=make_controller(full, age_t),
                          y_star=torch.tensor(Y_train[y_idx], dtype=torch.float64, device=device),
                          candidates=None, transition=tr, num_steps=steps, g=float(config.g),
                          use_sde=bool(config.use_sde), save_trajectories=False, save_controls=False, rng=rng_mode)
            ends[full] = out["x_1"].detach().cpu().numpy().astype(np.float64)
        gaps.append(ends[True] - ends[False])
        rec_ages.append(ages_raw[idx_x])
        rec_idx.append(idx_x)
        if (r + 1) % 10 == 0 or r + 1 == repeat:
            print(f"rollout {r + 1}/{repeat}", flush=True)

    gap, rec_ages, rec_idx = np.concatenate(gaps), np.concatenate(rec_ages), np.concatenate(rec_idx)
    finite = np.isfinite(rec_ages)
    cutoff = float(np.median(rec_ages[finite]))
    young, old = finite & (rec_ages < cutoff), finite & (rec_ages >= cutoff)
    y_mean, o_mean = gap[young].mean(axis=0), gap[old].mean(axis=0)
    _, p = stats.ttest_ind(gap[young], gap[old], axis=0, equal_var=False)
    q = bh_qvalues(p)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-(y_mean - o_mean))
    with open(out / "young_vs_old_roi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi", "braak_stage", "braak_label", "young_gap", "old_gap", "young_minus_old", "p", "q"])
        for i in order:
            w.writerow([roi_names[i], int(stages[i]), BRAAK_LABELS.get(int(stages[i]), ""), float(y_mean[i]),
                        float(o_mean[i]), float(y_mean[i] - o_mean[i]), float(p[i]), float(q[i])])
    summary = {
        "age_cutoff": cutoff, "n_records_young": int(young.sum()), "n_records_old": int(old.sum()),
        "n_unique_scans_young": int(len(set(rec_idx[young]))), "n_unique_scans_old": int(len(set(rec_idx[old]))),
        "young_stage_means": stage_means(y_mean, stages), "old_stage_means": stage_means(o_mean, stages),
        "repeat": repeat, "eval_batch": eval_batch, "gamma": config.gamma, "rho": rho,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
