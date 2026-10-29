"""Braak alignment of a trained controller (Section 5.1, Figure 2, Appendix I).

For each held-out source subject, compute a per-ROI estimand, take the
young-vs-old Cohen's d_k (median-age split), and correlate |d_k| with the
Braak stage over the 72 Braak-mapped ROIs (Spearman).

Estimands (--estimand):
  dudage          d u_theta,k / d age at time t (--t, one or more values)
  endpoint_shift  mean X_1 - X_0 over --n_rollouts DBPS rollouts at sampling weight --rho

Ages (--ages): "imputed" uses median-imputed ages for subjects without age;
"real" keeps only subjects with a recorded age.
--cohort restricts a pooled evaluation set to one cohort (diagnosis prefix).
--drop excludes ROIs whose name contains any of the given substrings from
the Spearman correlation (e.g. --drop entorhinal hippocampus).

The same script evaluates learned baselines (pass their config and
checkpoint; the estimand is computed on their network) and models trained
on randomized graphs (pass their processed data with --data).

usage:
  python -m scripts.braak_alignment --config configs/methods/dbps.py --gamma 100 \
      --checkpoint <run>/checkpoints/ckpt_epoch150.pth --estimand dudage --ages imputed \
      --t 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --out results/braak_adni
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dynamics.transition_computation import TransitionComputation
from evaluation.braak import (
    braak_stage, bh_qvalues, dudage, endpoint_shift, spearman_braak, two_sample_d,
)
from sampling.sample_db import build_controller
from utils.config_utils import load_config, resolve_data_path
from utils.seed import set_global_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None, help="sampling weight (endpoint_shift)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None, help="override processed_data.npz")
    ap.add_argument("--estimand", choices=("dudage", "endpoint_shift"), default="dudage")
    ap.add_argument("--t", type=float, nargs="+", default=[0.5])
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--ages", choices=("imputed", "real"), default="imputed")
    ap.add_argument("--cohort", default=None, help="keep only evaluation subjects of this cohort (e.g. adni)")
    ap.add_argument("--drop", nargs="*", default=[], help="ROI-name substrings excluded from the correlation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    if args.gamma is not None:
        config.gamma = int(args.gamma) if float(args.gamma).is_integer() else float(args.gamma)
    if args.data is not None:
        config.processed_data_path = Path(args.data)
    seed = int(config.reeval_seed)
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(resolve_data_path(config), allow_pickle=True)
    X = np.concatenate([z["X_val"], z["X_test"]], axis=0).astype(np.float64)
    ages = np.concatenate([z["X_val_ages"], z["X_test_ages"]], axis=0).astype(np.float64)
    if args.cohort is not None:
        diag = np.array([str(v).lower() for v in np.concatenate([z["X_val_diagnosis"], z["X_test_diagnosis"]])])
        keep = np.array([v.startswith(args.cohort + ":") or v.startswith(args.cohort) for v in diag])
        X, ages = X[keep], ages[keep]
    if args.ages == "real":
        keep = np.isfinite(ages)
        X, ages = X[keep], ages[keep]
    else:
        finite = ages[np.isfinite(ages)]
        ages = np.where(np.isnan(ages), float(np.median(finite)) if finite.size else 70.0, ages)

    roi_names = [str(r) for r in z["roi_names"]]
    stages = np.array([braak_stage(r) for r in roi_names], dtype=np.int64)
    keep_roi = np.array([not any(s in r.lower() for s in args.drop) for r in roi_names])

    model = build_controller(config, X.shape[1], device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def analyse(values, label):
        res = two_sample_d(values, ages)
        q = bh_qvalues(res["pvalue"])
        sp = spearman_braak(np.abs(res["d"]), stages, keep_roi)
        row = {"label": label, "spearman_rho": sp["rho"], "p_value": sp["p"], "n_roi": sp["n_roi"],
               "n_subjects": int(X.shape[0]), "n_young": res["n_young"], "n_old": res["n_old"],
               "median_age": res["median_age"],
               "fdr_sig_84": int((q <= 0.05).sum()),
               "fdr_sig_braak": int(((q <= 0.05) & (stages > 0)).sum())}
        rows.append(row)
        with open(out / f"per_roi_{label}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["roi", "braak_stage", "cohens_d", "abs_d", "pvalue", "qvalue_all84"])
            for i, r in enumerate(roi_names):
                w.writerow([r, int(stages[i]), float(res["d"][i]), float(abs(res["d"][i])),
                            float(res["pvalue"][i]), float(q[i])])
        print(f"{label}: rho={sp['rho']:+.4f} p={sp['p']:.2e} (n_roi={sp['n_roi']}, n={X.shape[0]}, "
              f"FDR {row['fdr_sig_braak']}/{int((stages > 0).sum())} Braak ROIs)", flush=True)

    if args.estimand == "dudage":
        for t in args.t:
            analyse(dudage(model, X, ages, t, device), f"dudage_t{t:g}")
    else:
        if args.rho is None:
            raise SystemExit("--rho is required for endpoint_shift")
        L = np.asarray(z["L"], dtype=np.float64)
        tr = TransitionComputation(L=L, beta=config.beta, cov_regularization=config.cov_regularization,
                                   device=device)
        delta = endpoint_shift(model, X, ages, np.asarray(z["Y_train"], dtype=np.float64), tr, config,
                               args.rho, args.n_rollouts, device, seed=seed)
        analyse(delta, "endpoint_shift")

    (out / "summary.json").write_text(json.dumps({"args": vars(args), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
