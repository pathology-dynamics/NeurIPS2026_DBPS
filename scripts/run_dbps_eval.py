"""Checkpoint selection and final evaluation for one DBPS training run.

1. Every saved checkpoint is evaluated at every rho on the TRAINING split
   (config.select_steps rollout steps x config.select_repeat repeats).
2. For each rho, the checkpoint with the lowest training-split SWD
   (ties broken by SWD standard deviation) is selected.
3. The selected checkpoint is evaluated on the held-out pool
   (config.num_rollout_steps_heavy steps x config.repeat_heavy repeats).

Outputs in <run_dir>/eval/: selection_train.csv, final_eval.csv and the
per-repeat SWD of every final evaluation in per_repeat.npz (keys rho<value>),
used by scripts.run_paired_tests. With --tag the final files are
final_eval_<tag>.csv / per_repeat_<tag>.npz, e.g. for the Union -> ADNI
evaluation of a Union-trained run, which reuses the training-split selection.

usage:
  python -m scripts.run_dbps_eval --config configs/methods/dbps.py --gamma 100 \
      --run_dir outputs/adni/dbps_gamma100_normalization --rho 0 0.1 0.2 0.3 0.4 0.5 0.8 1.0
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dynamics.transition_computation import TransitionComputation
from sampling.sample_db import FIELDS, evaluate_checkpoint, load_pool
from utils.config_utils import load_config

CKPT_RE = re.compile(r"^ckpt_epoch(\d+)\.pth$")


def checkpoints(run_dir: Path) -> list[Path]:
    found = [p for p in (run_dir / "checkpoints").iterdir() if CKPT_RE.match(p.name)]
    return sorted(found, key=lambda p: int(CKPT_RE.match(p.name).group(1)))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--rho", type=float, nargs="+", default=None,
                    help="default: config.rho_values")
    ap.add_argument("--data", default=None,
                    help="processed_data.npz used for training-split selection and, unless --eval_data is given, evaluation")
    ap.add_argument("--eval_data", default=None,
                    help="processed_data.npz for the held-out pool (default: training data)")
    ap.add_argument("--eval_repeat", type=int, default=None, help="default: config.repeat_heavy")
    ap.add_argument("--tag", default=None, help="suffix of the final evaluation file")
    ap.add_argument("--reuse_selection", action="store_true",
                    help="reuse an existing selection_train.csv instead of recomputing it")
    args = ap.parse_args()

    config = load_config(args.config)
    config.gamma = int(args.gamma) if float(args.gamma).is_integer() else float(args.gamma)
    if args.data is not None:
        config.processed_data_path = Path(args.data)
    rhos = list(args.rho) if args.rho is not None else list(config.rho_values)
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "eval"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed, batch = int(config.reeval_seed), int(config.eval_batch_size)

    # 1. Training-split scores for every checkpoint and rho.
    sel_path = out_dir / "selection_train.csv"
    if args.reuse_selection and sel_path.is_file():
        sel_rows = pd.read_csv(sel_path).to_dict("records")
    else:
        X, Y_ref, Y_train, L, X_ages = load_pool(config, "train")
        tr = TransitionComputation(L=L, beta=config.beta,
                                   cov_regularization=config.cov_regularization, device=device)
        sel_rows = []
        for ck in checkpoints(run_dir):
            for rho in rhos:
                row = evaluate_checkpoint(config, ck, rho, X, Y_ref, Y_train, X_ages, tr, device,
                                          n_repeat=int(config.select_repeat),
                                          n_steps=int(config.select_steps),
                                          seed=seed, eval_batch_size=batch)
                sel_rows.append(row)
                print(f"[train] {row['checkpoint']} rho={rho}: SWD {row['swd_mean']:.4f}")
        _write(sel_path, sel_rows)

    # 2-3. Select per rho, then evaluate on the held-out pool.
    if args.eval_data is not None:
        config.processed_data_path = Path(args.eval_data)
    X, Y_ref, Y_train, L, X_ages = load_pool(config, "eval")
    tr = TransitionComputation(L=L, beta=config.beta,
                               cov_regularization=config.cov_regularization, device=device)
    sel = pd.DataFrame(sel_rows)
    final_rows, per_repeat = [], {}
    for rho in rhos:
        best = (sel[sel["rho"] == float(rho)]
                .sort_values(["swd_mean", "swd_std"], ascending=True).iloc[0])
        ck = run_dir / "checkpoints" / f"ckpt_{best['checkpoint']}.pth"
        row = evaluate_checkpoint(config, ck, rho, X, Y_ref, Y_train, X_ages, tr, device,
                                  n_repeat=int(args.eval_repeat or config.repeat_heavy),
                                  n_steps=int(config.num_rollout_steps_heavy),
                                  seed=seed, eval_batch_size=batch)
        final_rows.append(row)
        per_repeat[f"rho{float(rho):g}"] = np.asarray(row["swd_per_repeat"])
        print(f"[eval] {row['checkpoint']} rho={rho}: SWD {row['swd_mean']:.4f} "
              f"FD {row['fd_mean']:.2f} MMD {row['mmd']:.3e}")
    suffix = f"_{args.tag}" if args.tag else ""
    _write(out_dir / f"final_eval{suffix}.csv", final_rows)
    np.savez(out_dir / f"per_repeat{suffix}.npz", **per_repeat)


if __name__ == "__main__":
    main()
