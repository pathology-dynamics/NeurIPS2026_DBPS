"""Sample a learned baseline from a selected checkpoint.

Loads <source_workdir>/checkpoints/ckpt_<ckpt>.pth (the training-split
selection from scripts.select_ckpt_trainset), samples the evaluation pool
of the config's data with the heavy protocol (config.repeat_heavy repeats,
config.num_rollout_steps_heavy steps, seeded rng), and writes
generated_endpoints.npy into the config's output workdir for
scripts.eval_baseline. With --data, the checkpoint is applied to another
setting's evaluation pool without retraining (cross-cohort transfer), e.g.
--setting union_to_adni for a checkpoint trained on the Union data.

Supports the flow-matching family (CFM, OT-CFM, SOC-Matching, DSB-lite)
and DSBM.

usage:
  python -m scripts.sample_baseline --config configs/methods/cfm_ot.py \
      --source_workdir outputs/adni/cfm_ot_normalization --ckpt epoch024 --setting adni
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.controller_net import ControllerNet
from utils.cli import add_setting_args, apply_setting_args
from utils.config_utils import load_config, resolve_data_path
from utils.io_utils import ensure_dir, save_config_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source_workdir", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint label, e.g. epoch024 or iter008")
    add_setting_args(ap)
    args = ap.parse_args()

    config = load_config(args.config)
    config.use_normalization = True
    config.suffix = "_normalization"
    apply_setting_args(config, args)

    src = Path(args.source_workdir)
    ckpt_path = src / "checkpoints" / f"ckpt_{args.ckpt}.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = str(config.method)
    is_dsbm = method.startswith("dsbm")
    if is_dsbm:
        from training.train_dsbm import _heavy_sample, _resolve_workdir
    elif method.startswith(("cfm_", "soc_matching", "dsb_lite")):
        from training.train_fm import _heavy_sample_best_ckpt as _heavy_sample, _resolve_workdir
    else:
        raise ValueError(f"sampling not supported for method {method!r}")
    from training.train_fm import _impute_age_nan

    z = np.load(resolve_data_path(config))
    X_eval = np.concatenate([np.asarray(z["X_val"], np.float64),
                             np.asarray(z["X_test"], np.float64)], axis=0)
    X_eval_ages = None
    if "X_val_ages" in z.files:
        X_eval_ages = _impute_age_nan(np.concatenate(
            [np.asarray(z["X_val_ages"], np.float64),
             np.asarray(z["X_test_ages"], np.float64)], axis=0))

    model = ControllerNet(
        d=X_eval.shape[1], hidden_dim=int(config.hidden_dim), num_layers=int(config.num_layers),
        cond_on_y_star=False,
        cond_on_age=bool(getattr(config, "cond_on_age", False)),
        age_emb_dim=int(getattr(config, "age_emb_dim", 32 if is_dsbm else 1)),
        time_emb_dim=int(config.time_emb_dim), dropout=float(config.dropout),
        activation=str(getattr(config, "activation", "silu")),
        time_embedding_type=str(getattr(config, "time_embedding_type", "sinusoidal")),
        block_type=str(getattr(config, "block_type", "residual")),
    ).to(device)

    workdir = _resolve_workdir(config)
    ensure_dir(workdir)
    save_config_snapshot(config, workdir)
    if is_dsbm:
        ge, ysi = _heavy_sample(model, ckpt_path, config, X_eval, X_eval_ages, device)
    else:
        ge, ysi = _heavy_sample(model, ckpt_path, config, X_eval, device, X_eval_ages)

    np.save(workdir / "generated_endpoints.npy", ge.astype(np.float64))
    np.save(workdir / "y_star_indices.npy", ysi)
    (workdir / "sample_summary.txt").write_text(
        f"{method} from {src} (ckpt {args.ckpt})\n"
        f"n_eval={ge.shape[0]} repeat={ge.shape[1]} d={ge.shape[2]}\n")
    print(f"DONE {method}: {workdir} (source {src.name} ckpt {args.ckpt})")


if __name__ == "__main__":
    main()
