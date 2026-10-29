"""Checkpoint selection for the learned baselines on the TRAINING split.

Same protocol as DBPS: every saved checkpoint generates endpoints for
config.select_repeat repeats of 100 training sources (config.select_steps
rollout steps) and is scored by SWD against 100 training targets. The
checkpoint with the lowest mean SWD is selected.

Writes <workdir>/trainset_results.csv and prints the selected checkpoint.
Pair with:

    python -m scripts.sample_baseline --config <cfg> --source_workdir <wd> --ckpt <selected>
    python -m scripts.eval_baseline --workdir <wd>
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import sliced_wasserstein_distance
from models.controller_net import ControllerNet
from training.train_fm import _impute_age_nan
from utils.config_utils import load_config, resolve_data_path
from utils.cli import add_setting_args, apply_setting_args
from utils.io_utils import ensure_dir
from utils.seed import make_rng, set_global_seed

EVAL_BATCH, SEED = 100, 42


def _family(method: str) -> str:
    # DSBM's network predicts the bridge endpoint rather than the drift, so it
    # needs its own rollout.
    return "dsbm" if method.startswith("dsbm") else "fm"


def _ckpts(workdir: Path):
    out = []
    for p in (workdir / "checkpoints").glob("ckpt_*.pth"):
        m = re.match(r"^ckpt_(.+)\.pth$", p.name)
        if m and m.group(1) != "last":
            out.append((m.group(1), p))
    return sorted(out, key=lambda t: t[0])


def _gen_endpoints(fam, model, config, x0, ages, steps, device, rng):
    """Generate endpoints for one training batch with the method's own sampler."""
    age_t = None
    if bool(getattr(config, "cond_on_age", False)):
        age_t = torch.tensor(ages, dtype=torch.float32, device=device).unsqueeze(-1)
    x_0 = torch.tensor(x0, dtype=torch.float64, device=device)
    if fam == "dsbm":
        from training.train_dsbm import _dsbm_rollout
        return _dsbm_rollout(
            x_0, model, steps, device, float(getattr(config, "g", 1.0)), rng,
            age=age_t,
            parameterization=getattr(config, "dsbm_parameterization", "endpoint"),
            drift_clamp=getattr(config, "dsbm_drift_clamp", 20.0))
    from training.train_fm import _fm_rollout
    out = _fm_rollout(x_0, model, steps, device, save_trajectories=False,
                      use_sde=bool(getattr(config, "use_sde_fm", False)),
                      g=float(getattr(config, "g", 1.0)), rng=rng, age=age_t)
    return out["x_1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--use_normalization", action="store_true")
    add_setting_args(ap)
    a = ap.parse_args()

    config = load_config(a.config)
    apply_setting_args(config, a)
    if a.use_normalization:
        config.use_normalization = True
        config.suffix = "_normalization"
    workdir = Path(a.workdir).resolve()
    set_global_seed(config.seed)

    steps = int(config.select_steps)
    repeat = int(config.select_repeat)
    print(f"[select_ckpt_trainset] protocol: train split, {steps} steps x {repeat} repeats")

    z = np.load(resolve_data_path(config))
    X_train = np.asarray(z["X_train"], np.float64)
    Y_train = np.asarray(z["Y_train"], np.float64)
    X_ages = _impute_age_nan(z["X_train_ages"]) if "X_train_ages" in z.files else None

    fam = _family(str(config.method))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = X_train.shape[1]
    model = ControllerNet(
        d=d, hidden_dim=int(config.hidden_dim), num_layers=int(config.num_layers),
        cond_on_y_star=False,
        cond_on_age=bool(getattr(config, "cond_on_age", False)),
        age_emb_dim=int(getattr(config, "age_emb_dim", 1)),
        time_emb_dim=int(config.time_emb_dim), dropout=float(config.dropout),
        activation=str(getattr(config, "activation", "silu")),
        time_embedding_type=str(getattr(config, "time_embedding_type", "sinusoidal")),
        block_type=str(getattr(config, "block_type", "residual"))).to(device)

    nX, nY = X_train.shape[0], Y_train.shape[0]
    eb = min(EVAL_BATCH, nX, nY)
    rows = []
    for label, path in _ckpts(workdir):
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        swds = []
        for r in range(repeat):
            rr = make_rng(SEED + r * 7919)
            ix = rr.choice(nX, eb, replace=False).astype(np.int64)
            iy = rr.choice(nY, eb, replace=False).astype(np.int64)
            xh = _gen_endpoints(fam, model, config, X_train[ix],
                                None if X_ages is None else X_ages[ix], steps, device, rr)
            Yb = torch.tensor(Y_train[iy], dtype=torch.float64, device=device)
            swds.append(float(sliced_wasserstein_distance(xh, Yb, num_projections=256, rng=rr)))
        arr = np.asarray([v for v in swds if np.isfinite(v)], dtype=np.float64)
        m = float(arr.mean()) if arr.size else float("nan")
        s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        rows.append({"checkpoint": label, "train_swd_mean": m, "train_swd_std": s,
                     "n_repeat": repeat, "steps": steps})
        print(f"  {label:14s} train_swd={m:.4f}")

    out = workdir / "trainset_results.csv"
    ensure_dir(workdir)
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if np.isfinite(r["train_swd_mean"])]
    best = min(ok, key=lambda r: r["train_swd_mean"])
    print(f"[select_ckpt_trainset] {workdir.name}: TRAIN-SELECTED = {best['checkpoint']} "
          f"(train_swd={best['train_swd_mean']:.4f}) -> {out}")
    print(f"TRAIN_SELECTED_CKPT={best['checkpoint']}")


if __name__ == "__main__":
    main()
