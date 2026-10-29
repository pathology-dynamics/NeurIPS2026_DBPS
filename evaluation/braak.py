"""Braak-stage analysis helpers (Section 5, Appendix I).

ROI -> Braak stage mapping, Benjamini-Hochberg q-values, young-vs-old and
one-sample Cohen's d per ROI, Spearman alignment with Braak stage, and the
two controller estimands:
  dudage:          d u_theta,k / d age at a fixed time t (forward-mode AD);
  endpoint_shift:  Delta = X_1 - X_0 of DBPS rollouts, averaged over rollouts.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import stats

from controls.db_control import compute_u_anchor
from dynamics.rollout import rollout
from utils.seed import make_rng

BRAAK_LABELS = {
    0: "non-Braak",
    2: "I-II early",
    3: "III limbic",
    4: "IV temporal/limbic",
    5: "V association",
    6: "VI late",
}


def braak_stage(name: str) -> int:
    """Braak stage of a FreeSurfer ROI name (2 = I-II, ..., 6 = VI; 0 = not mapped)."""
    n = str(name).lower()
    if "cerebellum" in n:
        return 0
    if "entorhinal" in n:
        return 2
    if any(k in n for k in ("hippocampus", "parahippocampal", "amygdala", "fusiform")):
        return 3
    if "insula" in n:
        return 4
    if any(k in n for k in (
        "rostralanteriorcingulate", "caudalanteriorcingulate",
        "rostralmiddlefrontal", "caudalmiddlefrontal", "orbitofrontal",
        "middletemporal", "superiortemporal", "inferiortemporal",
        "transversetemporal", "temporalpole", "bankssts",
    )):
        return 4
    if any(k in n for k in (
        "posteriorcingulate", "isthmuscingulate",
        "parietal", "supramarginal", "precuneus", "frontalpole",
        "parsopercularis", "parstriangularis", "parsorbitalis",
        "superiorfrontal", "lingual",
    )):
        return 5
    if any(k in n for k in (
        "precentral", "postcentral", "paracentral",
        "cuneus", "pericalcarine", "lateraloccipital",
    )):
        return 6
    return 0


def braak_priority(stage: int) -> float:
    """Higher priority means earlier Braak involvement (stage 2 -> 5, stage 6 -> 1)."""
    return math.nan if stage <= 0 else float(7 - stage)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    qvalues = np.full(pvalues.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return qvalues
    idx = np.where(valid)[0]
    p = pvalues[idx]
    order = np.argsort(p)
    q = p[order] * float(len(p)) / np.arange(1, len(p) + 1)
    q = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0.0, 1.0)
    qvalues[idx[order]] = q
    return qvalues


def subset_qvalues(pvalues: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full_like(np.asarray(pvalues, dtype=np.float64), np.nan)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        out[mask] = bh_qvalues(np.asarray(pvalues, dtype=np.float64)[mask])
    return out


def two_sample_d(values: np.ndarray, ages: np.ndarray) -> dict:
    """Young-vs-old Cohen's d per column (median split; young < median <= old), Welch p-values."""
    ages = np.asarray(ages, dtype=np.float64)
    median_age = float(np.median(ages))
    young, old = values[ages < median_age], values[ages >= median_age]
    n1, n2 = young.shape[0], old.shape[0]
    pooled = np.sqrt(((n1 - 1) * young.var(axis=0, ddof=1) + (n2 - 1) * old.var(axis=0, ddof=1)) / (n1 + n2 - 2))
    pooled = np.where(pooled < 1e-12, 1e-12, pooled)
    _, p = stats.ttest_ind(young, old, axis=0, equal_var=False)
    return {"d": (young.mean(axis=0) - old.mean(axis=0)) / pooled, "pvalue": np.asarray(p, dtype=np.float64),
            "median_age": median_age, "n_young": int(n1), "n_old": int(n2)}


def one_sample_d(values: np.ndarray) -> dict:
    sd = values.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    _, p = stats.ttest_1samp(values, 0.0, axis=0)
    return {"d": values.mean(axis=0) / sd, "pvalue": np.asarray(p, dtype=np.float64)}


def spearman_braak(abs_d: np.ndarray, stages: np.ndarray, keep: np.ndarray | None = None) -> dict:
    """Spearman correlation between |d| and Braak stage over Braak-mapped ROIs (optionally a subset)."""
    mask = np.asarray(stages) > 0
    if keep is not None:
        mask &= np.asarray(keep, dtype=bool)
    rho, p = stats.spearmanr(np.asarray(abs_d)[mask], np.asarray(stages)[mask])
    return {"rho": float(rho), "p": float(p), "n_roi": int(mask.sum())}


def dudage(model, X: np.ndarray, ages: np.ndarray, t: float, device) -> np.ndarray:
    """d u_theta / d age for every subject and ROI at time t (forward-mode AD)."""
    n = X.shape[0]
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    t_t = torch.full((n,), float(t), dtype=torch.float32, device=device)
    age = torch.tensor(ages, dtype=torch.float32, device=device).reshape(n, 1)
    _, out = torch.func.jvp(lambda a: model(X_t, t_t, age=a), (age,), (torch.ones_like(age),))
    return out.detach().to(torch.float64).cpu().numpy()


def endpoint_shift(model, X: np.ndarray, ages, Y_train: np.ndarray, transition, config, rho: float,
                   n_rollouts: int, device, seed: int = 42) -> np.ndarray:
    """Mean endpoint shift X_1 - X_0 of DBPS rollouts (anchors uniform from Y_train; rollout k seeded seed + 1000 k)."""
    n, d = X.shape
    total = np.zeros((n, d), dtype=np.float64)
    age_t = (torch.tensor(ages, dtype=torch.float32, device=device).unsqueeze(-1)
             if bool(getattr(config, "cond_on_age", False)) else None)
    rho = float(rho)

    def controller_fn(x_t, t, y_star, candidates):
        ua = compute_u_anchor(x_t, t, y_star, gamma=config.gamma, beta=config.beta, g=config.g,
                              transition=transition)
        with torch.no_grad():
            un = model(x_t.to(dtype=torch.float32, device=device), t.to(dtype=torch.float32, device=device),
                       age=age_t).to(dtype=torch.float64)
        return (1.0 - rho) * ua + rho * un

    for k in range(int(n_rollouts)):
        rng_k = make_rng(seed + 1000 * k)
        idx_y = rng_k.choice(Y_train.shape[0], size=n).astype(np.int64)
        out = rollout(
            x_0=torch.tensor(X, dtype=torch.float64, device=device), controller_fn=controller_fn,
            y_star=torch.tensor(Y_train[idx_y], dtype=torch.float64, device=device), candidates=None,
            transition=transition, num_steps=int(config.num_rollout_steps_heavy), g=float(config.g),
            use_sde=bool(config.use_sde), save_trajectories=False, save_controls=False, rng=rng_k,
        )
        total += out["x_1"].cpu().numpy() - X
    return total / int(n_rollouts)
