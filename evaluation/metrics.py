"""Evaluation metrics for unpaired-set evaluation.

All functions are torch-native and run on whatever device the input
tensors live on (GPU or CPU). Settings:

  - SWD:     num_slices=256, p=2; assumes n_true == n_pred.
             Caller must supply equal-size sample sets.
  - Sinkhorn-W2: epsilon=0.05, n_iters=200, **debiased=True**
                 (subtracts 0.5*OT(X,X) + 0.5*OT(Y,Y)).
  - FD:      eps=1e-6; custom sqrtm via stable eigh.
  - MMD:     median-heuristic bandwidth; raw mmd^2 (NO clamp to >= 0
             so cross-method comparisons keep sign info).

Inputs: torch.Tensor (any device, dtype) OR np.ndarray (auto-converted
to torch.float64 on CPU). Returns Python float.

Each metric carries `@torch.no_grad()` to avoid building autograd graphs.

For samples with n_a != n_b (e.g. concatenated-repeat output), use
maximum_mean_discrepancy and kde_log_likelihood (the only metrics that
support unequal sizes); SWD/Sinkhorn/FD callers must subsample or
loop-per-repeat and then mean+std.
"""
from __future__ import annotations

import math
import logging
from typing import Optional, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


# =====================================================================
# Internal helpers
# =====================================================================

def _to_tensor(x: ArrayLike, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert np.ndarray -> torch float64 on CPU. Pass-through tensors.
    If `ref` is provided, places result on the same device/dtype as ref.
    """
    if isinstance(x, torch.Tensor):
        return x
    t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    if ref is not None:
        t = t.to(device=ref.device, dtype=ref.dtype)
    return t


def match_sample_sizes(
    X: ArrayLike, Y: ArrayLike, rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Random-subsample the larger of (X, Y) to match the smaller one.

    SWD/Sinkhorn require n_X == n_Y;
    callers with unequal-size inputs (e.g. our X_eval=353 vs Y_eval=269
    in ADNI) should subsample using this helper before invoking the
    metric. FD does NOT need this (it fits independent Gaussians).

    Returns (X_sub, Y_sub) as torch tensors on X's device (or CPU if X
    was a numpy array). The caller controls reproducibility via `rng`.
    """
    if rng is None:
        rng = np.random.default_rng()
    Xt = _to_tensor(X)
    Yt = _to_tensor(Y).to(device=Xt.device)
    n_min = min(Xt.shape[0], Yt.shape[0])
    if Xt.shape[0] > n_min:
        idx = rng.choice(Xt.shape[0], n_min, replace=False).astype(np.int64)
        Xt = Xt[torch.from_numpy(idx).to(device=Xt.device)]
    if Yt.shape[0] > n_min:
        idx = rng.choice(Yt.shape[0], n_min, replace=False).astype(np.int64)
        Yt = Yt[torch.from_numpy(idx).to(device=Yt.device)]
    return Xt, Yt


def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.T)


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _eigh_stable(A: torch.Tensor, eps: float = 1e-6, max_tries: int = 8):
    """Robust symmetric eigendecomposition with adaptive jitter.

    Uses .double() internally for
    numerical robustness regardless of input dtype.
    """
    A = A.double()
    A = _sym(A)
    if not _is_finite_tensor(A):
        A = torch.nan_to_num(A)

    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    diag_mean = torch.clamp(torch.diag(A).mean(), min=0.0).item()
    scale = diag_mean if math.isfinite(diag_mean) and diag_mean > 0.0 else 1.0
    base = max(eps, 1e-12) * scale

    err = None
    for i in range(max_tries):
        jitter = base * (10.0 ** i)
        try:
            evals, evecs = torch.linalg.eigh(A + jitter * I)
            evals = torch.clamp(evals, min=0.0)
            return evals, evecs
        except Exception as e:
            err = e
            continue

    logging.warning(
        "eigh failed after %d tries (%s); using SVD fallback.", max_tries, err,
    )
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    S = torch.clamp(S, min=0.0)
    return S, U


def _sqrtm_psd_stable(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable matrix square root for symmetric PSD via _eigh_stable."""
    evals, evecs = _eigh_stable(A, eps=eps)
    S = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T
    return _sym(S)


def _mean_and_cov(
    x: torch.Tensor, eps: float = 1e-6, shrinkage: bool = False,
):
    """x: [B, D] -> mean [D], cov [D, D]. eps added on diagonal."""
    x = x.double()
    B, D = x.shape
    mu = x.mean(dim=0)
    xc = x - mu
    S = (xc.T @ xc) / max(B - 1, 1)
    if not shrinkage:
        cov = S
    else:
        alpha = torch.trace(S) / D
        x2 = xc * xc
        var_frob = (x2.T @ x2).sum() / (B * B) - (S ** 2).sum()
        phi_hat = torch.clamp(var_frob, min=0.0)
        gamma_hat = torch.norm(
            S - alpha * torch.eye(D, device=x.device, dtype=x.dtype),
            p="fro",
        ) ** 2
        lam = float(phi_hat / max(float(gamma_hat), eps))
        lam = max(0.0, min(1.0, lam))
        cov = (1.0 - lam) * S + lam * alpha * torch.eye(
            D, device=x.device, dtype=x.dtype,
        )
    cov = _sym(cov) + eps * torch.eye(D, device=x.device, dtype=x.dtype)
    return mu, cov


# =====================================================================
# Frechet Distance (raw d-D Gaussians; not image FID)
# =====================================================================

def _frechet_gaussians(
    mu1: torch.Tensor, cov1: torch.Tensor,
    mu2: torch.Tensor, cov2: torch.Tensor, eps: float = 1e-6,
) -> float:
    """W2^2 between two Gaussians N(mu1, cov1) and N(mu2, cov2)."""
    mu1 = mu1.double()
    mu2 = mu2.double()
    cov1 = _sym(cov1.double())
    cov2 = _sym(cov2.double())
    D = mu1.numel()
    I = torch.eye(D, device=mu1.device, dtype=mu1.dtype)
    cov1 = torch.nan_to_num(cov1) + eps * I
    cov2 = torch.nan_to_num(cov2) + eps * I

    S1 = _sqrtm_psd_stable(cov1, eps=eps)
    M = _sym(S1 @ cov2 @ S1)
    M_sqrt = _sqrtm_psd_stable(M, eps=eps)

    mean_diff = mu1 - mu2
    fid = mean_diff.dot(mean_diff) + torch.trace(cov1 + cov2 - 2.0 * M_sqrt)
    return float(torch.clamp(fid, min=0.0).item())


def frechet_distance(
    X_hat: ArrayLike,
    Y_real: ArrayLike,
    cov_eps: float = 1e-6,
    shrinkage: bool = False,
    notes_dict: Optional[dict] = None,
) -> float:
    """Frechet distance between two sample sets.

    FD = ||mu_a - mu_b||^2 + tr(Sigma_a + Sigma_b - 2 (Sigma_a Sigma_b)^{1/2})

    `cov_eps` (default 1e-6) regularizes both covariances and the eigh
    inside sqrtm.
    """
    X = _to_tensor(X_hat)
    Y = _to_tensor(Y_real)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"FD: X_hat and Y_real must be [n, d] with same d; "
            f"got {tuple(X.shape)} and {tuple(Y.shape)}"
        )

    eps_f = float(cov_eps)
    mu_t, cov_t = _mean_and_cov(X, eps=eps_f, shrinkage=shrinkage)
    mu_p, cov_p = _mean_and_cov(Y, eps=eps_f, shrinkage=shrinkage)

    if notes_dict is not None:
        # Useful diagnostics for reproducing FD across runs.
        notes_dict["frechet_distance_cov_eps"] = eps_f
        notes_dict["frechet_distance_cond_a"] = float(
            torch.linalg.cond(cov_t).item()
        )
        notes_dict["frechet_distance_cond_b"] = float(
            torch.linalg.cond(cov_p).item()
        )

    try:
        return _frechet_gaussians(mu_t, cov_t, mu_p, cov_p, eps=eps_f)
    except Exception as e:
        logging.warning("FD computation failed (%s). Returning NaN.", e)
        return float("nan")


# =====================================================================
# Sliced Wasserstein Distance (SWD)
# =====================================================================

@torch.no_grad()
def sliced_wasserstein_distance(
    X_hat: ArrayLike,
    Y_real: ArrayLike,
    num_projections: int = 256,
    p: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """SWD with random 1D projections.

    Requires n_true == n_pred. Caller must subsample
    or loop-per-repeat to satisfy this.

    `rng` (numpy generator) is honored: we draw the projection
    directions via `rng.standard_normal((num_projections, d))` so SWD
    is reproducible across re-evaluations with the same seed. If `rng`
    is None we fall back to torch's global RNG.
    """
    X = _to_tensor(X_hat)
    Y = _to_tensor(Y_real)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"SWD: X_hat and Y_real must be [n, d] with same d; "
            f"got {tuple(X.shape)} and {tuple(Y.shape)}"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"SWD requires n_true == n_pred; "
            f"got n_X={X.shape[0]}, n_Y={Y.shape[0]}. Caller must "
            "subsample or use per-repeat aggregation."
        )

    # Bring X, Y to a common device/dtype (use double for stability).
    target_device = X.device
    X = X.double()
    Y = Y.to(device=target_device).double()

    D = X.shape[1]
    if rng is not None:
        # Reproducible: draw on CPU via numpy rng, then move to device.
        np_dirs = rng.standard_normal((int(num_projections), D)).astype(np.float64)
        dirs = torch.from_numpy(np_dirs).to(device=target_device, dtype=torch.float64)
    else:
        dirs = torch.randn(
            (int(num_projections), D), device=target_device, dtype=torch.float64,
        )
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)

    proj_t = X @ dirs.T                                  # [n, num_projections]
    proj_p = Y @ dirs.T
    proj_t_sorted, _ = torch.sort(proj_t, dim=0)
    proj_p_sorted, _ = torch.sort(proj_p, dim=0)
    diff = proj_t_sorted - proj_p_sorted
    if p == 1:
        w_each = diff.abs().mean(dim=0)
    elif p == 2:
        w_each = diff.pow(2).mean(dim=0).sqrt()
    else:
        w_each = diff.abs().pow(p).mean(dim=0).pow(1.0 / p)
    return float(w_each.mean().item())


# =====================================================================
# Sinkhorn W2 (epsilon-regularized OT, debiased by default)
# =====================================================================

@torch.no_grad()
def sinkhorn_w2(
    X_hat: ArrayLike,
    Y_real: ArrayLike,
    epsilon: float = 0.05,
    num_iters: int = 200,
    debiased: bool = True,
) -> float:
    """Sinkhorn W2 (returns sqrt of entropic OT, optionally debiased).

    `debiased=True` subtracts 0.5*OT(X,X) + 0.5*OT(Y,Y) from OT(X,Y),
    which makes OT(X,X)=0 in the limit and gives a bona-fide divergence.
    """
    X = _to_tensor(X_hat)
    Y = _to_tensor(Y_real)
    target_device = X.device
    X = X.double()
    Y = Y.to(device=target_device).double()

    def _cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1, keepdim=True).T
        C = x2 - 2.0 * (x @ y.T) + y2
        return C.clamp_min_(0.0)

    def _sinkhorn_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Bx, By = x.size(0), y.size(0)
        a = torch.full((Bx,), 1.0 / Bx, device=x.device, dtype=x.dtype)
        b = torch.full((By,), 1.0 / By, device=y.device, dtype=y.dtype)
        C = _cost(x, y)
        K = torch.exp(-C / float(epsilon)) + 1e-12
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(int(num_iters)):
            u = a / (K @ v)
            v = b / (K.T @ u)
        P = (u.unsqueeze(1) * K) * v.unsqueeze(0)
        return torch.sum(P * C)

    ot_ab = _sinkhorn_cost(X, Y)
    if debiased:
        ot_aa = _sinkhorn_cost(X, X)
        ot_bb = _sinkhorn_cost(Y, Y)
        val = float((ot_ab - 0.5 * ot_aa - 0.5 * ot_bb).item())
    else:
        val = float(ot_ab.item())
    val = max(val, 0.0)
    return math.sqrt(val)


# =====================================================================
# MMD (RBF kernel, median bandwidth, drop-NaN, NO clamp)
# =====================================================================

@torch.no_grad()
def maximum_mean_discrepancy(
    X_hat: ArrayLike,
    Y_real: ArrayLike,
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
    max_n: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Unbiased MMD^2 with RBF kernel, median heuristic.

    Convention:
      - bandwidth = `median(d^2)` (NOT sqrt(median)) -- the formula
        already has 2*h in the denominator
      - drop NaN rows
      - unbiased estimator: K_XX, K_YY exclude diagonal
      - return raw mmd^2 (no clamp)

    Each side is capped at `max_n` random samples (default 4000) to bound
    peak memory on the pooled evaluation sample.
    """
    if kernel != "rbf":
        raise ValueError(f"MMD: only 'rbf' kernel is supported, got {kernel!r}")
    X = _to_tensor(X_hat)
    Y = _to_tensor(Y_real)

    mask_x = ~torch.isnan(X).any(dim=1)
    mask_y = ~torch.isnan(Y).any(dim=1)
    target_device = X.device
    x = X[mask_x].to(device=target_device).double()
    y = Y[mask_y].to(device=target_device).double()
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")

    # Subsample for memory safety. If the input fits under max_n, no
    # change. If above, draw a random sub-sample (reproducible if rng
    # supplied).
    if rng is None:
        rng = np.random.default_rng()
    if x.shape[0] > max_n:
        idx = rng.choice(x.shape[0], int(max_n), replace=False).astype(np.int64)
        x = x[torch.from_numpy(idx).to(device=target_device)]
    if y.shape[0] > max_n:
        idx = rng.choice(y.shape[0], int(max_n), replace=False).astype(np.int64)
        y = y[torch.from_numpy(idx).to(device=target_device)]

    def _pairwise_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).T
        return (a2 - 2.0 * (a @ b.T) + b2).clamp_min_(0.0)

    if bandwidth is None:
        all_pts = torch.cat([x, y], dim=0)
        dists = _pairwise_sq(all_pts, all_pts)
        positive = dists[dists > 0]
        if positive.numel() == 0:
            return float("nan")
        median_dist = float(torch.median(positive).item())
        bandwidth = max(median_dist, 1e-6)
    h = float(bandwidth)

    K_xx = torch.exp(-_pairwise_sq(x, x) / (2.0 * h))
    K_yy = torch.exp(-_pairwise_sq(y, y) / (2.0 * h))
    K_xy = torch.exp(-_pairwise_sq(x, y) / (2.0 * h))

    N, M = x.size(0), y.size(0)
    Kxx_sum = (K_xx.sum() - K_xx.trace()) / (N * (N - 1))
    Kyy_sum = (K_yy.sum() - K_yy.trace()) / (M * (M - 1))
    Kxy_sum = K_xy.sum() / (N * M)

    return float((Kxx_sum + Kyy_sum - 2.0 * Kxy_sum).item())


# =====================================================================
# KDE log-likelihood
# =====================================================================

def kde_log_likelihood(
    X_hat_samples: ArrayLike,
    Y_real: ArrayLike,
    bandwidth: Optional[float] = None,
) -> float:
    """Mean log-likelihood of X_hat under a Gaussian KDE fit on Y_real.

    Default bandwidth is Scott's rule applied to Y_real. Uses sklearn
    on CPU regardless of input device (sklearn is CPU-only).
    """
    from sklearn.neighbors import KernelDensity

    Y = _to_tensor(Y_real).cpu().numpy().astype(np.float64)
    X = _to_tensor(X_hat_samples).cpu().numpy().astype(np.float64)
    if Y.ndim != 2 or X.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"KDE: X and Y must be [n, d] with same d; "
            f"got {X.shape} and {Y.shape}"
        )
    n_b, d = Y.shape

    if bandwidth is None:
        per_dim_std = Y.std(axis=0, ddof=1)
        bandwidth = float(n_b ** (-1.0 / (d + 4)) * per_dim_std.mean())
    bandwidth = max(float(bandwidth), 1e-6)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(Y)
    return float(kde.score_samples(X).mean())
