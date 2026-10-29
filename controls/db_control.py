"""Closed-form control laws of the Doob-Bridge Posterior Sampler (DBPS).

All functions here implement closed-form expressions derived for the
Gaussian transition kernel of the uncontrolled SDE
    dX_t = -beta L X_t dt + sqrt(beta) dW_t
under linear graph dynamics. Inputs may be np.ndarray or torch.Tensor;
internal computation is torch.float64 throughout. Outputs are
torch.float64.

Math summary
------------
eta = beta / g**2
k   = gamma / eta
m   = M_{1-t} x_t              (predicted mean of X_1 given X_t)
C   = C_{1-t}
A   = I + k * C

Closed forms:
    log psi_{gamma, y}(x_t, t)
        = -0.5 * log det(A) - 0.5 * k * (y - m)^T A^{-1} (y - m)

    u*_{gamma, y}(x_t)
        = (beta / g) * k * M_{1-t}^T A^{-1} (y - m)

    alpha_k(x_t, t)
        = softmax_k log psi_{gamma, y_k}(x_t, t)  (finite-gamma posterior)
        Equivalently the residual is measured in the Sigma_soft norm
        Sigma_soft := C_{1-t} + (eta/gamma) I; the softmax cancels the
        log det(A) constant in y, so only the residual quadratic enters.

    u_pop(x_t, t)  = sum_k alpha_k * u*_{gamma, y_k}(x_t)

A is materialized only via its eigenvalues; we never form A as a dense
matrix. This piggybacks on TransitionComputation's cached eigendecomp.

t may be a Python scalar / 0-d tensor (rollout and sampling) or a
per-sample tensor of shape [batch] (training draws one time per
minibatch element).
"""
from typing import Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


# Per-element clamp on u*_{gamma, y}(x_t) before it enters u_pop / u_anchor.
# Near t = 0 / t = 1 with large gamma, u* reaches the 1e3 -- 1e5 range, which
# would dominate the MSE loss and prevent convergence.
U_STAR_CLAMP: float = 20.0


def _t_to_tensor(
    t: Union[float, ArrayLike], device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert t to a float64 torch.Tensor on the given device.

    Scalar (Python float, np scalar, 0-d tensor) -> 0-d tensor.
    [batch] (np.ndarray or 1-d tensor) -> 1-d tensor.
    """
    if isinstance(t, torch.Tensor):
        out = t.to(dtype=torch.float64)
    elif isinstance(t, np.ndarray):
        out = torch.from_numpy(np.asarray(t, dtype=np.float64))
    else:
        out = torch.tensor(float(t), dtype=torch.float64)
    if device is not None:
        out = out.to(device=device)
    return out


def _to_device(x: ArrayLike, device: torch.device) -> torch.Tensor:
    """Promote x to torch.float64 on `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float64)
    return torch.from_numpy(np.asarray(x, dtype=np.float64)).to(device=device)


def _solve_A_inv_v(
    transition, t: Union[float, ArrayLike], k: float, v: torch.Tensor,
) -> torch.Tensor:
    """Compute A^{-1} @ v where A = I + k * C_{1-t}, batched.

    Uses the cached eigendecomposition L = U diag(sigma) U^T (so
    C_{1-t} = U diag(d_i^{(1-t)}) U^T and A^{-1} = U diag(1/(1+k d_i)) U^T).
    Forming A in dense form is avoided.

    Args:
      transition: TransitionComputation instance.
      t: scalar OR [batch] tensor (matches v's leading dim).
      k: scalar gamma / eta.
      v: [batch, d] or [batch, K, d] torch.float64.

    Returns:
      Same shape as v.
    """
    t_t = _t_to_tensor(t)
    delta = 1.0 - t_t                                     # scalar or [...]
    d_i = transition._d_eigvals(delta)                    # [d] or [..., d]
    inv_diag = 1.0 / (1.0 + float(k) * d_i)
    return transition._apply_diag_in_eigbasis(inv_diag, v)


def compute_log_psi(
    x_t: ArrayLike,
    t: Union[float, ArrayLike],
    y: ArrayLike,
    gamma: float,
    beta: float,
    g: float,
    transition,
) -> torch.Tensor:
    """log psi_{gamma, y}(x_t, t).

    Shapes:
      x_t: [batch, d]
      y:   [d] or [batch, d] or [batch, K, d]
      t:   scalar OR [batch]
    Returns:
      [batch]    if y is [d] or [batch, d]
      [batch, K] if y is [batch, K, d]
    """
    dev = transition.U.device
    t_t = _t_to_tensor(t, device=dev)
    delta = 1.0 - t_t
    eta = float(beta) / (float(g) ** 2)
    k = float(gamma) / eta

    x_t_t = _to_device(x_t, dev)
    y_t = _to_device(y, dev)
    if x_t_t.ndim != 2:
        raise ValueError(f"x_t must be [batch, d], got shape {tuple(x_t_t.shape)}")

    m = transition.M_apply(delta, x_t_t)                  # [batch, d]

    d_i = transition._d_eigvals(delta)                    # [d] or [..., d]
    log_det_A = torch.sum(torch.log1p(k * d_i), dim=-1)   # scalar or [...]

    if y_t.ndim == 1:
        residual = y_t.unsqueeze(0) - m                   # [batch, d]
    elif y_t.ndim == 2:
        residual = y_t - m                                # [batch, d]
    elif y_t.ndim == 3:
        residual = y_t - m.unsqueeze(1)                   # [batch, K, d]
    else:
        raise ValueError(f"y must have 1, 2, or 3 dims, got {y_t.ndim}")

    A_inv_r = _solve_A_inv_v(transition, t_t, k, residual)
    if residual.ndim == 2:
        quad = torch.einsum("bd,bd->b", residual, A_inv_r)
    else:
        quad = torch.einsum("bkd,bkd->bk", residual, A_inv_r)

    if quad.ndim == 2 and log_det_A.ndim == 1:
        log_det_A = log_det_A.unsqueeze(-1)               # [B, 1] -> [B, K]

    return -0.5 * log_det_A - 0.5 * k * quad


def compute_u_anchor(
    x_t: ArrayLike,
    t: Union[float, ArrayLike],
    y_star: ArrayLike,
    gamma: float,
    beta: float,
    g: float,
    transition,
) -> torch.Tensor:
    """u*_{gamma, y*}(x_t) = (beta/g) * k * M_{1-t}^T A^{-1} (y* - m).

    Since M is symmetric (M = U diag(.) U^T), M^T == M, so we can
    compute (M @ A_inv_diff) directly via M_apply.
    """
    dev = transition.U.device
    t_t = _t_to_tensor(t, device=dev)
    delta = 1.0 - t_t
    eta = float(beta) / (float(g) ** 2)
    k = float(gamma) / eta

    x_t_t = _to_device(x_t, dev)
    y_star_t = _to_device(y_star, dev)

    m = transition.M_apply(delta, x_t_t)                  # [batch, d]
    diff = y_star_t - m                                   # [batch, d]
    A_inv_diff = _solve_A_inv_v(transition, t_t, k, diff)
    u = (float(beta) / float(g)) * k * transition.M_apply(delta, A_inv_diff)
    return torch.clamp(u, -U_STAR_CLAMP, U_STAR_CLAMP)


def compute_alpha(
    x_t: ArrayLike,
    t: Union[float, ArrayLike],
    candidates: ArrayLike,
    gamma: float,
    beta: float,
    g: float,
    transition,
) -> torch.Tensor:
    """Finite-gamma posterior weights over candidate endpoints.

    alpha_k(x_t, t) = softmax_k log psi_{gamma, y_k}(x_t, t), where (using
    the closed form in compute_log_psi):
        log psi_{gamma, y}(x_t, t)
            = -0.5 * log det(A) - 0.5 * k * (y - m)^T A^{-1} (y - m)
        with A = I + k * C_{1-t}, k = gamma / eta, eta = beta / g**2,
        and m = M_{1-t} x_t.

    The log det(A) term is constant in y for a fixed (x_t, t) and cancels
    in the row-wise softmax, so the effective scoring is the residual
    quadratic in the Sigma_soft norm:
        log psi (up to const)  =  -0.5 (y - m)^T Sigma_soft^{-1} (y - m)
        Sigma_soft := C_{1-t} + (eta/gamma) I

    Limit cases:
      gamma -> infty:  Sigma_soft -> C_{1-t}; alpha -> softmax(log p_{t1})
                       i.e. the previous gamma-free Bayesian posterior.
      gamma -> 0:      Sigma_soft -> infty; alpha -> uniform 1/K.

    Returns: [batch, K], rows sum to 1.
    """
    cand_t = _to_device(candidates, transition.U.device)
    if cand_t.ndim != 3:
        raise ValueError(
            f"candidates must be [batch, K, d], got {tuple(cand_t.shape)}"
        )
    log_psi = compute_log_psi(
        x_t, t, cand_t, gamma, beta, g, transition,
    )                                                     # [batch, K]
    return torch.softmax(log_psi, dim=-1)


def compute_u_pop_label(
    x_t: ArrayLike,
    t: Union[float, ArrayLike],
    candidates: ArrayLike,
    gamma: float,
    beta: float,
    g: float,
    transition,
) -> torch.Tensor:
    """u_pop = sum_k alpha_k * u*_{gamma, y_k}(x_t).

    candidates: [batch, K, d]. Anchor at slot 0 by convention; do NOT
    exclude it.
    """
    dev = transition.U.device
    t_t = _t_to_tensor(t, device=dev)
    delta = 1.0 - t_t
    eta = float(beta) / (float(g) ** 2)
    k = float(gamma) / eta

    x_t_t = _to_device(x_t, dev)
    cand_t = _to_device(candidates, dev)
    if cand_t.ndim != 3:
        raise ValueError(
            f"candidates must be [batch, K, d], got {tuple(cand_t.shape)}"
        )

    m = transition.M_apply(delta, x_t_t)                  # [batch, d]
    diff = cand_t - m.unsqueeze(1)                        # [batch, K, d]

    A_inv_diff = _solve_A_inv_v(transition, t_t, k, diff)
    u_per_cand = (
        (float(beta) / float(g)) * k * transition.M_apply(delta, A_inv_diff)
    )
    # Clamp per-candidate u* before averaging (keeps
    # boundary blow-up at t~0 / t~1 from dominating the soft target).
    u_per_cand = torch.clamp(u_per_cand, -U_STAR_CLAMP, U_STAR_CLAMP)

    alpha = compute_alpha(x_t, t_t, cand_t, gamma, beta, g, transition)
    return torch.einsum("bk,bkd->bd", alpha, u_per_cand)

