"""Euler / Euler-Maruyama rollout from t=0 to t=1 under the
controlled (or uncontrolled) bridge SDE.

The drift comes from the cached eigendecomposition stored on the
TransitionComputation instance: L_full = U diag(sigma) U^T is
materialized once at function entry; per-step drift is then a single
matmul.
"""
from typing import Callable, Optional, Union

import numpy as np
import torch


# Type alias for the controller function passed to rollout. Uniform
# across all methods (uncontrolled, analytic, learned).
ControllerFn = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    torch.Tensor,
]
# Signature: (x_t, t, y_star, candidates) -> u
#   x_t:        [batch, d]
#   t:          [batch] in [0, 1]
#   y_star:     [batch, d] or None
#   candidates: [batch, K, d] or None
#   returns u:  [batch, d]


def rollout(
    x_0: Union[np.ndarray, torch.Tensor],
    controller_fn: ControllerFn,
    y_star: Optional[torch.Tensor],
    candidates: Optional[torch.Tensor],
    transition,
    num_steps: int,
    g: float = 1.0,
    use_sde: bool = True,
    save_trajectories: bool = True,
    save_controls: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Euler-Maruyama (SDE) or Euler (ODE) rollout from t=0 to t=1.

    Discretization:
        dt = 1.0 / num_steps
        For step k = 0, 1, ..., num_steps - 1:
          t_k     = k * dt
          x_k     = current state, shape [batch, d]
          drift   = -beta * L @ x_k + g * u_k
          u_k     = controller_fn(x_k, t_k_batch, y_star, candidates)
          if use_sde:
              z_k = N(0, I) of shape [batch, d]
              x_{k+1} = x_k + drift * dt + sqrt(beta * dt) * z_k
          else:
              x_{k+1} = x_k + drift * dt

    All computation in torch.float64.

    Returns dict with keys:
      'x_1':            [batch, d] torch.float64
      'trajectories':   [batch, num_steps + 1, d] if save_trajectories
                        (slot 0 = x_0, slot num_steps = x_1)
      'controls':       [batch, num_steps, d] if save_controls

    The controller_fn argument t is passed as a torch.float64 tensor of
    shape [batch] filled with the current scalar t_k.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if rng is None:
        rng = np.random.default_rng()

    # Auto-promote x_0 onto transition.U.device. All subsequent ops run
    # on that device (CPU or GPU); CPU-only callers still work because
    # transition defaults to device='cpu'.
    target_device = transition.U.device
    if isinstance(x_0, torch.Tensor):
        x_0_t = x_0.to(device=target_device, dtype=torch.float64)
    else:
        x_0_t = torch.from_numpy(np.asarray(x_0, dtype=np.float64)).to(
            device=target_device,
        )
    if x_0_t.ndim != 2:
        raise ValueError(f"x_0 must be [batch, d], got shape {tuple(x_0_t.shape)}")
    batch, d = x_0_t.shape
    if d != transition.d:
        raise ValueError(
            f"x_0 dim {d} does not match transition.d={transition.d}"
        )

    # Auto-promote y_star and candidates if provided.
    if y_star is not None and isinstance(y_star, torch.Tensor):
        y_star = y_star.to(device=target_device, dtype=torch.float64)
    if candidates is not None and isinstance(candidates, torch.Tensor):
        candidates = candidates.to(device=target_device, dtype=torch.float64)

    # Materialize L_full once from cached eigendecomp (on target_device).
    L_full = (transition.U * transition.sigma.unsqueeze(0)) @ transition.U.T
    L_full = 0.5 * (L_full + L_full.T)

    dt = 1.0 / float(num_steps)
    sqrt_beta_dt = float(np.sqrt(transition.beta * dt))
    g_f = float(g)

    x_k = x_0_t.clone()

    if save_trajectories:
        trajectories = torch.empty(
            (batch, num_steps + 1, d), dtype=torch.float64, device=target_device,
        )
        trajectories[:, 0] = x_0_t
    else:
        trajectories = None

    if save_controls:
        controls = torch.empty(
            (batch, num_steps, d), dtype=torch.float64, device=target_device,
        )
    else:
        controls = None

    for k in range(num_steps):
        t_k = k * dt
        t_k_batch = torch.full(
            (batch,), t_k, dtype=torch.float64, device=target_device,
        )
        u_k = controller_fn(x_k, t_k_batch, y_star, candidates)
        # Promote u_k to target_device too if controller_fn returned on CPU.
        if u_k.device != target_device:
            u_k = u_k.to(device=target_device, dtype=torch.float64)

        drift = -transition.beta * (x_k @ L_full.T) + g_f * u_k

        if use_sde:
            # Reproducible noise: numpy rng -> CPU tensor -> target device.
            z_k = torch.from_numpy(
                rng.standard_normal((batch, d)).astype(np.float64),
            ).to(device=target_device)
            x_k = x_k + drift * dt + sqrt_beta_dt * z_k
        else:
            x_k = x_k + drift * dt

        if save_trajectories:
            trajectories[:, k + 1] = x_k
        if save_controls:
            controls[:, k] = u_k

    out: dict = {"x_1": x_k}
    if save_trajectories:
        out["trajectories"] = trajectories
    if save_controls:
        out["controls"] = controls
    return out
