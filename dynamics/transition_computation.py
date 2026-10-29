"""Transition kernels of the reference graph diffusion dX = -beta L X dt + sqrt(beta) dW.

Two APIs:

- compute_transition_matrices(...) -- numpy function used by the
  preprocessing pipeline (scripts/preprocess_candidate_groups.py).
  Returns a dict {L, eigvecs, eigvals, M1, C1, C1_inv}.

- TransitionComputation -- torch.float64 class used by training,
  sampling, and rollout. Provides M(delta), C(delta), C_inv(delta),
  log_det_C(delta), log_p_t1, and sample_x_t_uncontrolled. The dense
  accessors M(), C(), C_inv(), log_det_C() take a scalar delta; the
  "apply" accessors M_apply / C_inv_apply / log_det_C_b, log_p_t1 and
  sample_x_t_uncontrolled also accept a per-sample tensor of deltas and
  work in the cached eigenbasis without forming [B, d, d] matrices.
"""
from typing import Optional, Union

import numpy as np
import torch

from utils.math_utils import stable_eigendecomposition


ArrayLike = Union[np.ndarray, torch.Tensor]


def symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def compute_transition_matrices(
    laplacian,
    beta,
    eig_eps=1e-8,
    cov_regularization=1e-6,
):
    laplacian = np.asarray(laplacian, dtype=np.float64)
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError(f"Laplacian must be square, got shape {laplacian.shape}")

    laplacian = symmetrize(laplacian)
    sigma, U = np.linalg.eigh(laplacian)
    sigma = np.clip(sigma, 0.0, None)

    exp_diag = np.exp(-beta * sigma)
    delta = np.full_like(sigma, float(beta), dtype=np.float64)
    positive_mask = sigma > eig_eps
    delta[positive_mask] = (
        1.0 - np.exp(-2.0 * beta * sigma[positive_mask])
    ) / (2.0 * sigma[positive_mask])

    M1 = (U * exp_diag) @ U.T
    C1 = (U * delta) @ U.T
    C1 = symmetrize(C1)
    C1_reg = C1 + cov_regularization * np.eye(C1.shape[0], dtype=np.float64)
    C1_inv = np.linalg.inv(C1_reg)

    outputs = {
        "L": laplacian,
        "eigvecs": U,
        "eigvals": sigma,
        "M1": symmetrize(M1),
        "C1": C1,
        "C1_inv": symmetrize(C1_inv),
    }
    for name in ("L", "M1", "C1", "C1_inv"):
        if not np.isfinite(outputs[name]).all():
            raise ValueError(f"{name} contains NaN/Inf values")
    return outputs


class TransitionComputation:
    """Holds eigendecomposition of L and provides M_delta, C_delta,
    C_delta_inv, log_p_t1, and uncontrolled-process sampling.

    All public methods accept and return torch.Tensor in float64.
    Inputs may be np.ndarray; convert internally.

    Eigendecomposition is computed once in __init__ and cached. All
    M_delta, C_delta, C_delta_inv computations use the cached eigvals
    and eigvecs (no per-call eig).

    NUMERICAL CARE for C_delta:
      d_i = (1 - exp(-2 beta sigma_i delta)) / (2 sigma_i)
    When sigma_i is below `eig_threshold`, use the linear approximation
    d_i = beta * delta. This is the limit as sigma_i -> 0.
    Implemented via torch.where on a per-eigenvalue basis.

    NUMERICAL CARE for C_delta_inv:
      Add cov_regularization to each eigenvalue d_i before reciprocating:
          C_inv = U diag(1 / (d_i + cov_reg)) U^T
      Numerically equivalent to (C + cov_reg*I)^-1 but avoids forming
      the dense matrix and re-decomposing.

    NUMERICAL CARE for log_p_t1:
      log p(y | x_t, t) =
          -0.5 * (d * log(2 pi)
                  + log_det_C_{1-t}
                  + (y - M_{1-t} x_t)^T C_inv_{1-t} (y - M_{1-t} x_t))
      Compute log_det_C from sum of log(d_i + cov_reg) using cached
      eigenvalues -- do NOT call torch.logdet on the full matrix.
    """

    def __init__(
        self,
        L: ArrayLike,
        beta: float,
        cov_regularization: float = 1e-6,
        eig_threshold: float = 1e-8,
        device: Union[str, torch.device, None] = None,
    ):
        """Cache eigendecomp of L on the requested device.

        device: 'cpu', 'cuda', a torch.device, or None for CPU. The eig
        is always computed on CPU (LAPACK), then U / sigma are moved to
        device. Subsequent matmuls / log_p_t1 / sample_x_t_uncontrolled
        run on whatever device U lives on; inputs are auto-promoted to
        that device so callers can pass CPU OR GPU tensors freely.
        """
        if isinstance(L, torch.Tensor):
            L_t = L.detach().to(dtype=torch.float64, device="cpu")
        else:
            L_t = torch.from_numpy(np.asarray(L, dtype=np.float64))

        if L_t.ndim != 2 or L_t.shape[0] != L_t.shape[1]:
            raise ValueError(f"L must be a square matrix, got shape {tuple(L_t.shape)}")

        sigma, U = stable_eigendecomposition(L_t, regularization=0.0)
        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.U: torch.Tensor = U.to(device=target_device)
        self.sigma: torch.Tensor = sigma.to(device=target_device)
        self.beta: float = float(beta)

        self.cov_reg: float = float(cov_regularization)
        self.eig_threshold: float = float(eig_threshold)
        self.d: int = int(self.U.shape[0])

    @property
    def device(self) -> torch.device:
        return self.U.device

    def to(self, device) -> "TransitionComputation":
        """Move U / sigma to device, in place. Returns self."""
        target = torch.device(device)
        self.U = self.U.to(device=target)
        self.sigma = self.sigma.to(device=target)
        return self

    # ------------------------------------------------------------------
    # Helpers (eigenbasis machinery used by both scalar and per-batch t)
    # ------------------------------------------------------------------

    def _delta_to_tensor(self, delta) -> torch.Tensor:
        """Convert delta to a float64 tensor on self.device."""
        if isinstance(delta, torch.Tensor):
            return delta.to(device=self.U.device, dtype=torch.float64)
        if isinstance(delta, np.ndarray):
            return torch.from_numpy(np.asarray(delta, dtype=np.float64)).to(
                device=self.U.device,
            )
        return torch.tensor(float(delta), dtype=torch.float64, device=self.U.device)

    def _to_device(self, x: ArrayLike) -> torch.Tensor:
        """Auto-promote any input to torch.float64 on self.device."""
        if isinstance(x, torch.Tensor):
            return x.to(device=self.U.device, dtype=torch.float64)
        return torch.from_numpy(np.asarray(x, dtype=np.float64)).to(
            device=self.U.device,
        )

    def _d_eigvals(self, delta) -> torch.Tensor:
        """Per-eigenvalue covariance scalars d_i at delta.

        Returns shape:
          [d]            if delta is scalar (0-d tensor)
          [..., d]       if delta has shape [...] (e.g. [B] -> [B, d])

        Uses the analytic formula for sigma_i > eig_threshold and the
        linear limit beta * delta otherwise.
        """
        delta_t = self._delta_to_tensor(delta)
        sigma = self.sigma  # [d]
        sigma_safe = torch.where(
            sigma > self.eig_threshold, sigma, torch.ones_like(sigma),
        )
        if delta_t.ndim == 0:
            delta_f = float(delta_t.item())
            formula = (
                1.0 - torch.exp(-2.0 * self.beta * sigma * delta_f)
            ) / (2.0 * sigma_safe)
            linear_limit = torch.full_like(sigma, self.beta * delta_f)
            return torch.where(sigma > self.eig_threshold, formula, linear_limit)
        # Batched: insert d-axis on delta, broadcast against sigma.
        delta_b = delta_t.unsqueeze(-1)  # [..., 1]
        formula = (
            1.0 - torch.exp(-2.0 * self.beta * sigma * delta_b)
        ) / (2.0 * sigma_safe)
        linear_limit = (self.beta * delta_b).expand_as(formula)
        mask = (sigma > self.eig_threshold).expand_as(formula)
        return torch.where(mask, formula, linear_limit)

    def _apply_diag_in_eigbasis(
        self, eigvals: torch.Tensor, x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (U diag(eigvals) U^T) @ x via eigenbasis projection.

        eigvals:
          [d]            shared across all batch elements (scalar t)
          [..., d]       per-batch (broadcast leading dims with x)

        x: [..., d] (last axis must equal d)

        Returns: same shape as x.
        """
        x = self._to_device(x)                              # auto-promote
        proj = x @ self.U                                   # [..., d]
        if eigvals.ndim == 1:
            scaled = proj * eigvals
        else:
            # eigvals leading dims should match x's leading dims up to
            # a possible inserted [K] axis. If eigvals has fewer dims
            # than proj (e.g. [B, d] vs proj [B, K, d]), insert axes
            # before the last to broadcast.
            ev = eigvals
            while ev.ndim < proj.ndim:
                ev = ev.unsqueeze(-2)
            scaled = proj * ev
        return scaled @ self.U.T

    # ------------------------------------------------------------------
    # Public dense-matrix accessors (scalar delta only)
    # ------------------------------------------------------------------

    def M(self, delta: float) -> torch.Tensor:
        """M_delta = U diag(exp(-beta * sigma_i * delta)) U^T as a dense
        [d, d] matrix. Scalar delta only; for per-batch t use M_apply."""
        delta_f = float(delta)
        exp_diag = torch.exp(-self.beta * self.sigma * delta_f)
        M_delta = (self.U * exp_diag.unsqueeze(0)) @ self.U.T
        return 0.5 * (M_delta + M_delta.T)

    def C(self, delta: float) -> torch.Tensor:
        """C_delta = U diag(d_i) U^T. Scalar delta only."""
        d_i = self._d_eigvals(float(delta))
        C_delta = (self.U * d_i.unsqueeze(0)) @ self.U.T
        return 0.5 * (C_delta + C_delta.T)

    def C_inv(self, delta: float) -> torch.Tensor:
        """Inverse of (C_delta + cov_reg * I). Scalar delta only."""
        d_i = self._d_eigvals(float(delta))
        inv_d = 1.0 / (d_i + self.cov_reg)
        C_inv_delta = (self.U * inv_d.unsqueeze(0)) @ self.U.T
        return 0.5 * (C_inv_delta + C_inv_delta.T)

    def log_det_C(self, delta: float) -> torch.Tensor:
        """log det(C_delta + cov_reg I), scalar delta only."""
        d_i = self._d_eigvals(float(delta))
        return torch.sum(torch.log(d_i + self.cov_reg))

    # ------------------------------------------------------------------
    # Per-batch-t friendly "apply" accessors (eigenbasis fast path)
    # ------------------------------------------------------------------

    def M_apply(self, delta, x: torch.Tensor) -> torch.Tensor:
        """Apply M_delta to x without materializing M as [d, d].

        delta: scalar (Python float / 0-d tensor) or [...] tensor with
               leading dims matching x.
        x: [..., d]
        Returns: same shape as x.

        Math: result = (U diag(exp(-beta sigma delta)) U^T) @ x
        Uses M = M^T (symmetric) so we don't need a transpose.
        """
        delta_t = self._delta_to_tensor(delta)
        if delta_t.ndim == 0:
            eigvals = torch.exp(-self.beta * self.sigma * float(delta_t.item()))
        else:
            eigvals = torch.exp(-self.beta * self.sigma * delta_t.unsqueeze(-1))
        return self._apply_diag_in_eigbasis(eigvals, x)

    def C_inv_apply(self, delta, x: torch.Tensor) -> torch.Tensor:
        """Apply (C_delta + cov_reg I)^-1 to x via eigenbasis."""
        d_i = self._d_eigvals(delta)
        inv_d = 1.0 / (d_i + self.cov_reg)
        return self._apply_diag_in_eigbasis(inv_d, x)

    def log_det_C_b(self, delta) -> torch.Tensor:
        """log det(C_delta + cov_reg I) batched.

        delta: scalar -> returns scalar tensor.
               [...]  -> returns [...] tensor.
        """
        d_i = self._d_eigvals(delta)
        return torch.sum(torch.log(d_i + self.cov_reg), dim=-1)

    def sqrt_C_apply(self, delta, x: torch.Tensor) -> torch.Tensor:
        """Apply sqrt(C_delta + cov_reg I) to x via eigenbasis."""
        d_i = self._d_eigvals(delta)
        sqrt_eigvals = torch.sqrt(d_i + self.cov_reg)
        return self._apply_diag_in_eigbasis(sqrt_eigvals, x)

    # ------------------------------------------------------------------
    # log_p_t1 and sampling (now per-batch-t friendly)
    # ------------------------------------------------------------------

    def log_p_t1(
        self,
        y: ArrayLike,
        x_t: ArrayLike,
        t: Union[float, ArrayLike],
    ) -> torch.Tensor:
        """Log Gaussian density of X_1 | X_t = x_t evaluated at y.

        Mean = M_{1-t} x_t, Cov = C_{1-t} + cov_reg * I.

        Supports:
          x_t: [d] or [batch, d]
          y:   [d] or [batch, d] or [batch, K, d]
          t:   scalar (Python float / 0-d tensor) OR [batch] tensor

        Returns torch.float64 with shape:
          ()           if x_t was [d] and y was [d] and t scalar
          [batch]      if y is [d] or [batch, d]
          [batch, K]   if y is [batch, K, d]
        """
        t_t = self._delta_to_tensor(t)
        delta = 1.0 - t_t  # scalar or [...]

        x_t_tensor = self._to_device(x_t)
        y_tensor = self._to_device(y)

        x_was_1d = x_t_tensor.ndim == 1
        if x_was_1d:
            x_t_tensor = x_t_tensor.unsqueeze(0)
        if x_t_tensor.ndim != 2 or x_t_tensor.shape[1] != self.d:
            raise ValueError(
                f"x_t shape {tuple(x_t_tensor.shape)} incompatible with d={self.d}"
            )

        mean = self.M_apply(delta, x_t_tensor)               # [batch, d]

        if y_tensor.ndim == 1:
            diff = y_tensor.unsqueeze(0) - mean              # [batch, d]
        elif y_tensor.ndim == 2:
            diff = y_tensor - mean                           # [batch, d]
        elif y_tensor.ndim == 3:
            diff = y_tensor - mean.unsqueeze(1)              # [batch, K, d]
        else:
            raise ValueError(f"y must have 1, 2, or 3 dims, got {y_tensor.ndim}")

        C_inv_diff = self.C_inv_apply(delta, diff)
        if diff.ndim == 2:
            maha = torch.einsum("bd,bd->b", diff, C_inv_diff)
        else:
            maha = torch.einsum("bkd,bkd->bk", diff, C_inv_diff)

        log_det = self.log_det_C_b(delta)                    # scalar or [batch]
        if maha.ndim == 2 and log_det.ndim == 1:
            log_det = log_det.unsqueeze(-1)                  # [batch, 1]

        log_p = -0.5 * (
            self.d * float(np.log(2.0 * np.pi)) + log_det + maha
        )

        if x_was_1d and y_tensor.ndim == 1 and t_t.ndim == 0:
            return log_p.reshape(())
        return log_p

    def sample_x_t_uncontrolled(
        self,
        x_0: ArrayLike,
        t: Union[float, ArrayLike],
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        """Sample X_t under the uncontrolled SDE given X_0:
              X_t | X_0  ~  N(M_t x_0, C_t + cov_reg * I)

        Supports:
          x_0: [d] or [batch, d]
          t:   scalar OR [batch] tensor

        Implementation: all matmuls done in the cached eigenbasis.
        """
        if rng is None:
            rng = np.random.default_rng()

        x_0_t = self._to_device(x_0)
        if x_0_t.ndim == 1:
            x_0_t = x_0_t.unsqueeze(0)
        if x_0_t.ndim != 2 or x_0_t.shape[1] != self.d:
            raise ValueError(
                f"x_0 shape {tuple(x_0_t.shape)} incompatible with d={self.d}"
            )
        batch = x_0_t.shape[0]

        mean = self.M_apply(t, x_0_t)                        # [batch, d]

        # Noise: draw on CPU via numpy rng for reproducibility, then
        # move to self.device.
        z = torch.from_numpy(
            rng.standard_normal((batch, self.d)).astype(np.float64),
        ).to(device=self.U.device)
        noise = self.sqrt_C_apply(t, z)

        return mean + noise
