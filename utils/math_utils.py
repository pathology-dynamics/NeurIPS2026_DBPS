"""Numerically stable linear algebra utilities, backend-agnostic.

Functions accept either np.ndarray or torch.Tensor and return the same
backend. Internal numerics use float64 for stability and the output is
cast back to the input dtype.
"""
from typing import Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def stable_eigendecomposition(
    M: ArrayLike, regularization: float = 1e-6
) -> Tuple[ArrayLike, ArrayLike]:
    """Symmetrize M via (M + M.T) / 2, then eigendecompose.

    Eigenvalues are clipped to >= 0 (small negative values from
    floating-point noise are treated as 0). Then add `regularization`
    to all eigenvalues to ensure strict positivity for downstream use
    (e.g. matrix sqrt, matrix inverse).

    Internal computation uses float64 for stability. Output is cast
    back to the input dtype.

    Args:
      M: shape [d, d], np.ndarray or torch.Tensor.
      regularization: small positive scalar added to eigenvalues.

    Returns:
      eigvals: shape [d], same backend as input.
      eigvecs: shape [d, d], columns are eigenvectors, same backend.

    Use np.linalg.eigh (numpy) or torch.linalg.eigh (torch). We never
    use np.linalg.eig / torch.linalg.eig (non-Hermitian); the symmetric
    structure here is guaranteed.
    """
    if _is_torch(M):
        in_dtype = M.dtype
        in_device = M.device
        M64 = M.to(dtype=torch.float64)
        M_sym = 0.5 * (M64 + M64.T)
        eigvals, eigvecs = torch.linalg.eigh(M_sym)
        eigvals = torch.clamp(eigvals, min=0.0) + float(regularization)
        return (
            eigvals.to(dtype=in_dtype, device=in_device),
            eigvecs.to(dtype=in_dtype, device=in_device),
        )

    M_arr = np.asarray(M)
    in_dtype = M_arr.dtype
    M64 = M_arr.astype(np.float64)
    M_sym = 0.5 * (M64 + M64.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.clip(eigvals, 0.0, None) + float(regularization)
    return eigvals.astype(in_dtype), eigvecs.astype(in_dtype)


def stable_matrix_sqrt(M: ArrayLike, regularization: float = 1e-6) -> ArrayLike:
    """Compute M^{1/2} via stable_eigendecomposition.

    Returns a real symmetric matrix. The returned matrix R satisfies
    R @ R approx symmetrize(M) + regularization * I, to within float64
    precision.

    PERMANENT WARNING: This function REPLACES scipy's matrix-square-root
    routine. That scipy routine produces complex outputs for matrices
    with negative eigenvalues from floating-point noise -- a documented
    instability source in this project's Frechet Distance computation.
    NEVER import that scipy routine anywhere in this codebase; always
    use stable_matrix_sqrt instead.

    Args:
      M: shape [d, d], np.ndarray or torch.Tensor.
      regularization: passed through to stable_eigendecomposition.

    Returns:
      shape [d, d], same backend as input.
    """
    if _is_torch(M):
        in_dtype = M.dtype
        in_device = M.device
        M64 = M.to(dtype=torch.float64)
        eigvals, eigvecs = stable_eigendecomposition(M64, regularization)
        sqrt_eigvals = torch.sqrt(eigvals)
        R64 = (eigvecs * sqrt_eigvals.unsqueeze(0)) @ eigvecs.T
        R64 = 0.5 * (R64 + R64.T)
        return R64.to(dtype=in_dtype, device=in_device)

    M_arr = np.asarray(M)
    in_dtype = M_arr.dtype
    M64 = M_arr.astype(np.float64)
    eigvals, eigvecs = stable_eigendecomposition(M64, regularization)
    sqrt_eigvals = np.sqrt(eigvals)
    R64 = (eigvecs * sqrt_eigvals[None, :]) @ eigvecs.T
    R64 = 0.5 * (R64 + R64.T)
    return R64.astype(in_dtype)


def mahalanobis_distance_sq(
    x: ArrayLike, y: ArrayLike, A_inv: ArrayLike
) -> ArrayLike:
    """Compute (x - y)^T A_inv (x - y).

    Supports broadcasting:
      x:     [d]            or [batch, d]            or [batch, K, d]
      y:     [d]            or [batch, d]            or [batch, K, d]
      A_inv: [d, d]         (always)

    Returns:
      scalar, [batch], or [batch, K] correspondingly -- the contracted
      Mahalanobis squared distance.

    Backend-agnostic: detect numpy vs torch from input type of x and
    dispatch. The contraction is expressed as a single einsum so no
    large intermediate batched matrix is materialized.
    """
    if _is_torch(x):
        diff = x - y
        return torch.einsum("...d,de,...e->...", diff, A_inv, diff)

    diff = x - y
    return np.einsum("...d,de,...e->...", diff, A_inv, diff)
