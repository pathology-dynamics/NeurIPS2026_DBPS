import glob
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch-free preprocessing envs
    torch = None


def _read_matrix(path):
    return pd.read_csv(path, header=None).values


def _validate_square_matrix(values, dim, path):
    if values.shape != (dim, dim):
        raise ValueError(
            f"Matrix {path} has shape {tuple(values.shape)}; expected {(dim, dim)}"
        )


def _load_connectomes_numpy(connectomes_glob, dim):
    conns = sorted(glob.glob(connectomes_glob))
    if not conns:
        raise FileNotFoundError(f"No connectome CSVs matched: {connectomes_glob}")

    matrices = np.zeros((len(conns), dim, dim), dtype=np.float64)
    for idx, conn in enumerate(conns):
        values = _read_matrix(conn)
        _validate_square_matrix(values, dim, conn)
        matrices[idx] = values
    return matrices


def get_adj_zero_numpy(connectomes_glob, dim):
    return _load_connectomes_numpy(connectomes_glob, dim).mean(axis=0)


def from_adj_to_laplacian_zero_numpy(adjacency, norm=True):
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    if not norm:
        return laplacian

    max_value = float(np.max(laplacian))
    if max_value <= 0:
        raise ValueError("Cannot normalize Laplacian with non-positive max value")
    return laplacian / max_value * 2.0


def get_laplacian_zero_numpy(connectomes_glob, dim, norm=True):
    adjacency = get_adj_zero_numpy(connectomes_glob, dim)
    return from_adj_to_laplacian_zero_numpy(adjacency, norm=norm)


def _load_precomputed_laplacian(matrix_path, dim):
    matrix_path = Path(matrix_path)
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Precomputed Laplacian not found: {matrix_path}")
    if matrix_path.suffix == ".npy":
        matrix = np.load(matrix_path)
    else:
        matrix = _read_matrix(matrix_path)
    _validate_square_matrix(matrix, dim, matrix_path)
    return np.asarray(matrix, dtype=np.float64)


def build_laplacian_matrix(
    mode,
    dim,
    connectomes_glob=None,
    matrix_path=None,
    norm=True,
):
    if mode == "mean_connectomes":
        if connectomes_glob is None:
            raise ValueError("connectomes_glob is required for laplacian_mode='mean_connectomes'")
        return get_laplacian_zero_numpy(connectomes_glob, dim, norm=norm)
    if mode == "precomputed_matrix":
        if matrix_path is None:
            raise ValueError("matrix_path is required for laplacian_mode='precomputed_matrix'")
        return _load_precomputed_laplacian(matrix_path, dim)
    raise ValueError(f"Unsupported laplacian_mode: {mode}")


def _load_connectomes_torch(connectomes_glob, dim):
    if torch is None:
        raise ModuleNotFoundError("torch is required for torch Laplacian utilities")

    matrices = _load_connectomes_numpy(connectomes_glob, dim)
    return torch.tensor(matrices, dtype=torch.float32)


def get_adj_zero(connectomes_dir, dim):
    return _load_connectomes_torch(connectomes_dir, dim).mean(dim=0)


def from_adj_to_laplacian_zero(A0, norm=True):
    if torch is None:
        raise ModuleNotFoundError("torch is required for torch Laplacian utilities")

    L0 = A0.sum(dim=1).diag() - A0
    if not norm:
        return L0

    max_value = L0.max()
    if max_value <= 0:
        raise ValueError("Cannot normalize Laplacian with non-positive max value")
    return L0 / max_value * 2


def get_laplacian_zero(connectomes_dir, dim, norm=True):
    return from_adj_to_laplacian_zero(get_adj_zero(connectomes_dir, dim), norm=norm)
