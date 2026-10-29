import numpy as np


def _project_source(X, M1, cost_type):
    if cost_type in {"mahalanobis", "euclidean", "dynamics_mahalanobis", "dynamics_euclidean"}:
        if M1 is None:
            raise ValueError(f"M1 is required for cost_type='{cost_type}'")
        return np.asarray(X, dtype=np.float64) @ np.asarray(M1, dtype=np.float64).T
    if cost_type == "raw_euclidean":
        return np.asarray(X, dtype=np.float64)
    raise ValueError(f"Unsupported cost_type: {cost_type}")


def _compute_costs_against_all_targets(reference, Y, C1_inv, cost_type):
    diff = Y[None, :, :] - reference[:, None, :]
    if cost_type in {"mahalanobis", "dynamics_mahalanobis"}:
        if C1_inv is None:
            raise ValueError(f"C1_inv is required for cost_type='{cost_type}'")
        return np.einsum("bnd,dd,bnd->bn", diff, C1_inv, diff, optimize=True)
    if cost_type in {"euclidean", "dynamics_euclidean", "raw_euclidean"}:
        return np.einsum("bnd,bnd->bn", diff, diff, optimize=True)
    raise ValueError(f"Unsupported cost_type: {cost_type}")


def compute_pairwise_cost_matrix(
    X,
    Y,
    M1=None,
    C1_inv=None,
    cost_type="mahalanobis",
    chunk_size=128,
):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    projected = _project_source(X, M1, cost_type)
    costs = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)

    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        costs[start:end] = _compute_costs_against_all_targets(
            projected[start:end],
            Y,
            C1_inv,
            cost_type,
        )

    if not np.isfinite(costs).all():
        raise ValueError("Pairwise cost matrix contains NaN/Inf values")
    return costs


def topk_lowest(cost_matrix, topk):
    if topk <= 0:
        raise ValueError("topk must be positive")
    if topk > cost_matrix.shape[1]:
        raise ValueError(f"topk={topk} exceeds target size {cost_matrix.shape[1]}")

    idx_part = np.argpartition(cost_matrix, kth=topk - 1, axis=1)[:, :topk]
    cost_part = np.take_along_axis(cost_matrix, idx_part, axis=1)
    order = np.argsort(cost_part, axis=1)
    idx_sorted = np.take_along_axis(idx_part, order, axis=1)
    costs_sorted = np.take_along_axis(cost_part, order, axis=1)
    return idx_sorted.astype(np.int64), costs_sorted.astype(np.float64)


def topk_uncontrolled_candidates(
    X,
    Y_train,
    M1,
    C1_inv,
    topk,
    cost_type="mahalanobis",
    chunk_size=128,
):
    cost_matrix = compute_pairwise_cost_matrix(
        X,
        Y_train,
        M1=M1,
        C1_inv=C1_inv,
        cost_type=cost_type,
        chunk_size=chunk_size,
    )
    return topk_lowest(cost_matrix, topk)


def compute_selected_costs(
    X,
    Y_train,
    candidate_indices,
    M1,
    C1_inv,
    cost_type="mahalanobis",
    chunk_size=128,
):
    X = np.asarray(X, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    projected = _project_source(X, M1, cost_type)
    costs = np.empty(candidate_indices.shape, dtype=np.float64)

    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        reference = projected[start:end]
        target_block = Y_train[candidate_indices[start:end]]
        diff = target_block - reference[:, None, :]
        if cost_type in {"mahalanobis", "dynamics_mahalanobis"}:
            block_costs = np.einsum("bkd,dd,bkd->bk", diff, C1_inv, diff, optimize=True)
        else:
            block_costs = np.einsum("bkd,bkd->bk", diff, diff, optimize=True)
        costs[start:end] = block_costs

    if not np.isfinite(costs).all():
        raise ValueError("Selected candidate costs contain NaN/Inf values")
    return costs
