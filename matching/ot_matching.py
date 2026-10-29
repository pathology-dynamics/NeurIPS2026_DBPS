import numpy as np

from matching.unc_reachability import compute_pairwise_cost_matrix


def topk_highest(weight_matrix, topk):
    if topk <= 0:
        raise ValueError("topk must be positive")
    if topk > weight_matrix.shape[1]:
        raise ValueError(f"topk={topk} exceeds target size {weight_matrix.shape[1]}")

    idx_part = np.argpartition(-weight_matrix, kth=topk - 1, axis=1)[:, :topk]
    weight_part = np.take_along_axis(weight_matrix, idx_part, axis=1)
    order = np.argsort(-weight_part, axis=1)
    idx_sorted = np.take_along_axis(idx_part, order, axis=1)
    weights_sorted = np.take_along_axis(weight_part, order, axis=1)
    return idx_sorted.astype(np.int64), weights_sorted.astype(np.float64)


def _sinkhorn_numpy(a, b, cost_matrix, epsilon, max_iter, tol):
    scale = max(float(cost_matrix.max()), 1e-12)
    scaled_cost = cost_matrix / scale
    kernel = np.exp(-scaled_cost / epsilon)
    kernel = np.clip(kernel, 1e-300, None)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u.copy()
        Kv = kernel @ v
        u = a / np.clip(Kv, 1e-300, None)
        KTu = kernel.T @ u
        v = b / np.clip(KTu, 1e-300, None)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    coupling = (u[:, None] * kernel) * v[None, :]
    return coupling.astype(np.float64), scale


def solve_sinkhorn(a, b, cost_matrix, epsilon, max_iter, tol):
    try:
        import ot  # type: ignore
    except ModuleNotFoundError:
        ot = None

    if ot is not None:
        scale = max(float(cost_matrix.max()), 1e-12)
        scaled_cost = cost_matrix / scale
        coupling = ot.sinkhorn(
            a,
            b,
            scaled_cost,
            reg=float(epsilon),
            numItermax=int(max_iter),
            stopThr=float(tol),
            verbose=False,
        )
        return np.asarray(coupling, dtype=np.float64), scale

    return _sinkhorn_numpy(a, b, cost_matrix, epsilon, max_iter, tol)


def compute_ot_train_candidates(
    X_train,
    Y_train,
    M1,
    C1_inv,
    topk,
    ot_cost_type="dynamics_mahalanobis",
    epsilon=0.05,
    max_iter=500,
    tol=1e-7,
    chunk_size=128,
):
    cost_matrix = compute_pairwise_cost_matrix(
        X_train,
        Y_train,
        M1=M1,
        C1_inv=C1_inv,
        cost_type=ot_cost_type,
        chunk_size=chunk_size,
    )
    a = np.full(X_train.shape[0], 1.0 / X_train.shape[0], dtype=np.float64)
    b = np.full(Y_train.shape[0], 1.0 / Y_train.shape[0], dtype=np.float64)
    coupling, cost_scale = solve_sinkhorn(a, b, cost_matrix, epsilon, max_iter, tol)
    row_sums = np.clip(coupling.sum(axis=1, keepdims=True), 1e-300, None)
    row_probs = coupling / row_sums
    top_indices, top_weights = topk_highest(row_probs, topk)
    return {
        "cost_matrix": cost_matrix,
        "coupling": coupling,
        "row_probs": row_probs,
        "top_indices": top_indices,
        "top_weights": top_weights,
        "cost_scale": cost_scale,
    }


def _project_terminal(X, M1):
    return np.asarray(X, dtype=np.float64) @ np.asarray(M1, dtype=np.float64).T


def _squared_distance_matrix(A, B):
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1)[None, :]
    distances = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(distances, 0.0)


def transfer_ot_candidates(
    X_query,
    X_train,
    train_row_probs,
    M1,
    topk,
    knn_k=32,
    temperature=1.0,
    chunk_size=128,
):
    X_query_proj = _project_terminal(X_query, M1)
    X_train_proj = _project_terminal(X_train, M1)
    top_indices = np.empty((X_query_proj.shape[0], topk), dtype=np.int64)
    top_weights = np.empty((X_query_proj.shape[0], topk), dtype=np.float64)

    knn_k = min(int(knn_k), X_train_proj.shape[0])
    if knn_k <= 0:
        raise ValueError("knn_k must be positive")

    for start in range(0, X_query_proj.shape[0], chunk_size):
        end = min(start + chunk_size, X_query_proj.shape[0])
        dist_block = _squared_distance_matrix(X_query_proj[start:end], X_train_proj)
        nn_part = np.argpartition(dist_block, kth=knn_k - 1, axis=1)[:, :knn_k]
        nn_dist = np.take_along_axis(dist_block, nn_part, axis=1)
        order = np.argsort(nn_dist, axis=1)
        nn_idx = np.take_along_axis(nn_part, order, axis=1)
        nn_dist = np.take_along_axis(nn_dist, order, axis=1)

        shifted = nn_dist - nn_dist.min(axis=1, keepdims=True)
        logits = -shifted / max(float(temperature), 1e-12)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-300, None)
        row_probs = np.einsum("bk,bkd->bd", weights, train_row_probs[nn_idx])

        idx_block, weight_block = topk_highest(row_probs, topk)
        top_indices[start:end] = idx_block
        top_weights[start:end] = weight_block

    return top_indices, top_weights
