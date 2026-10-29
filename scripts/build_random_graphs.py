"""Randomized anatomical graphs for the graph-perturbation control (Table 13, Appendix I.3).

Permuted: L_perm = P L P^T for a random permutation P. Spectrum, edge-weight
multiset and degree structure are unchanged; only the ROI <-> graph-node
correspondence is destroyed.

Rewired: the connectome's core subgraph is the complete graph on 82 nodes
(plus two isolated cerebellar nodes), so no degree-preserving double-edge swap
can be accepted. The randomization therefore reassigns the original edge
weights at random over the unchanged support (isolated nodes stay isolated).

Both use the combinatorial Laplacian L = D - A, like the anatomical graph.
For every graph the source processed_data.npz is copied with L, M1, C1 and
C1_inv replaced (recomputed by compute_transition_matrices); models are then
trained and evaluated on these files with --data.

usage:
  python -m scripts.build_random_graphs \
      --src data/processed/candidate_groups/adni/normalization/processed_data.npz --out_dir data/random_graphs
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dynamics.transition_computation import compute_transition_matrices

GRAPH_SEEDS = [101, 202, 303, 404, 505]
BETA = 0.2                      # configs.base beta
COV_REG = 1e-6                  # configs.base cov_regularization
SWAP_ATTEMPT_FACTOR = 20        # attempted swaps = factor * n_edges


def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def _components(adj_bool: np.ndarray) -> list[int]:
    n = adj_bool.shape[0]
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, c = [s], 0
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            c += 1
            stack.extend(np.where(adj_bool[u])[0].tolist())
        comps.append(c)
    return sorted(comps, reverse=True)


def _lap_from_adj(A: np.ndarray) -> np.ndarray:
    """Combinatorial L = D - A, matching the anatomical convention."""
    L = -A.copy()
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -L.sum(1))
    return 0.5 * (L + L.T)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/candidate_groups/adni/normalization/processed_data.npz")
    ap.add_argument("--out_dir", default="data/random_graphs")
    args = ap.parse_args()
    SRC, OUT = Path(args.src), Path(args.out_dir)
    d = np.load(SRC)
    L0 = np.asarray(d["L"], dtype=np.float64)
    n = L0.shape[0]
    A0 = -L0.copy()
    np.fill_diagonal(A0, 0.0)
    deg_w0 = A0.sum(1)
    iso = np.where(deg_w0 <= 1e-12)[0]
    core = np.array([i for i in range(n) if i not in set(iso.tolist())])
    iu = np.triu_indices(n, 1)
    E0 = (A0 != 0.0)
    w0_sorted = np.sort(A0[iu][E0[iu]])
    ev0 = np.sort(np.linalg.eigvalsh(L0))

    OUT.mkdir(parents=True, exist_ok=True)
    checks = {
        "source_npz": str(SRC),
        "normalisation": "combinatorial L = D - A (same as the anatomical graph)",
        "beta": BETA, "cov_regularization": COV_REG,
        "anatomical": {
            "n_nodes": int(n),
            "n_edges_undirected": int(E0[iu].sum()),
            "isolated_nodes": [int(i) for i in iso],
            "isolated_roi_names": [str(d["roi_names"][i]) for i in iso],
            "core_nodes": int(len(core)),
            "core_is_complete_graph": bool(
                E0[np.ix_(core, core)][np.triu_indices(len(core), 1)].all()),
            "components": _components(E0),
            "unweighted_degree_min_max": [int(E0.sum(1).min()), int(E0.sum(1).max())],
            "L_hash": _hash(L0),
            "eigenvalue_min_max": [float(ev0.min()), float(ev0.max())],
        },
        "permuted": [], "rewired": [],
    }
    print("anatomical: %d nodes, %d undirected edges, core K_%d complete=%s, components=%s"
          % (n, E0[iu].sum(), len(core),
             checks["anatomical"]["core_is_complete_graph"], _components(E0)))

    perms = {}

    # ---------------- Condition B: permuted ----------------
    for seed in GRAPH_SEEDS:
        rng = np.random.default_rng(seed)
        p = rng.permutation(n)
        Lp = L0[np.ix_(p, p)]
        Ap = -Lp.copy()
        np.fill_diagonal(Ap, 0.0)
        evp = np.sort(np.linalg.eigvalsh(Lp))
        Ep = (Ap != 0.0)
        rec = {
            "graph_seed": seed,
            "permutation_vector": [int(x) for x in p],
            "is_identity": bool(np.array_equal(p, np.arange(n))),
            "n_fixed_points": int((p == np.arange(n)).sum()),
            "eigenvalues_match": bool(np.allclose(evp, ev0, atol=1e-10, rtol=0)),
            "eigenvalue_max_abs_diff": float(np.abs(evp - ev0).max()),
            "nonzero_count_match": bool(int(Ep[iu].sum()) == int(E0[iu].sum())),
            "n_edges_undirected": int(Ep[iu].sum()),
            "symmetric": bool(np.abs(Lp - Lp.T).max() == 0.0),
            "edge_weight_multiset_match": bool(np.allclose(
                np.sort(Ap[iu][Ep[iu]]), w0_sorted, atol=0, rtol=0)),
            "unweighted_degree_multiset_match": bool(np.array_equal(
                np.sort(Ep.sum(1)), np.sort(E0.sum(1)))),
            "weighted_degree_multiset_match": bool(np.allclose(
                np.sort(Ap.sum(1)), np.sort(deg_w0), atol=1e-12)),
            "L_hash": _hash(Lp),
        }
        perms[seed] = p
        checks["permuted"].append(rec)
        _write_npz(d, Lp, OUT / f"permuted_seed{seed}.npz")
        print("  permuted seed %d: eig_diff=%.2e nnz_ok=%s fixed_pts=%d hash=%s"
              % (seed, rec["eigenvalue_max_abs_diff"], rec["nonzero_count_match"],
                 rec["n_fixed_points"], rec["L_hash"]))

    pairs = [(a, b) for i, a in enumerate(GRAPH_SEEDS) for b in GRAPH_SEEDS[i + 1:]]
    checks["permuted_pairwise_distinct"] = {
        f"{a}_vs_{b}": bool(not np.array_equal(perms[a], perms[b])) for a, b in pairs}
    print("  permutations pairwise distinct: %s"
          % all(checks["permuted_pairwise_distinct"].values()))

    # ---------------- Condition C: rewired ----------------
    core_pairs = [(int(core[i]), int(core[j]))
                  for i in range(len(core)) for j in range(i + 1, len(core))]
    for seed in GRAPH_SEEDS:
        rng = np.random.default_rng(seed)

        # (i) double-edge swaps on the binary support -- provably 0 successes on K_82
        attempted = SWAP_ATTEMPT_FACTOR * len(core_pairs)
        successful = 0
        Ecur = E0.copy()
        for _ in range(min(attempted, 200000)):
            (a, b) = core_pairs[rng.integers(len(core_pairs))]
            (c, e) = core_pairs[rng.integers(len(core_pairs))]
            if len({a, b, c, e}) < 4:
                continue
            if Ecur[a, e] or Ecur[c, b]:
                continue                       # target already an edge -> reject
            Ecur[a, b] = Ecur[b, a] = False
            Ecur[c, e] = Ecur[e, c] = False
            Ecur[a, e] = Ecur[e, a] = True
            Ecur[c, b] = Ecur[b, c] = True
            successful += 1

        # (ii) random reassignment of the original edge-weight multiset
        w = A0[iu][E0[iu]].copy()
        rng.shuffle(w)
        Ar = np.zeros((n, n), dtype=np.float64)
        for (i, j), val in zip(core_pairs, w):
            Ar[i, j] = Ar[j, i] = val
        Lr = _lap_from_adj(Ar)
        Er = (Ar != 0.0)
        evr = np.sort(np.linalg.eigvalsh(Lr))

        rec = {
            "graph_seed": seed,
            "attempted_swaps": int(min(attempted, 200000)),
            "successful_swaps": int(successful),
            "swap_note": ("core subgraph is the complete graph K_%d, so every "
                          "degree-preserving double-edge swap target already exists "
                          "as an edge and no swap can be accepted; randomisation is "
                          "carried entirely by the edge-weight reassignment"
                          % len(core)),
            "symmetric": bool(np.abs(Ar - Ar.T).max() == 0.0),
            "no_self_loops": bool(np.diag(Ar).max() == 0.0 and np.diag(Ar).min() == 0.0),
            "n_edges_undirected": int(Er[iu].sum()),
            "edge_count_match": bool(int(Er[iu].sum()) == int(E0[iu].sum())),
            "unweighted_degree_sequence_match": bool(np.array_equal(
                Er.sum(1), E0.sum(1))),
            "edge_weight_multiset_match": bool(np.array_equal(
                np.sort(Ar[iu][Er[iu]]), w0_sorted)),
            "isolated_nodes_preserved": [int(i) for i in np.where(Ar.sum(1) <= 1e-12)[0]],
            "components": _components(Er),
            "core_connected": bool(_components(Er)[0] == len(core)),
            "weighted_degree_preserved": bool(np.allclose(Ar.sum(1), deg_w0, atol=1e-12)),
            "weighted_degree_corr_with_anatomical": float(
                np.corrcoef(Ar.sum(1)[core], deg_w0[core])[0, 1]),
            "edge_weight_corr_with_anatomical": float(
                np.corrcoef(Ar[iu][E0[iu]], A0[iu][E0[iu]])[0, 1]),
            "eigenvalue_max_abs_diff_vs_anatomical": float(np.abs(evr - ev0).max()),
            "L_hash": _hash(Lr),
        }
        checks["rewired"].append(rec)
        _write_npz(d, Lr, OUT / f"rewired_seed{seed}.npz")
        print("  rewired  seed %d: swaps %d/%d  wdeg_preserved=%s  w_corr=%+.4f  "
              "eig_shift=%.4f  hash=%s"
              % (seed, successful, rec["attempted_swaps"],
                 rec["weighted_degree_preserved"],
                 rec["edge_weight_corr_with_anatomical"],
                 rec["eigenvalue_max_abs_diff_vs_anatomical"], rec["L_hash"]))

    hashes = [r["L_hash"] for r in checks["permuted"]] + \
             [r["L_hash"] for r in checks["rewired"]] + [checks["anatomical"]["L_hash"]]
    checks["all_L_hashes_distinct"] = bool(len(set(hashes)) == len(hashes))

    (OUT / "graph_randomization_checks.json").write_text(json.dumps(checks, indent=2))
    print("\nwrote %s" % (OUT / "graph_randomization_checks.json"))
    print("all 11 L hashes distinct: %s" % checks["all_L_hashes_distinct"])


def _write_npz(src, L_new, path: Path):
    """Copy the source npz verbatim, replacing ONLY L / M1 / C1 / C1_inv."""
    tr = compute_transition_matrices(L_new, beta=BETA, cov_regularization=COV_REG)
    out = {k: src[k] for k in src.files}
    out["L"] = tr["L"]
    out["M1"] = tr["M1"]
    out["C1"] = tr["C1"]
    out["C1_inv"] = tr["C1_inv"]
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    main()
