import numpy as np
import pandas as pd


UNCONTROLLED_LABEL = 0
OT_LABEL = 1
RANDOM_LABEL = 2


def _row_rng(seed, split_name, row_index):
    split_offset = {"train": 0, "val": 1, "test": 2}[split_name]
    return np.random.default_rng(seed + 100_000 * split_offset + row_index)


def build_candidate_groups(
    split_name,
    sample_ids,
    y_train_size,
    uncontrolled_top_indices,
    uncontrolled_top_costs,
    ot_top_indices,
    ot_top_weights,
    candidate_group_size,
    num_uncontrolled,
    num_ot,
    num_random,
    seed,
):
    n_rows = uncontrolled_top_indices.shape[0]
    candidate_indices = np.empty((n_rows, candidate_group_size), dtype=np.int64)
    source_labels = np.empty((n_rows, candidate_group_size), dtype=np.int64)
    candidate_uncontrolled_costs = np.full(
        (n_rows, candidate_group_size), np.nan, dtype=np.float64
    )
    candidate_ot_weights = np.full(
        (n_rows, candidate_group_size), np.nan, dtype=np.float64
    )
    random_indices = np.full((n_rows, num_random), -1, dtype=np.int64)

    summary_rows = []
    warnings = []
    overlap_counts = []

    all_targets = np.arange(y_train_size, dtype=np.int64)
    for row_idx in range(n_rows):
        rng = _row_rng(seed, split_name, row_idx)
        selected = []
        labels = []
        unc_scores = []
        ot_scores = []
        seen = set()
        ot_overlap = 0

        for idx, cost in zip(
            uncontrolled_top_indices[row_idx, :num_uncontrolled],
            uncontrolled_top_costs[row_idx, :num_uncontrolled],
        ):
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
            labels.append(UNCONTROLLED_LABEL)
            unc_scores.append(float(cost))
            ot_scores.append(np.nan)

        for idx, weight in zip(
            ot_top_indices[row_idx, :num_ot],
            ot_top_weights[row_idx, :num_ot],
        ):
            idx = int(idx)
            if idx in seen:
                ot_overlap += 1
                continue
            seen.add(idx)
            selected.append(idx)
            labels.append(OT_LABEL)
            unc_scores.append(np.nan)
            ot_scores.append(float(weight))

        overlap_counts.append(ot_overlap)

        base_available = all_targets[~np.isin(all_targets, np.asarray(selected, dtype=np.int64))]
        if base_available.size >= num_random:
            base_random = rng.choice(base_available, size=num_random, replace=False)
        elif base_available.size > 0:
            warnings.append(
                f"{split_name}: row {row_idx} had only {base_available.size} unselected targets; "
                "sampling extra random candidates with replacement."
            )
            extra = rng.choice(base_available, size=num_random - base_available.size, replace=True)
            base_random = np.concatenate([base_available, extra])
        else:
            warnings.append(
                f"{split_name}: row {row_idx} had no unselected targets left; random candidates sampled with replacement from all Y_train."
            )
            base_random = rng.choice(all_targets, size=num_random, replace=True)

        random_indices[row_idx] = base_random
        for idx in base_random.tolist():
            idx = int(idx)
            selected.append(idx)
            labels.append(RANDOM_LABEL)
            unc_scores.append(np.nan)
            ot_scores.append(np.nan)

        extra_random = 0
        while len(selected) < candidate_group_size:
            remaining = all_targets[~np.isin(all_targets, np.asarray(selected, dtype=np.int64))]
            if remaining.size > 0:
                idx = int(rng.choice(remaining))
            else:
                idx = int(rng.choice(all_targets))
                warnings.append(
                    f"{split_name}: row {row_idx} required replacement sampling to reach {candidate_group_size} candidates."
                )
            selected.append(idx)
            labels.append(RANDOM_LABEL)
            unc_scores.append(np.nan)
            ot_scores.append(np.nan)
            extra_random += 1

        candidate_indices[row_idx] = np.asarray(selected[:candidate_group_size], dtype=np.int64)
        source_labels[row_idx] = np.asarray(labels[:candidate_group_size], dtype=np.int64)
        candidate_uncontrolled_costs[row_idx] = np.asarray(
            unc_scores[:candidate_group_size], dtype=np.float64
        )
        candidate_ot_weights[row_idx] = np.asarray(
            ot_scores[:candidate_group_size], dtype=np.float64
        )

        unc_mask = source_labels[row_idx] == UNCONTROLLED_LABEL
        ot_mask = source_labels[row_idx] == OT_LABEL
        random_mask = source_labels[row_idx] == RANDOM_LABEL
        unc_vals = candidate_uncontrolled_costs[row_idx, unc_mask]
        ot_vals = candidate_ot_weights[row_idx, ot_mask]

        summary_rows.append(
            {
                "sample_id": sample_ids[row_idx],
                "split": split_name,
                "num_candidates": int(candidate_group_size),
                "num_uncontrolled": int(unc_mask.sum()),
                "num_ot": int(ot_mask.sum()),
                "num_random": int(random_mask.sum()),
                "duplicate_count_before_random": int(ot_overlap),
                "extra_random_due_to_duplicates": int(extra_random),
                "mean_uncontrolled_cost_selected": float(np.mean(unc_vals)) if unc_vals.size else np.nan,
                "min_uncontrolled_cost_selected": float(np.min(unc_vals)) if unc_vals.size else np.nan,
                "max_uncontrolled_cost_selected": float(np.max(unc_vals)) if unc_vals.size else np.nan,
                "mean_ot_weight_selected": float(np.mean(ot_vals)) if ot_vals.size else np.nan,
                "max_ot_weight_selected": float(np.max(ot_vals)) if ot_vals.size else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return {
        "candidate_indices": candidate_indices,
        "source_labels": source_labels,
        "candidate_uncontrolled_costs": candidate_uncontrolled_costs,
        "candidate_ot_weights": candidate_ot_weights,
        "random_indices": random_indices,
        "summary_df": summary_df,
        "warnings": warnings,
        "overlap_counts": np.asarray(overlap_counts, dtype=np.int64),
    }
