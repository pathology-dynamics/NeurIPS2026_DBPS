import argparse
import importlib.util
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.laplacian import build_laplacian_matrix
from dataset.roi_mapping import map_region_columns
from dynamics.transition_computation import compute_transition_matrices
from matching.candidate_groups import (
    OT_LABEL,
    RANDOM_LABEL,
    UNCONTROLLED_LABEL,
    build_candidate_groups,
)
from matching.ot_matching import compute_ot_train_candidates, transfer_ot_candidates
from matching.unc_reachability import compute_selected_costs, topk_uncontrolled_candidates


def load_config(config_path):
    config_path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("data_process_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_config"):
        config = module.get_config()
    elif hasattr(module, "CONFIG"):
        config = module.CONFIG
    else:
        raise ValueError(f"{config_path} must define get_config() or CONFIG")
    return config, config_path


def validate_config(config):
    total = (
        float(config["train_fraction"])
        + float(config["eval_fraction"])
        + float(config["test_fraction"])
    )
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/eval/test fractions must sum to 1.0, got {total}")

    candidate_total = (
        int(config["num_uncontrolled"])
        + int(config["num_ot"])
        + int(config["num_random"])
    )
    if candidate_total != int(config["candidate_group_size"]):
        raise ValueError(
            "num_uncontrolled + num_ot + num_random must equal candidate_group_size"
        )

    cohort_name = config["cohort_name"]
    if cohort_name not in config["cohorts"]:
        raise ValueError(f"Unknown cohort_name '{cohort_name}'")


def resolve_config(config):
    validate_config(config)
    resolved = dict(config)
    cohort = dict(config["cohorts"][config["cohort_name"]])
    resolved["cohort"] = cohort
    return resolved


def load_roi_spec(roi_list_path):
    names_df = pd.read_csv(roi_list_path)
    fs_labels = list(names_df["FS_label"])
    if len(fs_labels) != 84:
        raise ValueError(f"Expected 84 ROI labels, found {len(fs_labels)} in {roi_list_path}")
    return fs_labels


def extract_aligned_roi_columns(df, fs_labels, dataset_name):
    mapped = map_region_columns(df, fs_labels)
    missing = [fs for fs, col in zip(fs_labels, mapped) if col is None]
    if missing:
        raise ValueError(
            f"{dataset_name}: {len(missing)}/84 ROIs unmatched. First unmapped: {missing[:5]}"
        )
    if len(set(mapped)) != len(mapped):
        raise ValueError(f"{dataset_name}: duplicate ROI column mapping detected")
    return mapped


def _string_series(df, column_name, fallback_prefix):
    if column_name and column_name in df.columns:
        values = df[column_name].astype("string").fillna("").astype(str).tolist()
    else:
        values = [""] * len(df)
    output = []
    for idx, value in enumerate(values):
        value = value.strip()
        output.append(value if value else f"{fallback_prefix}_{idx:06d}")
    return np.asarray(output, dtype=str)


def _float_series(df, column_name):
    if column_name and column_name in df.columns:
        return pd.to_numeric(df[column_name], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(df), np.nan, dtype=np.float64)


def _optional_string_series(df, column_name):
    if column_name and column_name in df.columns:
        return df[column_name].astype("string").fillna("").astype(str).to_numpy(dtype=str)
    return np.asarray([""] * len(df), dtype=str)


def load_endpoint_dataframe(csv_path, fs_labels, cohort_cfg, max_missing_fraction, endpoint_name):
    df = pd.read_csv(csv_path)
    roi_columns = extract_aligned_roi_columns(df, fs_labels, endpoint_name)
    roi_frame = df[roi_columns].apply(pd.to_numeric, errors="coerce")

    missing_fraction = roi_frame.isna().mean(axis=1).to_numpy(dtype=np.float64)
    keep_mask = missing_fraction <= float(max_missing_fraction)
    kept_df = df.loc[keep_mask].reset_index(drop=True)
    kept_roi = roi_frame.loc[keep_mask].reset_index(drop=True)
    kept_missing_fraction = missing_fraction[keep_mask]

    return {
        "dataframe": kept_df,
        "roi_frame": kept_roi,
        "roi_columns": roi_columns,
        "canonical_roi_names": fs_labels,
        "sample_ids": _string_series(
            kept_df,
            cohort_cfg.get("sample_id_col"),
            f"{endpoint_name.lower()}_sample",
        ),
        "subject_ids": _string_series(
            kept_df,
            cohort_cfg.get("subject_id_col"),
            f"{endpoint_name.lower()}_subject",
        ),
        "ages": _float_series(kept_df, cohort_cfg.get("age_col")),
        "diagnoses": _optional_string_series(kept_df, cohort_cfg.get("diagnosis_col")),
        "sexes": _optional_string_series(kept_df, cohort_cfg.get("sex_col")),
        "scan_dates": _optional_string_series(kept_df, cohort_cfg.get("scan_date_col")),
        "rows_before_drop": int(len(df)),
        "rows_after_drop": int(len(kept_df)),
        "rows_dropped_for_missingness": int((~keep_mask).sum()),
        "missing_fraction_mean_kept": float(kept_missing_fraction.mean()) if kept_missing_fraction.size else np.nan,
        "missing_fraction_max_kept": float(kept_missing_fraction.max()) if kept_missing_fraction.size else np.nan,
    }


def split_indices_by_subject(subject_ids, train_fraction, eval_fraction, seed):
    subject_ids = np.asarray(subject_ids, dtype=str)
    unique_subjects = np.unique(subject_ids)
    unique_subjects = np.sort(unique_subjects)
    rng = np.random.default_rng(seed)
    permuted = unique_subjects[rng.permutation(len(unique_subjects))]

    n_subjects = len(permuted)
    n_train = int(np.floor(n_subjects * train_fraction))
    n_eval = int(np.floor(n_subjects * eval_fraction))
    n_test = n_subjects - n_train - n_eval
    if n_test < 0:
        raise ValueError("Negative test subject count during split")

    train_subjects = set(permuted[:n_train])
    eval_subjects = set(permuted[n_train : n_train + n_eval])
    test_subjects = set(permuted[n_train + n_eval :])

    split_masks = {
        "train": np.isin(subject_ids, list(train_subjects)),
        "val": np.isin(subject_ids, list(eval_subjects)),
        "test": np.isin(subject_ids, list(test_subjects)),
    }
    return {
        split: np.flatnonzero(mask)
        for split, mask in split_masks.items()
    }


def split_loaded_dataset(dataset_dict, train_fraction, eval_fraction, seed):
    split_idx = split_indices_by_subject(
        dataset_dict["subject_ids"],
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    outputs = {}
    for split_name, indices in split_idx.items():
        outputs[split_name] = {
            "X": dataset_dict["roi_frame"].iloc[indices].to_numpy(dtype=np.float64),
            "sample_ids": dataset_dict["sample_ids"][indices],
            "subject_ids": dataset_dict["subject_ids"][indices],
            "ages": dataset_dict["ages"][indices],
            "diagnoses": dataset_dict["diagnoses"][indices],
            "sexes": dataset_dict["sexes"][indices],
            "scan_dates": dataset_dict["scan_dates"][indices],
            "indices": indices,
            "n_subjects": int(np.unique(dataset_dict["subject_ids"][indices]).size),
        }
    return outputs


def impute_from_x_train(split_data):
    X_train = split_data["X_train"]
    roi_mean = np.nanmean(X_train, axis=0)
    if np.isnan(roi_mean).any():
        raise ValueError("At least one ROI column is all-NaN in X_train; cannot impute")

    for key in ("X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"):
        arr = split_data[key]
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            split_data[key] = np.where(nan_mask, roi_mean[None, :], arr)
    return roi_mean


def normalize_from_x_train(split_data, std_eps, use_normalization):
    roi_mean = split_data["X_train"].mean(axis=0)
    roi_std = split_data["X_train"].std(axis=0, ddof=0)
    roi_std = np.maximum(roi_std, std_eps)

    if use_normalization:
        for key in ("X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"):
            split_data[key] = (split_data[key] - roi_mean[None, :]) / roi_std[None, :]
    return roi_mean, roi_std


def ensure_finite_arrays(split_data):
    for key in ("X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"):
        arr = split_data[key]
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} contains NaN/Inf after preprocessing")


def make_output_dir(project_root, cohort_name, use_normalization):
    mode = "normalization" if use_normalization else "raw"
    outdir = project_root / "data" / "processed" / "candidate_groups" / cohort_name / mode
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def gather_processed_arrays(source_splits, target_splits):
    return {
        "X_train": source_splits["train"]["X"],
        "X_val": source_splits["val"]["X"],
        "X_test": source_splits["test"]["X"],
        "Y_train": target_splits["train"]["X"],
        "Y_val": target_splits["val"]["X"],
        "Y_test": target_splits["test"]["X"],
    }


def aggregate_warnings(*warning_lists):
    merged = []
    for warning_list in warning_lists:
        merged.extend(warning_list)
    return merged


def write_config_snapshot(outdir, resolved_config):
    serializable = {}
    for key, value in resolved_config.items():
        if key == "cohorts":
            continue
        serializable[key] = value
    text = "# Auto-generated preprocessing config snapshot.\n\nCONFIG = "
    text += pprint.pformat(serializable, sort_dicts=False, width=100)
    text += "\n"
    (outdir / "config_snapshot.py").write_text(text)


def write_candidate_summary(outdir, summary_frames):
    summary_df = pd.concat(summary_frames, axis=0, ignore_index=True)
    summary_df.to_csv(outdir / "candidate_summary.csv", index=False)
    return summary_df


def describe_split(name, split_dict):
    return (
        f"{name}: rows(train/val/test)="
        f"{split_dict['train']['X'].shape[0]}/{split_dict['val']['X'].shape[0]}/{split_dict['test']['X'].shape[0]} | "
        f"subjects(train/val/test)="
        f"{split_dict['train']['n_subjects']}/{split_dict['val']['n_subjects']}/{split_dict['test']['n_subjects']}"
    )


def mean_cost_by_label(costs, labels, label):
    mask = labels == label
    if not np.any(mask):
        return np.nan
    return float(np.mean(costs[mask]))


def write_data_summary(
    outdir,
    resolved_config,
    source_info,
    target_info,
    source_splits,
    target_splits,
    summary_df,
    overlap_stats,
    cost_stats,
    warnings,
):
    lines = [
        f"cohort: {resolved_config['cohort_name']}",
        f"roi_features: {resolved_config['dimension']}",
        f"use_normalization: {resolved_config['use_normalization']}",
        "normalization_source: X_train only",
        f"split_fractions: train={resolved_config['train_fraction']} eval={resolved_config['eval_fraction']} test={resolved_config['test_fraction']}",
        f"seed: {resolved_config['seed']}",
        f"beta: {resolved_config['beta']}",
        f"covariance_regularization: {resolved_config['cov_regularization']}",
        f"candidate_group_design: {resolved_config['num_uncontrolled']} uncontrolled + {resolved_config['num_ot']} OT + {resolved_config['num_random']} random",
        f"laplacian_mode: {resolved_config['laplacian_mode']}",
        f"ot_method: sinkhorn epsilon={resolved_config['ot_epsilon']} max_iter={resolved_config['ot_max_iter']}",
        f"out_of_sample_ot_method: knn_transfer k={resolved_config['ot_transfer_k']} temperature={resolved_config['ot_transfer_temperature']}",
        f"missing_value_rule: drop rows with ROI missing fraction > {resolved_config['max_missing_fraction']}; impute remaining NaNs from X_train mean",
        f"source_rows_before_after_drop: {source_info['rows_before_drop']} -> {source_info['rows_after_drop']}",
        f"target_rows_before_after_drop: {target_info['rows_before_drop']} -> {target_info['rows_after_drop']}",
        f"source_rows_dropped_for_missingness: {source_info['rows_dropped_for_missingness']}",
        f"target_rows_dropped_for_missingness: {target_info['rows_dropped_for_missingness']}",
        describe_split("source", source_splits),
        describe_split("target", target_splits),
        "candidate_rows_per_split: "
        f"train={len(summary_df[summary_df['split'] == 'train'])}, "
        f"val={len(summary_df[summary_df['split'] == 'val'])}, "
        f"test={len(summary_df[summary_df['split'] == 'test'])}",
        "unc_ot_overlap_per_source: "
        f"mean={overlap_stats['mean']:.3f}, max={overlap_stats['max']}, min={overlap_stats['min']}",
        "extra_random_due_to_duplicates: "
        f"mean={summary_df['extra_random_due_to_duplicates'].mean():.3f}, "
        f"max={int(summary_df['extra_random_due_to_duplicates'].max())}",
    ]

    for split_name, stats in cost_stats.items():
        lines.append(
            f"{split_name}_mean_uncontrolled_costs: uncontrolled={stats['uncontrolled']:.6f} "
            f"ot={stats['ot']:.6f} random={stats['random']:.6f}"
        )

    lines.append("warnings:")
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    (outdir / "data_summary.txt").write_text("\n".join(lines) + "\n")


def save_processed_npz(
    outdir,
    arrays,
    source_splits,
    target_splits,
    roi_names,
    roi_impute_values,
    roi_mean,
    roi_std,
    use_normalization,
    transitions,
    candidate_results,
):
    save_dict = dict(arrays)
    save_dict.update(
        {
            "X_train_ids": source_splits["train"]["sample_ids"],
            "X_val_ids": source_splits["val"]["sample_ids"],
            "X_test_ids": source_splits["test"]["sample_ids"],
            "Y_train_ids": target_splits["train"]["sample_ids"],
            "Y_val_ids": target_splits["val"]["sample_ids"],
            "Y_test_ids": target_splits["test"]["sample_ids"],
            "X_train_subject_ids": source_splits["train"]["subject_ids"],
            "X_val_subject_ids": source_splits["val"]["subject_ids"],
            "X_test_subject_ids": source_splits["test"]["subject_ids"],
            "Y_train_subject_ids": target_splits["train"]["subject_ids"],
            "Y_val_subject_ids": target_splits["val"]["subject_ids"],
            "Y_test_subject_ids": target_splits["test"]["subject_ids"],
            "X_train_ages": source_splits["train"]["ages"],
            "X_val_ages": source_splits["val"]["ages"],
            "X_test_ages": source_splits["test"]["ages"],
            "Y_train_ages": target_splits["train"]["ages"],
            "Y_val_ages": target_splits["val"]["ages"],
            "Y_test_ages": target_splits["test"]["ages"],
            "X_train_diagnosis": source_splits["train"]["diagnoses"],
            "X_val_diagnosis": source_splits["val"]["diagnoses"],
            "X_test_diagnosis": source_splits["test"]["diagnoses"],
            "Y_train_diagnosis": target_splits["train"]["diagnoses"],
            "Y_val_diagnosis": target_splits["val"]["diagnoses"],
            "Y_test_diagnosis": target_splits["test"]["diagnoses"],
            "X_train_scan_dates": source_splits["train"]["scan_dates"],
            "X_val_scan_dates": source_splits["val"]["scan_dates"],
            "X_test_scan_dates": source_splits["test"]["scan_dates"],
            "Y_train_scan_dates": target_splits["train"]["scan_dates"],
            "Y_val_scan_dates": target_splits["val"]["scan_dates"],
            "Y_test_scan_dates": target_splits["test"]["scan_dates"],
            "roi_names": np.asarray(roi_names, dtype=str),
            "roi_impute_values": np.asarray(roi_impute_values, dtype=np.float64),
            "roi_mean": np.asarray(roi_mean, dtype=np.float64),
            "roi_std": np.asarray(roi_std, dtype=np.float64),
            "use_normalization": np.asarray(use_normalization),
            "L": transitions["L"],
            "M1": transitions["M1"],
            "C1": transitions["C1"],
            "C1_inv": transitions["C1_inv"],
            "candidate_indices_train": candidate_results["train"]["candidate_indices"],
            "candidate_indices_val": candidate_results["val"]["candidate_indices"],
            "candidate_indices_test": candidate_results["test"]["candidate_indices"],
            "candidate_source_labels_train": candidate_results["train"]["source_labels"],
            "candidate_source_labels_val": candidate_results["val"]["source_labels"],
            "candidate_source_labels_test": candidate_results["test"]["source_labels"],
            "uncontrolled_top_indices_train": candidate_results["train"]["uncontrolled_top_indices"],
            "uncontrolled_top_indices_val": candidate_results["val"]["uncontrolled_top_indices"],
            "uncontrolled_top_indices_test": candidate_results["test"]["uncontrolled_top_indices"],
            "uncontrolled_top_costs_train": candidate_results["train"]["uncontrolled_top_costs"],
            "uncontrolled_top_costs_val": candidate_results["val"]["uncontrolled_top_costs"],
            "uncontrolled_top_costs_test": candidate_results["test"]["uncontrolled_top_costs"],
            "ot_top_indices_train": candidate_results["train"]["ot_top_indices"],
            "ot_top_indices_val": candidate_results["val"]["ot_top_indices"],
            "ot_top_indices_test": candidate_results["test"]["ot_top_indices"],
            "ot_top_weights_train": candidate_results["train"]["ot_top_weights"],
            "ot_top_weights_val": candidate_results["val"]["ot_top_weights"],
            "ot_top_weights_test": candidate_results["test"]["ot_top_weights"],
            "random_indices_train": candidate_results["train"]["random_indices"],
            "random_indices_val": candidate_results["val"]["random_indices"],
            "random_indices_test": candidate_results["test"]["random_indices"],
            "candidate_uncontrolled_costs_train": candidate_results["train"]["candidate_uncontrolled_costs"],
            "candidate_uncontrolled_costs_val": candidate_results["val"]["candidate_uncontrolled_costs"],
            "candidate_uncontrolled_costs_test": candidate_results["test"]["candidate_uncontrolled_costs"],
            "candidate_ot_weights_train": candidate_results["train"]["candidate_ot_weights"],
            "candidate_ot_weights_val": candidate_results["val"]["candidate_ot_weights"],
            "candidate_ot_weights_test": candidate_results["test"]["candidate_ot_weights"],
        }
    )
    np.savez_compressed(outdir / "processed_data.npz", **save_dict)


def preprocess_candidate_groups(config):
    resolved = resolve_config(config)
    cohort_cfg = resolved["cohort"]
    fs_labels = load_roi_spec(resolved["roi_list_path"])

    source_info = load_endpoint_dataframe(
        cohort_cfg["source_csv_path"],
        fs_labels,
        cohort_cfg,
        resolved["max_missing_fraction"],
        endpoint_name=f"{resolved['cohort_name']}_source",
    )
    target_info = load_endpoint_dataframe(
        cohort_cfg["target_csv_path"],
        fs_labels,
        cohort_cfg,
        resolved["max_missing_fraction"],
        endpoint_name=f"{resolved['cohort_name']}_target",
    )

    if source_info["canonical_roi_names"] != target_info["canonical_roi_names"]:
        raise ValueError("Source and target canonical ROI order do not match")

    source_splits = split_loaded_dataset(
        source_info,
        train_fraction=resolved["train_fraction"],
        eval_fraction=resolved["eval_fraction"],
        seed=int(resolved["seed"]),
    )
    target_splits = split_loaded_dataset(
        target_info,
        train_fraction=resolved["train_fraction"],
        eval_fraction=resolved["eval_fraction"],
        seed=int(resolved["seed"]) + 1,
    )

    arrays = gather_processed_arrays(source_splits, target_splits)
    impute_values = impute_from_x_train(arrays)
    roi_mean, roi_std = normalize_from_x_train(
        arrays,
        std_eps=float(resolved["std_eps"]),
        use_normalization=bool(resolved["use_normalization"]),
    )
    ensure_finite_arrays(arrays)

    dim = arrays["X_train"].shape[1]
    if dim != int(resolved["dimension"]):
        raise ValueError(f"Expected dimension {resolved['dimension']}, got {dim}")

    L = build_laplacian_matrix(
        resolved["laplacian_mode"],
        dim=dim,
        connectomes_glob=resolved.get("connectome_glob"),
        matrix_path=resolved.get("laplacian_matrix_path"),
        norm=bool(resolved["laplacian_normalize"]),
    )
    transitions = compute_transition_matrices(
        L,
        beta=float(resolved["beta"]),
        eig_eps=float(resolved["eig_eps"]),
        cov_regularization=float(resolved["cov_regularization"]),
    )
    if transitions["L"].shape != (dim, dim):
        raise ValueError(f"L has shape {transitions['L'].shape}, expected {(dim, dim)}")

    top_unc = int(resolved["num_uncontrolled"])
    top_ot = int(resolved["num_ot"])
    chunk_size = int(resolved["matching_chunk_size"])

    unc_train_idx, unc_train_costs = topk_uncontrolled_candidates(
        arrays["X_train"],
        arrays["Y_train"],
        transitions["M1"],
        transitions["C1_inv"],
        topk=top_unc,
        cost_type=resolved["unc_cost_type"],
        chunk_size=chunk_size,
    )
    unc_val_idx, unc_val_costs = topk_uncontrolled_candidates(
        arrays["X_val"],
        arrays["Y_train"],
        transitions["M1"],
        transitions["C1_inv"],
        topk=top_unc,
        cost_type=resolved["unc_cost_type"],
        chunk_size=chunk_size,
    )
    unc_test_idx, unc_test_costs = topk_uncontrolled_candidates(
        arrays["X_test"],
        arrays["Y_train"],
        transitions["M1"],
        transitions["C1_inv"],
        topk=top_unc,
        cost_type=resolved["unc_cost_type"],
        chunk_size=chunk_size,
    )

    ot_train = compute_ot_train_candidates(
        arrays["X_train"],
        arrays["Y_train"],
        transitions["M1"],
        transitions["C1_inv"],
        topk=top_ot,
        ot_cost_type=resolved["ot_cost_type"],
        epsilon=float(resolved["ot_epsilon"]),
        max_iter=int(resolved["ot_max_iter"]),
        tol=float(resolved["ot_tolerance"]),
        chunk_size=chunk_size,
    )
    ot_val_idx, ot_val_weights = transfer_ot_candidates(
        arrays["X_val"],
        arrays["X_train"],
        ot_train["row_probs"],
        transitions["M1"],
        topk=top_ot,
        knn_k=int(resolved["ot_transfer_k"]),
        temperature=float(resolved["ot_transfer_temperature"]),
        chunk_size=chunk_size,
    )
    ot_test_idx, ot_test_weights = transfer_ot_candidates(
        arrays["X_test"],
        arrays["X_train"],
        ot_train["row_probs"],
        transitions["M1"],
        topk=top_ot,
        knn_k=int(resolved["ot_transfer_k"]),
        temperature=float(resolved["ot_transfer_temperature"]),
        chunk_size=chunk_size,
    )

    candidate_results = {}
    split_inputs = {
        "train": (
            source_splits["train"]["sample_ids"],
            unc_train_idx,
            unc_train_costs,
            ot_train["top_indices"],
            ot_train["top_weights"],
        ),
        "val": (
            source_splits["val"]["sample_ids"],
            unc_val_idx,
            unc_val_costs,
            ot_val_idx,
            ot_val_weights,
        ),
        "test": (
            source_splits["test"]["sample_ids"],
            unc_test_idx,
            unc_test_costs,
            ot_test_idx,
            ot_test_weights,
        ),
    }

    summary_frames = []
    warning_lists = []
    for split_name, (
        sample_ids,
        unc_idx,
        unc_costs,
        ot_idx,
        ot_weights,
    ) in split_inputs.items():
        result = build_candidate_groups(
            split_name=split_name,
            sample_ids=sample_ids,
            y_train_size=arrays["Y_train"].shape[0],
            uncontrolled_top_indices=unc_idx,
            uncontrolled_top_costs=unc_costs,
            ot_top_indices=ot_idx,
            ot_top_weights=ot_weights,
            candidate_group_size=int(resolved["candidate_group_size"]),
            num_uncontrolled=int(resolved["num_uncontrolled"]),
            num_ot=int(resolved["num_ot"]),
            num_random=int(resolved["num_random"]),
            seed=int(resolved["seed"]),
        )
        result["uncontrolled_top_indices"] = unc_idx
        result["uncontrolled_top_costs"] = unc_costs
        result["ot_top_indices"] = ot_idx
        result["ot_top_weights"] = ot_weights
        candidate_results[split_name] = result
        summary_frames.append(result["summary_df"])
        warning_lists.append(result["warnings"])

    for split_name in ("train", "val", "test"):
        candidate_indices = candidate_results[split_name]["candidate_indices"]
        if candidate_indices.shape[1] != int(resolved["candidate_group_size"]):
            raise ValueError(f"{split_name}: candidate group width mismatch")
        if candidate_indices.min() < 0 or candidate_indices.max() >= arrays["Y_train"].shape[0]:
            raise ValueError(f"{split_name}: candidate indices fall outside Y_train")
        if arrays["Y_train"].shape[0] >= int(resolved["candidate_group_size"]):
            unique_counts = np.apply_along_axis(lambda row: np.unique(row).size, 1, candidate_indices)
            if np.any(unique_counts != int(resolved["candidate_group_size"])):
                raise ValueError(f"{split_name}: duplicate candidates detected despite sufficient Y_train size")

    overlap_counts = np.concatenate(
        [candidate_results[split]["overlap_counts"] for split in ("train", "val", "test")]
    )
    overlap_stats = {
        "mean": float(overlap_counts.mean()) if overlap_counts.size else 0.0,
        "min": int(overlap_counts.min()) if overlap_counts.size else 0,
        "max": int(overlap_counts.max()) if overlap_counts.size else 0,
    }

    cost_stats = {}
    for split_name, x_key in (("train", "X_train"), ("val", "X_val"), ("test", "X_test")):
        candidate_costs = compute_selected_costs(
            arrays[x_key],
            arrays["Y_train"],
            candidate_results[split_name]["candidate_indices"],
            transitions["M1"],
            transitions["C1_inv"],
            cost_type=resolved["unc_cost_type"],
            chunk_size=chunk_size,
        )
        labels = candidate_results[split_name]["source_labels"]
        cost_stats[split_name] = {
            "uncontrolled": mean_cost_by_label(candidate_costs, labels, UNCONTROLLED_LABEL),
            "ot": mean_cost_by_label(candidate_costs, labels, OT_LABEL),
            "random": mean_cost_by_label(candidate_costs, labels, RANDOM_LABEL),
        }

    outdir = make_output_dir(
        PROJECT_ROOT,
        resolved["cohort_name"],
        bool(resolved["use_normalization"]),
    )
    write_config_snapshot(outdir, resolved)
    summary_df = write_candidate_summary(outdir, summary_frames)
    warnings = aggregate_warnings(*warning_lists)
    write_data_summary(
        outdir,
        resolved,
        source_info,
        target_info,
        source_splits,
        target_splits,
        summary_df,
        overlap_stats,
        cost_stats,
        warnings,
    )
    save_processed_npz(
        outdir,
        arrays,
        source_splits,
        target_splits,
        roi_names=fs_labels,
        roi_impute_values=impute_values,
        roi_mean=roi_mean,
        roi_std=roi_std,
        use_normalization=bool(resolved["use_normalization"]),
        transitions=transitions,
        candidate_results=candidate_results,
    )

    print(f"Wrote preprocessing outputs to {outdir}")
    print(
        f"unc/OT overlap mean={overlap_stats['mean']:.3f}, "
        f"max={overlap_stats['max']}"
    )
    for split_name, stats in cost_stats.items():
        print(
            f"{split_name}: mean uncontrolled cost "
            f"(unc={stats['uncontrolled']:.6f}, ot={stats['ot']:.6f}, random={stats['random']:.6f})"
        )


def main():
    default_config_path = PROJECT_ROOT / "configs" / "data_process.py"
    default_config, _ = load_config(default_config_path)
    cohort_choices = sorted(default_config["cohorts"].keys())

    parser = argparse.ArgumentParser(description="Preprocess candidate endpoint groups")
    parser.add_argument(
        "--config",
        default=str(default_config_path),
        help="Path to the preprocessing config file.",
    )
    parser.add_argument(
        "--cohort",
        choices=cohort_choices,
        default=None,
        help="Optional cohort override, e.g. adni or nacc.",
    )
    parser.add_argument(
        "--output-variant",
        choices=["normalization", "raw"],
        default=None,
        help="Optional output variant override. 'normalization' sets use_normalization=True and 'raw' sets it to False.",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    if args.cohort is not None:
        config["cohort_name"] = args.cohort
    if args.output_variant is not None:
        config["use_normalization"] = args.output_variant == "normalization"
    preprocess_candidate_groups(config)


if __name__ == "__main__":
    main()
