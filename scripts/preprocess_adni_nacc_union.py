"""Build the multi-cohort (Union) processed data.

  - Source distribution X is the CN cohort of both datasets; target
    distribution Y is the impaired cohort of both datasets.
  - Source and target are split with the standard 70/20/10 train/val/test
    fractions (configs/data_process.py); evaluation uses val + test.
  - Imputation and z-score normalization are fit on X_train only, then
    applied to all X/Y splits.

usage:
  python -m scripts.preprocess_adni_nacc_union
"""
from __future__ import annotations

import argparse
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.data_process import get_config
from matching.candidate_groups import (
    OT_LABEL,
    RANDOM_LABEL,
    UNCONTROLLED_LABEL,
    build_candidate_groups,
)
from matching.ot_matching import compute_ot_train_candidates, transfer_ot_candidates
from matching.unc_reachability import compute_selected_costs, topk_uncontrolled_candidates
from scripts.preprocess_candidate_groups import (
    aggregate_warnings,
    build_laplacian_matrix,
    compute_transition_matrices,
    describe_split,
    ensure_finite_arrays,
    impute_from_x_train,
    load_endpoint_dataframe,
    load_roi_spec,
    mean_cost_by_label,
    normalize_from_x_train,
    save_processed_npz,
    split_loaded_dataset,
    write_candidate_summary,
    write_config_snapshot,
)


OUTDIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_groups"
    / "adni_nacc_union"
    / "normalization"
)


def _combine_infos(named_infos: list[tuple[str, dict]], endpoint_name: str) -> dict:
    canonical = named_infos[0][1]["canonical_roi_names"]
    frames = []
    roi_frames = []
    sample_ids = []
    subject_ids = []
    ages = []
    diagnoses = []
    sexes = []
    scan_dates = []
    rows_before = 0
    rows_after = 0
    rows_dropped = 0
    missing_mean_terms = []
    missing_max = []

    for cohort, info in named_infos:
        if info["canonical_roi_names"] != canonical:
            raise ValueError(f"{endpoint_name}: ROI order mismatch for {cohort}")
        n = int(info["rows_after_drop"])
        frames.append(info["dataframe"].copy())
        roi_frames.append(info["roi_frame"].reset_index(drop=True))
        sample_ids.append(np.asarray([f"{cohort}:{x}" for x in info["sample_ids"]], dtype=str))
        subject_ids.append(np.asarray([f"{cohort}:{x}" for x in info["subject_ids"]], dtype=str))
        ages.append(np.asarray(info["ages"], dtype=np.float64))
        diagnoses.append(np.asarray([f"{cohort}:{x}" for x in info["diagnoses"]], dtype=str))
        sexes.append(np.asarray(info["sexes"], dtype=str))
        scan_dates.append(np.asarray(info["scan_dates"], dtype=str))
        rows_before += int(info["rows_before_drop"])
        rows_after += n
        rows_dropped += int(info["rows_dropped_for_missingness"])
        if n:
            missing_mean_terms.append(float(info["missing_fraction_mean_kept"]) * n)
            missing_max.append(float(info["missing_fraction_max_kept"]))

    return {
        "dataframe": pd.concat(frames, axis=0, ignore_index=True),
        "roi_frame": pd.concat(roi_frames, axis=0, ignore_index=True),
        "roi_columns": named_infos[0][1]["roi_columns"],
        "canonical_roi_names": canonical,
        "sample_ids": np.concatenate(sample_ids),
        "subject_ids": np.concatenate(subject_ids),
        "ages": np.concatenate(ages),
        "diagnoses": np.concatenate(diagnoses),
        "sexes": np.concatenate(sexes),
        "scan_dates": np.concatenate(scan_dates),
        "rows_before_drop": rows_before,
        "rows_after_drop": rows_after,
        "rows_dropped_for_missingness": rows_dropped,
        "missing_fraction_mean_kept": (
            float(np.sum(missing_mean_terms) / rows_after) if rows_after else np.nan
        ),
        "missing_fraction_max_kept": float(np.max(missing_max)) if missing_max else np.nan,
    }


def _write_summary(
    outdir: Path,
    resolved: dict,
    source_parts: list[tuple[str, dict]],
    target_parts: list[tuple[str, dict]],
    source_info: dict,
    target_info: dict,
    source_splits: dict,
    target_splits: dict,
    summary_df: pd.DataFrame,
    overlap_stats: dict,
    cost_stats: dict,
    warnings: list[str],
) -> None:
    lines = [
        "experiment: adni_nacc_union",
        "protocol: X=ADNI CN + NACC CN; Y=ADNI ADMCI + NACC ADMCI",
        "split_protocol: standard split, train=70%, eval=val+test=30%",
        "normalization_source: X_train only",
        f"use_normalization: {resolved['use_normalization']}",
        f"split_fractions: train={resolved['train_fraction']} eval={resolved['eval_fraction']} test={resolved['test_fraction']}",
        f"seed: {resolved['seed']}",
        f"roi_features: {resolved['dimension']}",
        f"beta: {resolved['beta']}",
        f"candidate_group_design: {resolved['num_uncontrolled']} uncontrolled + {resolved['num_ot']} OT + {resolved['num_random']} random",
        f"missing_value_rule: drop rows with ROI missing fraction > {resolved['max_missing_fraction']}; impute remaining NaNs from X_train mean",
    ]
    for cohort, info in source_parts:
        lines.append(
            f"source_{cohort}_cn_rows_before_after_drop: "
            f"{info['rows_before_drop']} -> {info['rows_after_drop']}"
        )
    for cohort, info in target_parts:
        lines.append(
            f"target_{cohort}_admci_rows_before_after_drop: "
            f"{info['rows_before_drop']} -> {info['rows_after_drop']}"
        )
    lines.extend(
        [
            f"source_union_rows_before_after_drop: {source_info['rows_before_drop']} -> {source_info['rows_after_drop']}",
            f"target_union_rows_before_after_drop: {target_info['rows_before_drop']} -> {target_info['rows_after_drop']}",
            describe_split("source_union", source_splits),
            describe_split("target_union", target_splits),
            "candidate_rows_per_split: "
            f"train={len(summary_df[summary_df['split'] == 'train'])}, "
            f"val={len(summary_df[summary_df['split'] == 'val'])}, "
            f"test={len(summary_df[summary_df['split'] == 'test'])}",
            "unc_ot_overlap_per_source: "
            f"mean={overlap_stats['mean']:.3f}, max={overlap_stats['max']}, min={overlap_stats['min']}",
        ]
    )
    for split_name, stats in cost_stats.items():
        lines.append(
            f"{split_name}_mean_uncontrolled_costs: uncontrolled={stats['uncontrolled']:.6f} "
            f"ot={stats['ot']:.6f} random={stats['random']:.6f}"
        )
    lines.append("warnings:")
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- none"])
    (outdir / "data_summary.txt").write_text("\n".join(lines) + "\n")


def build_dataset(outdir: Path, force: bool = False) -> None:
    if outdir.exists() and (outdir / "processed_data.npz").is_file() and not force:
        raise FileExistsError(f"{outdir}/processed_data.npz exists; pass --force")
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    cfg["cohort_name"] = "adni_nacc_union"
    cfg["use_normalization"] = True
    fs_labels = load_roi_spec(cfg["roi_list_path"])

    adni_cfg = cfg["cohorts"]["adni"]
    nacc_cfg = cfg["cohorts"]["nacc"]
    source_parts = [
        (
            "adni",
            load_endpoint_dataframe(
                adni_cfg["source_csv_path"], fs_labels, adni_cfg,
                cfg["max_missing_fraction"], endpoint_name="adni_cn_source",
            ),
        ),
        (
            "nacc",
            load_endpoint_dataframe(
                nacc_cfg["source_csv_path"], fs_labels, nacc_cfg,
                cfg["max_missing_fraction"], endpoint_name="nacc_cn_source",
            ),
        ),
    ]
    target_parts = [
        (
            "adni",
            load_endpoint_dataframe(
                adni_cfg["target_csv_path"], fs_labels, adni_cfg,
                cfg["max_missing_fraction"], endpoint_name="adni_admci_target",
            ),
        ),
        (
            "nacc",
            load_endpoint_dataframe(
                nacc_cfg["target_csv_path"], fs_labels, nacc_cfg,
                cfg["max_missing_fraction"], endpoint_name="nacc_admci_target",
            ),
        ),
    ]

    source_info = _combine_infos(source_parts, "source_union")
    target_info = _combine_infos(target_parts, "target_union")
    source_splits = split_loaded_dataset(
        source_info,
        train_fraction=float(cfg["train_fraction"]),
        eval_fraction=float(cfg["eval_fraction"]),
        seed=int(cfg["seed"]),
    )
    target_splits = split_loaded_dataset(
        target_info,
        train_fraction=float(cfg["train_fraction"]),
        eval_fraction=float(cfg["eval_fraction"]),
        seed=int(cfg["seed"]) + 1,
    )

    arrays = {
        "X_train": source_splits["train"]["X"],
        "X_val": source_splits["val"]["X"],
        "X_test": source_splits["test"]["X"],
        "Y_train": target_splits["train"]["X"],
        "Y_val": target_splits["val"]["X"],
        "Y_test": target_splits["test"]["X"],
    }
    impute_values = impute_from_x_train(arrays)
    roi_mean, roi_std = normalize_from_x_train(
        arrays, std_eps=float(cfg["std_eps"]), use_normalization=True,
    )
    ensure_finite_arrays(arrays)

    dim = int(cfg["dimension"])
    L = build_laplacian_matrix(
        cfg["laplacian_mode"],
        dim=dim,
        connectomes_glob=cfg.get("connectome_glob"),
        matrix_path=cfg.get("laplacian_matrix_path"),
        norm=bool(cfg["laplacian_normalize"]),
    )
    transitions = compute_transition_matrices(
        L,
        beta=float(cfg["beta"]),
        eig_eps=float(cfg["eig_eps"]),
        cov_regularization=float(cfg["cov_regularization"]),
    )

    top_unc = int(cfg["num_uncontrolled"])
    top_ot = int(cfg["num_ot"])
    chunk_size = int(cfg["matching_chunk_size"])
    unc = {}
    for split_name, x_key in (("train", "X_train"), ("val", "X_val"), ("test", "X_test")):
        unc[split_name] = topk_uncontrolled_candidates(
            arrays[x_key], arrays["Y_train"], transitions["M1"], transitions["C1_inv"],
            topk=top_unc, cost_type=cfg["unc_cost_type"], chunk_size=chunk_size,
        )

    ot_train = compute_ot_train_candidates(
        arrays["X_train"], arrays["Y_train"], transitions["M1"], transitions["C1_inv"],
        topk=top_ot, ot_cost_type=cfg["ot_cost_type"],
        epsilon=float(cfg["ot_epsilon"]), max_iter=int(cfg["ot_max_iter"]),
        tol=float(cfg["ot_tolerance"]), chunk_size=chunk_size,
    )
    ot_val = transfer_ot_candidates(
        arrays["X_val"], arrays["X_train"], ot_train["row_probs"], transitions["M1"],
        topk=top_ot, knn_k=int(cfg["ot_transfer_k"]),
        temperature=float(cfg["ot_transfer_temperature"]), chunk_size=chunk_size,
    )
    ot_test = transfer_ot_candidates(
        arrays["X_test"], arrays["X_train"], ot_train["row_probs"], transitions["M1"],
        topk=top_ot, knn_k=int(cfg["ot_transfer_k"]),
        temperature=float(cfg["ot_transfer_temperature"]), chunk_size=chunk_size,
    )

    split_inputs = {
        "train": (source_splits["train"]["sample_ids"], unc["train"][0], unc["train"][1], ot_train["top_indices"], ot_train["top_weights"]),
        "val": (source_splits["val"]["sample_ids"], unc["val"][0], unc["val"][1], ot_val[0], ot_val[1]),
        "test": (source_splits["test"]["sample_ids"], unc["test"][0], unc["test"][1], ot_test[0], ot_test[1]),
    }
    candidate_results = {}
    summary_frames = []
    warning_lists = []
    for split_name, (sample_ids, unc_idx, unc_costs, ot_idx, ot_weights) in split_inputs.items():
        result = build_candidate_groups(
            split_name=split_name,
            sample_ids=sample_ids,
            y_train_size=arrays["Y_train"].shape[0],
            uncontrolled_top_indices=unc_idx,
            uncontrolled_top_costs=unc_costs,
            ot_top_indices=ot_idx,
            ot_top_weights=ot_weights,
            candidate_group_size=int(cfg["candidate_group_size"]),
            num_uncontrolled=int(cfg["num_uncontrolled"]),
            num_ot=int(cfg["num_ot"]),
            num_random=int(cfg["num_random"]),
            seed=int(cfg["seed"]),
        )
        result["uncontrolled_top_indices"] = unc_idx
        result["uncontrolled_top_costs"] = unc_costs
        result["ot_top_indices"] = ot_idx
        result["ot_top_weights"] = ot_weights
        candidate_results[split_name] = result
        summary_frames.append(result["summary_df"])
        warning_lists.append(result["warnings"])

    overlap_counts = np.concatenate(
        [candidate_results[s]["overlap_counts"] for s in ("train", "val", "test")]
    )
    overlap_stats = {
        "mean": float(overlap_counts.mean()) if overlap_counts.size else 0.0,
        "min": int(overlap_counts.min()) if overlap_counts.size else 0,
        "max": int(overlap_counts.max()) if overlap_counts.size else 0,
    }
    cost_stats = {}
    for split_name, x_key in (("train", "X_train"), ("val", "X_val"), ("test", "X_test")):
        candidate_costs = compute_selected_costs(
            arrays[x_key], arrays["Y_train"],
            candidate_results[split_name]["candidate_indices"],
            transitions["M1"], transitions["C1_inv"],
            cost_type=cfg["unc_cost_type"], chunk_size=chunk_size,
        )
        labels = candidate_results[split_name]["source_labels"]
        cost_stats[split_name] = {
            "uncontrolled": mean_cost_by_label(candidate_costs, labels, UNCONTROLLED_LABEL),
            "ot": mean_cost_by_label(candidate_costs, labels, OT_LABEL),
            "random": mean_cost_by_label(candidate_costs, labels, RANDOM_LABEL),
        }

    write_config_snapshot(outdir, cfg)
    summary_df = write_candidate_summary(outdir, summary_frames)
    warnings = aggregate_warnings(*warning_lists)
    _write_summary(
        outdir, cfg, source_parts, target_parts, source_info, target_info,
        source_splits, target_splits, summary_df, overlap_stats, cost_stats, warnings,
    )
    save_processed_npz(
        outdir, arrays, source_splits, target_splits,
        roi_names=fs_labels, roi_impute_values=impute_values,
        roi_mean=roi_mean, roi_std=roi_std, use_normalization=True,
        transitions=transitions, candidate_results=candidate_results,
    )
    metadata = {
        "experiment": "adni_nacc_union",
        "processed_data": str(outdir / "processed_data.npz"),
        "X_train": arrays["X_train"].shape,
        "X_eval": (arrays["X_val"].shape[0] + arrays["X_test"].shape[0], dim),
        "Y_train": arrays["Y_train"].shape,
        "Y_eval": (arrays["Y_val"].shape[0] + arrays["Y_test"].shape[0], dim),
        "normalization_source": "ADNI+NACC CN train split only",
    }
    (outdir / "experiment_metadata.txt").write_text(pprint.pformat(metadata) + "\n")
    print(f"Wrote {outdir / 'processed_data.npz'}")
    print(metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_dataset(Path(args.outdir), force=bool(args.force))


if __name__ == "__main__":
    main()
