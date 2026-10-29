"""Build the Union -> ADNI evaluation data (cross-cohort generalization).

Keeps the Union training split (X_train, Y_train, normalization statistics,
graph, training candidate groups) and restricts the validation and test
splits to ADNI scans. Models trained on the Union data are evaluated on this
file without retraining.

usage:
  python -m scripts.build_union_to_adni \
      --union data/processed/candidate_groups/adni_nacc_union/normalization/processed_data.npz \
      --out data/processed/candidate_groups/union_to_adni/normalization/processed_data.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

X_CAND_KEYS = (
    "candidate_indices",
    "candidate_source_labels",
    "uncontrolled_top_indices",
    "uncontrolled_top_costs",
    "ot_top_indices",
    "ot_top_weights",
    "random_indices",
    "candidate_uncontrolled_costs",
    "candidate_ot_weights",
)
X_META_KEYS = ("ids", "subject_ids", "ages", "diagnosis", "scan_dates")
Y_META_KEYS = ("ids", "subject_ids", "ages", "diagnosis", "scan_dates")
HELD_OUT = ("val", "test")


def _prefixed_mask(values, prefix: str) -> np.ndarray:
    values = np.asarray(values).astype(str)
    return np.char.startswith(values, prefix)


def build(union_path: Path, out: Path, prefix: str = "adni:") -> dict:
    union = np.load(union_path)
    save = {}
    for k in union.files:
        if any(k.startswith(f"X_{s}_") for s in HELD_OUT):
            continue
        if any(k.startswith(f"Y_{s}_") for s in HELD_OUT):
            continue
        if any(k.startswith(f"{base}_{s}") for base in X_CAND_KEYS for s in HELD_OUT):
            continue
        if k in ("X_val", "X_test", "Y_val", "Y_test"):
            continue
        save[k] = union[k]

    for split in HELD_OUT:
        mx = _prefixed_mask(union[f"X_{split}_ids"], prefix)
        my = _prefixed_mask(union[f"Y_{split}_ids"], prefix)
        save[f"X_{split}"] = np.asarray(union[f"X_{split}"])[mx]
        save[f"Y_{split}"] = np.asarray(union[f"Y_{split}"])[my]
        for base in X_META_KEYS:
            key = f"X_{split}_{base}"
            if key in union.files:
                save[key] = np.asarray(union[key])[mx]
        for base in Y_META_KEYS:
            key = f"Y_{split}_{base}"
            if key in union.files:
                save[key] = np.asarray(union[key])[my]
        for base in X_CAND_KEYS:
            key = f"{base}_{split}"
            if key in union.files:
                save[key] = np.asarray(union[key])[mx]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **save)
    return save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", default="data/processed/candidate_groups/adni_nacc_union/normalization/processed_data.npz")
    ap.add_argument("--out", default="data/processed/candidate_groups/union_to_adni/normalization/processed_data.npz")
    args = ap.parse_args()
    save = build(Path(args.union), Path(args.out))
    print(f"Wrote {args.out}: X_train={save['X_train'].shape} Y_train={save['Y_train'].shape} "
          f"X_val={save['X_val'].shape} X_test={save['X_test'].shape} "
          f"Y_val={save['Y_val'].shape} Y_test={save['Y_test'].shape}")


if __name__ == "__main__":
    main()
