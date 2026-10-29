"""Write the ADNI split manifest (scan identifier -> split) from the processed data.

Columns: cohort, class (CN / ADMCI), scan_id (LONIUID), split_adni (ADNI->ADNI
train / val / test), split_union (train / val / test in the Union data; the
Union->ADNI evaluation uses the val and test scans).

usage:
  python -m scripts.make_split_manifest --out splits/adni_manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/processed/candidate_groups")
SPLITS = ("train", "val", "test")
CLASSES = (("X", "CN"), ("Y", "ADMCI"))


def split_table(npz_path: Path, prefix: str = "") -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    rows = []
    for key, cls in CLASSES:
        for s in SPLITS:
            for sid in z[f"{key}_{s}_ids"].astype(str):
                if prefix and not sid.startswith(prefix):
                    continue
                rows.append({"class": cls, "scan_id": sid[len(prefix):], "split": s})
    # A scan listed twice in the source table appears twice in the processed data
    # (always within the same split); the manifest has one row per scan.
    return pd.DataFrame(rows).drop_duplicates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adni", default=str(ROOT / "adni/normalization/processed_data.npz"))
    ap.add_argument("--union", default=str(ROOT / "adni_nacc_union/normalization/processed_data.npz"))
    ap.add_argument("--out", default="splits/adni_manifest.csv")
    args = ap.parse_args()

    adni = split_table(Path(args.adni)).rename(columns={"split": "split_adni"})
    union = split_table(Path(args.union), prefix="adni:").rename(columns={"split": "split_union"})
    m = adni.merge(union, on=["class", "scan_id"], how="outer", validate="one_to_one")
    if m.isna().any().any():
        raise RuntimeError("ADNI scans differ between the ADNI and Union processed data")
    m.insert(0, "cohort", "ADNI")
    m = m.sort_values(["class", "split_adni", "scan_id"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)
    print(m.groupby(["class", "split_adni"]).size().to_string())
    print(m.groupby(["class", "split_union"]).size().to_string())


if __name__ == "__main__":
    main()
