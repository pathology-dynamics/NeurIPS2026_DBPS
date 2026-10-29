"""Table 6: scan counts, participant counts and demographics per cohort.

Scans are those in the processed data (after the missing-value filter).
Demographics are scan-weighted; each variable is summarized over scans with
an available value (missing values excluded). Age is age at scan; sex and
APOE e4 carrier status (>= 1 e4 allele) come from the clinical tables.

Inputs:
  --data        directory with processed_data.npz files (data/processed/candidate_groups)
  --adni_raw    directory with ADNIMERGE_09Dec2025.csv
  --adni_csv    the ADNI CSVs written by scripts.prepare_adni_csv (for LONIUID -> PTID / RID)
  --nacc_raw    directory with investigator_scan_taupetnpdka_nacc72.csv and investigator_nacc72.csv

usage:
  python -m scripts.make_demographics_table --adni_raw <ADNI downloads> --nacc_raw <NACC downloads>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _pool(npz, prefix):
    ids = np.concatenate([npz[f"{prefix}_{s}_ids"] for s in SPLITS]).astype(str)
    ages = np.concatenate([npz[f"{prefix}_{s}_ages"] for s in SPLITS]).astype(float)
    return ids, ages


def _row(label, df):
    pct = lambda s: f"{100 * s.dropna().mean():.1f}"
    age = df["age"].dropna()
    return (f"| {label} | {len(df):,} | {df['subject'].nunique():,} | "
            f"{age.mean():.1f} ± {age.std(ddof=1):.1f} | {pct(df['female'])} | {pct(df['e4'])} |")


def adni_frames(data: Path, adni_raw: Path, adni_csv: Path):
    npz = np.load(data / "adni/normalization/processed_data.npz", allow_pickle=True)
    ids_map = pd.concat([pd.read_csv(adni_csv / f, usecols=["LONIUID", "PTID", "RID"], low_memory=False)
                         for f in ("tau_CN_resolved_with_age.csv", "tau_ADMCI_with_age.csv")])
    ids_map = ids_map.drop_duplicates("LONIUID").assign(LONIUID=lambda d: d.LONIUID.astype(str)).set_index("LONIUID")
    am = pd.read_csv(adni_raw / "ADNIMERGE_09Dec2025.csv", usecols=["RID", "PTGENDER", "APOE4"], low_memory=False)
    sex = am.dropna(subset=["PTGENDER"]).groupby("RID").PTGENDER.first().map({"Female": 1.0, "Male": 0.0})
    apoe = am.dropna(subset=["APOE4"]).groupby("RID").APOE4.first()
    out = []
    for prefix in ("X", "Y"):
        ids, ages = _pool(npz, prefix)
        rid = ids_map.loc[ids, "RID"].values
        out.append(pd.DataFrame({
            "subject": ids_map.loc[ids, "PTID"].values, "age": ages,
            "female": pd.Series(rid).map(sex).values,
            "e4": (pd.Series(rid).map(apoe) >= 1).where(pd.Series(rid).map(apoe).notna()).astype(float).values,
        }))
    return out


def nacc_frames(data: Path, nacc_raw: Path):
    npz = np.load(data / "nacc/normalization/processed_data.npz", allow_pickle=True)
    scans = pd.read_csv(nacc_raw / "investigator_scan_taupetnpdka_nacc72.csv",
                        usecols=["LONIUID", "NACCID"], dtype=str).drop_duplicates("LONIUID").set_index("LONIUID")
    uds = pd.read_csv(nacc_raw / "investigator_nacc72.csv", usecols=["NACCID", "SEX", "NACCNE4S"], low_memory=False)
    sex = uds.dropna(subset=["SEX"]).groupby("NACCID").SEX.first().map({2: 1.0, 1: 0.0})
    apoe = uds[uds.NACCNE4S.isin([0, 1, 2])].groupby("NACCID").NACCNE4S.first()
    out = []
    for prefix in ("X", "Y"):
        ids, ages = _pool(npz, prefix)
        subj = scans.loc[ids, "NACCID"].values
        e4 = pd.Series(subj).map(apoe)
        out.append(pd.DataFrame({
            "subject": subj, "age": ages, "female": pd.Series(subj).map(sex).values,
            "e4": (e4 >= 1).where(e4.notna()).astype(float).values,
        }))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/candidate_groups")
    ap.add_argument("--adni_raw", required=True)
    ap.add_argument("--adni_csv", default="data")
    ap.add_argument("--nacc_raw", required=True)
    ap.add_argument("--out", default="tables/table6_demographics.md")
    args = ap.parse_args()
    data = Path(args.data)
    adni_cn, adni_imp = adni_frames(data, Path(args.adni_raw), Path(args.adni_csv))
    nacc_cn, nacc_imp = nacc_frames(data, Path(args.nacc_raw))
    lines = ["| Cohort | Scans | Subjects | Age | Female (%) | APOE ε4 carrier (%) |",
             "|---|---|---|---|---|---|",
             _row("ADNI CN", adni_cn), _row("ADNI ADMCI", adni_imp),
             _row("NACC CN", nacc_cn), _row("NACC ADMCI", nacc_imp)]
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
