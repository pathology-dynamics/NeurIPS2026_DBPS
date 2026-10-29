"""Build the NACC regional tau CSVs (CN source, impaired target) from the raw NACC tables.

Inputs (--raw_dir):
  investigator_scan_taupetnpdka_nacc72.csv   regional tau PET SUVRs
  investigator_nacc72.csv                    clinical visits (NACCUDSD, birth year/month)
Template (--template): the ADNI CN CSV produced by scripts/prepare_adni_csv.py;
the outputs use exactly its column order.

Each scan takes the diagnosis of the clinical visit closest in date (a future
visit is preferred among equally close visits, then the higher severity).
Age at scan is computed from the birth year/month (day 15).

Outputs (--out_dir): nacc_tau_CN_resolved_with_age.csv, nacc_tau_ADMCI_with_age.csv

usage:
  python -m scripts.prepare_nacc_csv --raw_dir <NACC downloads> \
      --template data/tau_CN_resolved_with_age.csv --out_dir data
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

TAU_FILE = "investigator_scan_taupetnpdka_nacc72.csv"
NACC_MAIN_FILE = "investigator_nacc72.csv"
OUT_CN = "nacc_tau_CN_resolved_with_age.csv"
OUT_ADMCI = "nacc_tau_ADMCI_with_age.csv"

CHUNKSIZE = 200000
TREAT_UDSD2_AS_MCI = True


def parse_date_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    if out.isna().any():
        out = out.fillna(pd.to_datetime(s, errors="coerce", format="%m/%d/%Y"))
    if out.isna().any():
        out = out.fillna(pd.to_datetime(s, errors="coerce", infer_datetime_format=True))
    return out


def build_birthdate(birthyr: pd.Series, birthmo: pd.Series) -> pd.Series:
    y = pd.to_numeric(birthyr, errors="coerce")
    m = pd.to_numeric(birthmo, errors="coerce")
    return pd.to_datetime(dict(year=y, month=m, day=15), errors="coerce")


def map_diag_from_naccudsd(naccudsd: float, treat_udsd2_as_mci: bool) -> float:
    """1 = CN, 2 = MCI, 3 = dementia; NaN if unknown."""
    if pd.isna(naccudsd):
        return np.nan
    try:
        x = int(naccudsd)
    except Exception:
        return np.nan
    if x == 1:
        return 1
    if x == 3:
        return 2
    if x == 4:
        return 3
    if x == 2 and treat_udsd2_as_mci:
        return 2
    return np.nan


def fill_like_template(df: pd.DataFrame, template_cols: list) -> pd.DataFrame:
    return pd.DataFrame({c: (df[c] if c in df.columns else np.nan) for c in template_cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.template, "r", newline="") as f:
        template_cols = next(csv.reader(f))
    if not {"AGE", "Years_bl", "AGE_at_scan"} <= set(template_cols):
        raise RuntimeError("template must contain AGE, Years_bl, AGE_at_scan")

    tau = pd.read_csv(raw_dir / TAU_FILE, dtype={"NACCID": "string"}, low_memory=False)
    tau = tau[tau["TRACER"] == 6].copy()          # flortaucipir scans
    tau["NACCID"] = tau["NACCID"].astype(str)
    tau["SCANDATE_dt"] = parse_date_series(tau["SCANDATE"])
    tau_ids = set(tau["NACCID"].dropna().unique())

    need_cols = ["NACCID", "VISITYR", "VISITMO", "VISITDAY", "NACCUDSD", "BIRTHYR", "BIRTHMO"]
    parts = []
    reader = pd.read_csv(raw_dir / NACC_MAIN_FILE, usecols=lambda c: c in need_cols,
                         dtype={"NACCID": "string"}, chunksize=CHUNKSIZE, low_memory=False)
    for chunk in reader:
        chunk["NACCID"] = chunk["NACCID"].astype(str)
        chunk = chunk[chunk["NACCID"].isin(tau_ids)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise RuntimeError("No matching participants between the tau and clinical tables.")
    vis = pd.concat(parts, ignore_index=True)

    for c in ["VISITYR", "VISITMO", "VISITDAY", "NACCUDSD", "BIRTHYR", "BIRTHMO"]:
        vis[c] = pd.to_numeric(vis[c], errors="coerce")
    vis["VISITDATE_dt"] = pd.to_datetime(
        dict(year=vis["VISITYR"], month=vis["VISITMO"], day=vis["VISITDAY"]), errors="coerce")

    births = vis[["NACCID", "BIRTHYR", "BIRTHMO"]].dropna(subset=["BIRTHYR", "BIRTHMO"]).drop_duplicates("NACCID")
    births["BIRTHDATE_dt"] = build_birthdate(births["BIRTHYR"], births["BIRTHMO"])
    births = births[["NACCID", "BIRTHDATE_dt"]]

    base = vis.dropna(subset=["VISITDATE_dt"])[["NACCID", "VISITDATE_dt"]].sort_values(["NACCID", "VISITDATE_dt"])
    base = base.groupby("NACCID", as_index=False).first().rename(columns={"VISITDATE_dt": "BASEDATE_dt"})

    vis["DIAGNOSIS"] = vis["NACCUDSD"].apply(lambda x: map_diag_from_naccudsd(x, bool(TREAT_UDSD2_AS_MCI)))
    vis["severity"] = vis["DIAGNOSIS"]
    vis = vis.sort_values(["NACCID", "VISITDATE_dt"])
    groups = {k: g for k, g in vis.groupby("NACCID", sort=False)}

    diag_list = []
    for _, r in tau.iterrows():
        g = groups.get(r["NACCID"])
        scand = r["SCANDATE_dt"]
        if g is None or pd.isna(scand):
            diag_list.append(np.nan)
            continue
        deltas = (g["VISITDATE_dt"] - scand).dt.days
        absd = deltas.abs()
        min_abs = absd.min()
        cand = g[absd == min_abs].copy()
        cand["delta_days"] = deltas[absd == min_abs].values
        if (cand["delta_days"] >= 0).any():
            cand = cand[cand["delta_days"] >= 0]
        cand = cand.sort_values(["severity"], ascending=False)
        diag_list.append(cand.iloc[0]["DIAGNOSIS"])
    tau["DIAGNOSIS"] = diag_list

    tau = tau.merge(births, on="NACCID", how="left")
    tau = tau.merge(base, on="NACCID", how="left")
    tau["AGE_at_scan"] = (tau["SCANDATE_dt"] - tau["BIRTHDATE_dt"]).dt.days / 365.25
    tau["Years_bl"] = (tau["SCANDATE_dt"] - tau["BASEDATE_dt"]).dt.days / 365.25
    tau["AGE"] = tau["AGE_at_scan"] - tau["Years_bl"]

    out_cn = fill_like_template(tau[tau["DIAGNOSIS"] == 1].copy(), template_cols)
    out_admci = fill_like_template(tau[tau["DIAGNOSIS"].isin([2, 3])].copy(), template_cols)
    out_cn.to_csv(out_dir / OUT_CN, index=False)
    out_admci.to_csv(out_dir / OUT_ADMCI, index=False)
    print(f"CN rows={len(out_cn)}, impaired rows={len(out_admci)}")


if __name__ == "__main__":
    main()
