"""Build the ADNI regional tau CSVs (CN source, MCI + AD target) from the raw ADNI tables.

Inputs (--raw_dir):
  UCBERKELEY_TAU_6MM_27Oct2025.csv   regional tau PET SUVRs
  DXSUM_27Oct2025.csv                diagnosis summary (1 = CN, 2 = MCI, 3 = AD)
  ADNIMERGE_09Dec2025.csv            baseline age (AGE) and years from baseline (Years_bl)

Steps:
  1. Diagnosis per scan: exact match on RID + VISCODE2; otherwise the DXSUM
     visit of the same RID closest in date (same day first; among equally
     close visits, a future visit is preferred, then the higher severity).
  2. Target cohort = MCI + AD. For impaired scans matched by visit date,
     the clinical visit must lie within TARGET_WINDOW_DAYS of the scan.
  3. Age at scan = AGE + Years_bl from ADNIMERGE, matched on RID + VISCODE2.
  4. Participants (PTID) with both CN and impaired scans are excluded so that
     the source and target cohorts are participant-disjoint.

Outputs (--out_dir): tau_CN_resolved_with_age.csv, tau_ADMCI_with_age.csv;
intermediate tables are written to <out_dir>/intermediate/.

usage:
  python -m scripts.prepare_adni_csv --raw_dir <ADNI downloads> --out_dir data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DX_FILE = "DXSUM_27Oct2025.csv"
TAU_FILE = "UCBERKELEY_TAU_6MM_27Oct2025.csv"
ADNIMERGE_FILE = "ADNIMERGE_09Dec2025.csv"
WINDOW_DAYS = 365000  # date tolerance for the nearest-visit match (effectively unlimited)
TARGET_WINDOW_DAYS = 90  # date tolerance for impaired (MCI + AD) scans

SEVERITY = {1: 0, 2: 1, 3: 2}  # CN < MCI < AD


def find_col(df, names):
    u = {c.upper(): c for c in df.columns}
    for n in names:
        if n.upper() in u:
            return u[n.upper()]
    return None


def norm_rid(s):  return pd.to_numeric(s, errors="coerce").astype("Int64")
def norm_date(s): return pd.to_datetime(s, errors="coerce", infer_datetime_format=True).dt.normalize()
def norm_vc2(s):  return s.astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)


def resolve_diagnosis(raw_dir: Path, work: Path) -> None:
    """Step 1: write tau_with_dx_resolved.csv and tau_CN_resolved.csv to `work`."""
    dx = pd.read_csv(raw_dir / DX_FILE, low_memory=False)
    tau = pd.read_csv(raw_dir / TAU_FILE, low_memory=False)

    dx_rid = find_col(dx, ["RID"]);        tau_rid = find_col(tau, ["RID"])
    dx_date = find_col(dx, ["EXAMDATE", "USERDATE", "VISDATE"])
    tau_date = find_col(tau, ["SCANDATE", "EXAMDATE", "EXAM_DATE", "SCAN_DATE", "USERDATE", "VISDATE"])
    dx_dx = find_col(dx, ["DIAGNOSIS", "DXCURREN", "DX"])
    dx_vc2 = find_col(dx, ["VISCODE2"]);   tau_vc2 = find_col(tau, ["VISCODE2"])
    if any(c is None for c in [dx_rid, dx_date, dx_dx, tau_rid, tau_date]):
        raise SystemExit("Missing required columns. Check your headers.")

    dx["_RID_"] = norm_rid(dx[dx_rid]);    tau["_RID_"] = norm_rid(tau[tau_rid])
    dx["_DATE_"] = norm_date(dx[dx_date]); tau["_DATE_"] = norm_date(tau[tau_date])
    dx["_VC2_"] = norm_vc2(dx[dx_vc2]) if dx_vc2 else pd.Series([None] * len(dx))
    tau["_VC2_"] = norm_vc2(tau[tau_vc2]) if tau_vc2 else pd.Series([None] * len(tau))

    # 1a) exact RID + VISCODE2
    dx_small = dx[["_RID_", "_VC2_", "_DATE_", dx_dx]].copy()
    m = tau.merge(dx_small, how="left", on=["_RID_", "_VC2_"], suffixes=("", "_dx1"))
    m["match_rule"] = np.where(m[dx_dx].notna(), "VISCODE2", "UNMATCHED")

    # 1b) nearest date within +/- WINDOW_DAYS for the remaining scans
    unmatched_idx = m.index[m[dx_dx].isna()].tolist()
    if unmatched_idx:
        left = m.loc[unmatched_idx, ["_RID_", "_DATE_"]].copy()
        left["row_id"] = unmatched_idx

        right = dx[["_RID_", "_DATE_", dx_dx]].dropna(subset=["_DATE_"]).copy()
        cand = left.merge(right, how="left", on="_RID_", suffixes=("_tau", "_dx"))
        cand = cand.dropna(subset=["_DATE__dx"])
        cand["delta_days"] = (cand["_DATE__dx"] - cand["_DATE__tau"]).dt.days
        cand = cand[cand["delta_days"].abs() <= WINDOW_DAYS]

        picked = []
        for rid_row, grp in cand.groupby("row_id"):
            # same-day visit first; several same-day labels -> highest severity
            same = grp[grp["delta_days"] == 0]
            if not same.empty:
                same = same.copy()
                same["sev"] = same[dx_dx].map(SEVERITY)
                row = same.sort_values(["sev"], ascending=False).head(1).iloc[0]
                picked.append((rid_row, row[dx_dx], row["delta_days"],
                               "DATE_SAME_DAY_MAX_SEVERITY" if len(same) > 1 else "DATE_SAME_DAY"))
                continue

            # nearest by |delta days|
            grp = grp.copy()
            grp["absd"] = grp["delta_days"].abs()
            grp = grp.sort_values(["absd"]).reset_index(drop=True)
            min_abs = grp["absd"].iloc[0]
            near = grp[grp["absd"] == min_abs]

            if len(near) == 1:
                row = near.iloc[0]
                rule = "DATE_NEAREST_FUTURE" if row["delta_days"] >= 0 else "DATE_NEAREST_PAST"
                picked.append((rid_row, row[dx_dx], row["delta_days"], rule))
            else:
                # tie: prefer a future visit, then the highest severity
                fut = near[near["delta_days"] >= 0]
                if not fut.empty:
                    fut = fut.copy(); fut["sev"] = fut[dx_dx].map(SEVERITY)
                    row = fut.sort_values(["sev"], ascending=False).head(1).iloc[0]
                    picked.append((rid_row, row[dx_dx], row["delta_days"], "DATE_NEAREST_PREF_FUTURE"))
                else:
                    pst = near.copy(); pst["sev"] = pst[dx_dx].map(SEVERITY)
                    row = pst.sort_values(["sev"], ascending=False).head(1).iloc[0]
                    picked.append((rid_row, row[dx_dx], row["delta_days"], "DATE_NEAREST_PAST_TIE_MAX_SEVERITY"))

        if picked:
            upd = pd.DataFrame(picked, columns=["row_id", "diag", "delta_days", "rule"]).set_index("row_id")
            m.loc[upd.index, dx_dx] = upd["diag"].values
            m.loc[upd.index, "match_rule"] = upd["rule"].values
            m.loc[upd.index, "dx_delta_days"] = upd["delta_days"].values

    m["DIAGNOSIS"] = m[dx_dx] if "DIAGNOSIS" not in m.columns else m["DIAGNOSIS"].fillna(m[dx_dx])
    m["diagnosis_label"] = m["DIAGNOSIS"].map({1: "CN/Normal", 2: "MCI", 3: "AD/Dementia"})

    m.to_csv(work / "tau_with_dx_resolved.csv", index=False)
    m[m["DIAGNOSIS"].astype("Int64") == 1].to_csv(work / "tau_CN_resolved.csv", index=False)
    print("Total tau rows:", len(tau), "| labeled rows:", int(m["DIAGNOSIS"].notna().sum()))


def build_target(work: Path) -> None:
    """Step 2: tau_ADMCI.csv = MCI + AD scans (date-matched visits within TARGET_WINDOW_DAYS)."""
    df = pd.read_csv(work / "tau_with_dx_resolved.csv", low_memory=False)
    diag = pd.to_numeric(df["DIAGNOSIS"], errors="coerce").astype("Int64")
    too_far = df["dx_delta_days"].abs() > TARGET_WINDOW_DAYS
    out = df[diag.isin([2, 3]) & ~too_far].copy()
    out.to_csv(work / "tau_ADMCI.csv", index=False)
    print(f"tau_ADMCI.csv rows={len(out)}")


def add_age(raw_dir: Path, in_path: Path, out_path: Path) -> None:
    """Step 3: merge AGE and Years_bl from ADNIMERGE and compute AGE_at_scan."""
    adni = pd.read_csv(raw_dir / ADNIMERGE_FILE, low_memory=False)
    adni_age = adni[["RID", "VISCODE", "AGE", "Years_bl"]].rename(columns={"VISCODE": "VISCODE_adni"})
    df = pd.read_csv(in_path, low_memory=False)
    df_with_age = df.merge(
        adni_age,
        left_on=["RID", "VISCODE2"],
        right_on=["RID", "VISCODE_adni"],
        how="left",
    )
    df_with_age["AGE_at_scan"] = df_with_age["AGE"] + df_with_age["Years_bl"]
    df_with_age.to_csv(out_path, index=False)
    print(f"{out_path.name} rows={len(df_with_age)}")


def exclude_participants_in_both(cn_path: Path, admci_path: Path,
                                 cn_out: Path, admci_out: Path,
                                 subject_col: str = "PTID") -> None:
    """Step 4: drop every scan of participants with both CN and impaired scans."""
    cn = pd.read_csv(cn_path, low_memory=False)
    admci = pd.read_csv(admci_path, low_memory=False)
    both = set(cn[subject_col].dropna()) & set(admci[subject_col].dropna())
    cn_keep = cn[~cn[subject_col].isin(both)]
    admci_keep = admci[~admci[subject_col].isin(both)]
    cn_keep.to_csv(cn_out, index=False)
    admci_keep.to_csv(admci_out, index=False)
    print(f"excluded {len(both)} participants with both CN and impaired scans: "
          f"CN {len(cn)} -> {len(cn_keep)}, impaired {len(admci)} -> {len(admci_keep)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    work = out_dir / "intermediate"
    work.mkdir(parents=True, exist_ok=True)

    resolve_diagnosis(raw_dir, work)
    build_target(work)
    add_age(raw_dir, work / "tau_CN_resolved.csv", work / "tau_CN_resolved_with_age_all.csv")
    add_age(raw_dir, work / "tau_ADMCI.csv", work / "tau_ADMCI_with_age_all.csv")
    exclude_participants_in_both(
        work / "tau_CN_resolved_with_age_all.csv", work / "tau_ADMCI_with_age_all.csv",
        out_dir / "tau_CN_resolved_with_age.csv", out_dir / "tau_ADMCI_with_age.csv",
    )


if __name__ == "__main__":
    main()
