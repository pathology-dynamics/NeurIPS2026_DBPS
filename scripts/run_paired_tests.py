"""Table 10: paired t-tests of per-repeat SWD, DBPS champion vs. every baseline.

Repeats are paired by source seed: repeat r of every method uses the same
source and target batches (generator seeded with 42 + 7919 r). For each
baseline configuration, Delta = mean(baseline) - mean(DBPS) (positive favours
DBPS) and the two-sided p-value comes from scipy.stats.ttest_rel.

Inputs (layout as in scripts/make_paper_tables.py):
  tables/dbps_sweep.csv                              to locate the champion (K, gamma, rho)
  <dbps run>/eval/per_repeat[_<tag>].npz             champion per-repeat SWD (scripts.run_dbps_eval)
  <baseline workdir>/swd_per_repeat.npy              per-repeat SWD (scripts.eval_baseline)

usage:
  python -m scripts.run_paired_tests --outputs outputs --sweep tables/dbps_sweep.csv --tables tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.make_paper_tables import KS, SETTINGS

LABELS = {
    "uncontrolled": "Uncontrolled", "meanshift": "Mean-shift", "fkpp": "FKPP",
    "sinkhorn": "Sinkhorn-OT (no-AGE)", "sinkhorn_age": "Sinkhorn-OT-AGE",
    "cfm_ot": "OT-CFM (no-AGE)", "cfm_ot_age": "OT-CFM-AGE",
    "cfm_random": "CFM-random", "cfm_random_age": "CFM-random-AGE",
    "soc_matching": "SOC-Matching (no-AGE)", "soc_matching_age": "SOC-Matching-AGE",
    "dsb_lite": "DSB-lite", "dsb_lite_age": "DSB-lite-AGE",
}
for kind, name in (("random", "UniDB-random"), ("age", "UniDB-age"), ("ot", "UniDB-OT")):
    for g in (100, 1000, 10000):
        LABELS[f"unidb_analytic_{kind}_gamma{g}"] = f"{name} (γ={g})"


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def paired_tests(outputs: Path, sweep: pd.DataFrame, key: str) -> pd.DataFrame:
    _, _, train_dir, eval_file, bdir = next(s for s in SETTINGS if s[0] == key)
    champ = sweep[sweep.setting == key].sort_values("swd_mean").iloc[0]
    tag = eval_file.replace("final_eval", "").replace(".csv", "")
    run = outputs / train_dir / f"{KS[int(champ.K)]}_gamma{int(champ.gamma)}_normalization"
    a = np.load(run / "eval" / f"per_repeat{tag}.npz")[f"rho{float(champ.rho):g}"]
    rows = []
    for name, label in LABELS.items():
        f = outputs / bdir / f"{name}_normalization" / "swd_per_repeat.npy"
        if not f.is_file():
            continue
        b = np.load(f)
        if b.shape != a.shape:
            raise ValueError(f"{f}: {b.shape} repeats vs DBPS {a.shape}")
        t, p = stats.ttest_rel(b, a)
        rows.append({"baseline": label, "delta": float((b - a).mean()), "t": float(t),
                     "p": float(p), "sig": stars(float(p)), "n": int(a.size)})
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--sweep", default="tables/dbps_sweep.csv")
    ap.add_argument("--tables", default="tables")
    args = ap.parse_args()
    outputs, tables = Path(args.outputs), Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    sweep = pd.read_csv(args.sweep)
    lines = ["| Setting | Baseline | Δ | t | p | |", "|---|---|---|---|---|---|"]
    for key, lab, *_ in SETTINGS:
        df = paired_tests(outputs, sweep, key)
        df.to_csv(tables / f"table10_paired_{key}.csv", index=False)
        for r in df.itertuples():
            lines.append(f"| {lab} (n={r.n}) | {r.baseline} | {r.delta:+.3f} | {r.t:+.2f} | {r.p:.1e} | {r.sig} |")
    text = "\n".join(lines)
    (tables / "table10_paired.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
