"""Assemble the paper's result tables from the evaluation outputs.

Expected layout under --outputs (default: outputs/):
  adni/, union/                 training settings
    dbps_gamma{G}_normalization/eval/final_eval.csv          DBPS, K = 128
    dbps_k50_gamma{G}_normalization/eval/final_eval.csv      DBPS, K = 50
    <baseline>_normalization/baseline_metrics.csv
  union/dbps*/eval/final_eval_union_to_adni.csv             Union -> ADNI evaluation
  union_to_adni/<baseline>_normalization/baseline_metrics.csv

Writes Markdown tables to --tables (default: tables/):
  table2_main.md, table4_boundary.md, table7_age.md, table9_gamma.md,
  table11_k.md, table12_rho.md, and dbps_sweep.csv (all DBPS cells).

usage:
  python -m scripts.make_paper_tables --outputs outputs --tables tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GAMMAS = (100, 1000, 10000)
KS = {128: "dbps", 50: "dbps_k50"}
SETTINGS = (
    # key, label, DBPS training dir, DBPS eval file, baseline dir
    ("adni", "ADNI→ADNI", "adni", "final_eval.csv", "adni"),
    ("union", "Union→Union", "union", "final_eval.csv", "union"),
    ("union_to_adni", "Union→ADNI", "union", "final_eval_union_to_adni.csv", "union_to_adni"),
)
# Table 2 rows: (label, [candidate workdir names]); the lowest-SWD candidate is reported.
UNIDB = {k: [f"unidb_analytic_{k}_gamma{g}" for g in GAMMAS] for k in ("age", "random", "ot")}
TABLE2_ROWS = (
    ("Reference (non-learning)", None),
    ("Uncontrolled", ["uncontrolled"]),
    ("Mean-shift", ["meanshift"]),
    ("FKPP", ["fkpp"]),
    ("Diffusion bridges", None),
    ("UniDB-age", UNIDB["age"]),
    ("UniDB-random", UNIDB["random"]),
    ("DSB-lite", ["dsb_lite", "dsb_lite_age"]),
    ("Flow / OT / SOC matching", None),
    ("OT-CFM", ["cfm_ot", "cfm_ot_age"]),
    ("Sinkhorn-OT", ["sinkhorn", "sinkhorn_age"]),
    ("CFM", ["cfm_random", "cfm_random_age"]),
    ("SOC-Matching", ["soc_matching", "soc_matching_age"]),
)
AGE_PAIRS = (("OT-CFM", "cfm_ot"), ("Sinkhorn-OT", "sinkhorn"), ("CFM", "cfm_random"),
             ("SOC-Matching", "soc_matching"), ("DSB-lite", "dsb_lite"))


def swd(r):  return f"{r['swd_mean']:.3f} ± {r['swd_std']:.3f}"
def fd(r):   return f"{r['fd_mean']:.1f}"
def fdpm(r): return f"{r['fd_mean']:.1f} ± {r['fd_std']:.1f}"


def sci(x: float) -> str:
    mant, exp = f"{x:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def load_dbps(outputs: Path) -> pd.DataFrame:
    rows = []
    for key, _, train_dir, fname, _ in SETTINGS:
        for k, prefix in KS.items():
            for g in GAMMAS:
                f = outputs / train_dir / f"{prefix}_gamma{g}_normalization" / "eval" / fname
                if not f.is_file():
                    continue
                df = pd.read_csv(f)
                df["setting"], df["K"], df["gamma"] = key, k, g
                rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no DBPS evaluation files under {outputs}")
    return pd.concat(rows, ignore_index=True)


def load_baseline(outputs: Path, setting_dir: str, name: str):
    f = outputs / setting_dir / f"{name}_normalization" / "baseline_metrics.csv"
    return pd.read_csv(f).iloc[0].to_dict() if f.is_file() else None


def best_baseline(outputs, setting_dir, names):
    found = [(n, r) for n in names if (r := load_baseline(outputs, setting_dir, n)) is not None]
    return min(found, key=lambda t: t[1]["swd_mean"]) if found else (None, None)


def table2(outputs, dbps):
    head = "| Method | " + " | ".join(f"{lab} SWD ↓ | FD ↓ | MMD ↓" for _, lab, *_ in SETTINGS) + " |"
    lines = [head, "|---|" + "---|" * 9]
    for label, names in TABLE2_ROWS:
        if names is None:
            lines.append(f"| *{label}* |" + " |" * 9)
            continue
        cells = []
        for _, _, _, _, bdir in SETTINGS:
            _, r = best_baseline(outputs, bdir, names)
            cells += [swd(r), fd(r), sci(r["mmd"])] if r else ["–"] * 3
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for key, *_ in SETTINGS:
        r = dbps[dbps.setting == key].sort_values("swd_mean").iloc[0]
        cells += [swd(r), fd(r), sci(r["mmd"])]
    lines.append("| DBPS (ours) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def champion(dbps, key):
    return dbps[dbps.setting == key].sort_values("swd_mean").iloc[0]


def table4(dbps, k_boundary=128):
    """Boundary cases at the champion's gamma: rho = 0 and rho = 1 from the
    K = 128 sweep (as in Table 12), the interior row is the champion."""
    lines = ["| Setting (γ) | Method | ρ | SWD |", "|---|---|---|---|"]
    for key, lab, *_ in SETTINGS:
        c = champion(dbps, key)
        grid = dbps[(dbps.setting == key) & (dbps.gamma == c.gamma) & (dbps.K == k_boundary)]
        for name, r in (("UniDB", grid[grid.rho == 0.0].iloc[0]), ("DBPS (ours)", c),
                        ("Population", grid[grid.rho == 1.0].iloc[0])):
            lines.append(f"| {lab} (γ={int(c.gamma)}) | {name} | {r.rho:g} | {swd(r)} |")
    return "\n".join(lines)


def table7(outputs):
    lines = ["| Setting | Method | NO-AGE SWD | FD | MMD | +AGE SWD | FD | MMD |", "|---|---|---|---|---|---|---|---|"]
    for _, lab, _, _, bdir in SETTINGS:
        for name, base in AGE_PAIRS:
            a, b = load_baseline(outputs, bdir, base), load_baseline(outputs, bdir, base + "_age")
            fmt = lambda r: [swd(r), fdpm(r), sci(r["mmd"])] if r else ["–"] * 3
            lines.append(f"| {lab} | {name} | " + " | ".join(fmt(a) + fmt(b)) + " |")
    return "\n".join(lines)


def table9(outputs, dbps):
    lines = ["| Setting | Method | γ=100 | γ=1,000 | γ=10,000 |", "|---|---|---|---|---|"]
    for key, lab, _, _, bdir in SETTINGS:
        for kind in ("age", "random", "ot"):
            vals = []
            for g in GAMMAS:
                r = load_baseline(outputs, bdir, f"unidb_analytic_{kind}_gamma{g}")
                vals.append(f"{r['swd_mean']:.3f}" if r else "–")
            lines.append(f"| {lab} | UniDB ({kind.upper() if kind == 'ot' else kind}) | " + " | ".join(vals) + " |")
        vals = [f"{dbps[(dbps.setting == key) & (dbps.gamma == g)].swd_mean.min():.3f}" for g in GAMMAS]
        lines.append(f"| {lab} | DBPS (ours) | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def table11(dbps):
    lines = ["| Setting | K | best (γ, ρ) | SWD | FD | MMD |", "|---|---|---|---|---|---|"]
    for key, lab, *_ in SETTINGS:
        for k in (50, 128):
            r = dbps[(dbps.setting == key) & (dbps.K == k)].sort_values("swd_mean").iloc[0]
            lines.append(f"| {lab} | {k} | ({int(r.gamma)}, {r.rho:g}) | {swd(r)} | {fd(r)} | {sci(r['mmd'])} |")
    return "\n".join(lines)


def table12(dbps, k=128):
    rhos = sorted(dbps.rho.unique())
    lines = ["| Setting | γ | " + " | ".join(f"ρ={r:g}" for r in rhos) + " |", "|---|---|" + "---|" * len(rhos)]
    for key, lab, *_ in SETTINGS:
        for g in GAMMAS:
            cell = dbps[(dbps.setting == key) & (dbps.gamma == g) & (dbps.K == k)].set_index("rho")
            vals = [f"{cell.loc[r, 'swd_mean']:.3f}" if r in cell.index else "–" for r in rhos]
            lines.append(f"| {lab} | {g} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--tables", default="tables")
    args = ap.parse_args()
    outputs, tables = Path(args.outputs), Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)

    dbps = load_dbps(outputs)
    dbps.to_csv(tables / "dbps_sweep.csv", index=False)
    for name, text in (("table2_main", table2(outputs, dbps)), ("table4_boundary", table4(dbps)),
                       ("table7_age", table7(outputs)), ("table9_gamma", table9(outputs, dbps)),
                       ("table11_k", table11(dbps)), ("table12_rho", table12(dbps))):
        (tables / f"{name}.md").write_text(text + "\n")
        print(f"## {name}\n{text}\n")


if __name__ == "__main__":
    main()
