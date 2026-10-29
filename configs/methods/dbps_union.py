"""DBPS on the multi-cohort (Union) setting, K = 128 candidates.

Uses the pooled processed data built by scripts/preprocess_adni_nacc_union.py
(X = CN of both cohorts, Y = impaired of both cohorts).
"""
from pathlib import Path

from configs.methods.dbps import *

dataset_name = "adni_nacc_union"
processed_data_path = Path(
    "data/processed/candidate_groups/adni_nacc_union/normalization/processed_data.npz"
)
workdir_root = Path("outputs/union")

repeat_heavy = 80
repeat = repeat_heavy
