from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def get_config():
    return {
        "cohort_name": "adni",
        "use_normalization": True,
        "train_fraction": 0.7,
        "eval_fraction": 0.2,
        "test_fraction": 0.1,
        "seed": 42,
        "max_missing_fraction": 0.2,
        "std_eps": 1e-6,
        "missing_impute_strategy": "mean_from_x_train",
        "roi_list_path": str(DATA_ROOT / "TauRegionList.csv"),
        "dimension": 84,
        "laplacian_mode": "mean_connectomes",
        "laplacian_normalize": True,
        "connectome_glob": str(DATA_ROOT / "connectomes" / "*.csv"),
        "laplacian_matrix_path": None,
        "beta": 0.2,
        "eig_eps": 1e-8,
        "cov_regularization": 1e-6,
        "unc_cost_type": "mahalanobis",
        "ot_cost_type": "dynamics_mahalanobis",
        "ot_epsilon": 0.05,
        "ot_max_iter": 500,
        "ot_tolerance": 1e-7,
        "ot_transfer_k": 32,
        "ot_transfer_temperature": 1.0,
        "candidate_group_size": 50,
        "num_uncontrolled": 25,
        "num_ot": 15,
        "num_random": 10,
        "matching_chunk_size": 128,
        "cohorts": {
            "adni": {
                "source_csv_path": str(DATA_ROOT / "tau_CN_resolved_with_age.csv"),
                "target_csv_path": str(DATA_ROOT / "tau_ADMCI_with_age.csv"),
                "sample_id_col": "LONIUID",
                "subject_id_col": "PTID",
                "scan_date_col": "SCANDATE",
                "age_col": "AGE_at_scan",
                "diagnosis_col": "diagnosis_label",
                "sex_col": None,
            },
            "nacc": {
                "source_csv_path": str(DATA_ROOT / "nacc_tau_CN_resolved_with_age.csv"),
                "target_csv_path": str(DATA_ROOT / "nacc_tau_ADMCI_with_age.csv"),
                "sample_id_col": "LONIUID",
                "subject_id_col": "LONIUID",
                "scan_date_col": "SCANDATE",
                "age_col": "AGE_at_scan",
                "diagnosis_col": "DIAGNOSIS",
                "sex_col": None,
            },
        },
    }
