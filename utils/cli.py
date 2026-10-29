"""Command-line selection of the evaluation setting.

--setting adni            ADNI -> ADNI     (250 evaluation repeats)
--setting union           Union -> Union   (80 repeats)
--setting union_to_adni   Union -> ADNI    (200 repeats; Union training data, ADNI held-out pool)

The setting fixes the processed data file, the output root and the number of
evaluation repeats; --data / --workdir_root / --repeat override them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path("data/processed/candidate_groups")
SETTINGS = {
    "adni": {"dataset_name": "adni", "data": ROOT / "adni/normalization/processed_data.npz",
             "workdir_root": Path("outputs/adni"), "repeat": 250},
    "union": {"dataset_name": "adni_nacc_union", "data": ROOT / "adni_nacc_union/normalization/processed_data.npz",
              "workdir_root": Path("outputs/union"), "repeat": 80},
    "union_to_adni": {"dataset_name": "adni_nacc_union", "data": ROOT / "union_to_adni/normalization/processed_data.npz",
                      "workdir_root": Path("outputs/union_to_adni"), "repeat": 200},
}


def add_setting_args(parser) -> None:
    parser.add_argument("--setting", choices=sorted(SETTINGS), default=None)
    parser.add_argument("--data", default=None, help="override processed_data.npz")
    parser.add_argument("--workdir_root", default=None, help="override the output root")
    parser.add_argument("--repeat", type=int, default=None, help="override the number of evaluation repeats")


def apply_setting_args(config, args) -> None:
    if getattr(args, "setting", None):
        s = SETTINGS[args.setting]
        config.dataset_name = s["dataset_name"]
        config.processed_data_path = s["data"]
        config.workdir_root = s["workdir_root"]
        config.repeat_heavy = config.repeat = s["repeat"]
    if getattr(args, "data", None):
        config.processed_data_path = Path(args.data)
    if getattr(args, "workdir_root", None):
        config.workdir_root = Path(args.workdir_root)
    if getattr(args, "repeat", None):
        config.repeat_heavy = config.repeat = int(args.repeat)
