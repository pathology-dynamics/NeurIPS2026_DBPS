#!/usr/bin/env bash
# Table 13 / Appendix I.3: retrain DBPS (ADNI, gamma = 100, K = 128, seed 42) on
# permuted and rewired anatomical graphs, select the checkpoint on the training
# split at rho = 0.1, evaluate on the held-out pool, and compute the Braak
# alignment (d u / d age at t = 0.5).
#
# usage: bash scripts/run_graph_controls.sh [output_root]
set -euo pipefail

OUT=${1:-outputs/graph_controls}
SRC=data/processed/candidate_groups/adni/normalization/processed_data.npz
GRAPHS=data/random_graphs
CFG=configs/methods/dbps.py

python -m scripts.build_random_graphs --src "$SRC" --out_dir "$GRAPHS"

for cond in permuted rewired; do
  for seed in 101 202 303 404 505; do
    tag=${cond}_seed${seed}
    data=$GRAPHS/${tag}.npz
    root=$OUT/$tag
    python -m training.train_db --config $CFG --gamma 100 --use_normalization --data "$data" --workdir_root "$root"
    run=$root/dbps_gamma100_normalization
    python -m scripts.run_dbps_eval --config $CFG --gamma 100 --run_dir "$run" --rho 0.1 --data "$data"
    ckpt=$(python -c "import pandas as p; print(p.read_csv('$run/eval/final_eval.csv').checkpoint[0])")
    python -m scripts.braak_alignment --config $CFG --gamma 100 --data "$data" \
        --checkpoint "$run/checkpoints/ckpt_${ckpt}.pth" --estimand dudage --ages imputed --t 0.5 \
        --out "$root/braak"
  done
done
