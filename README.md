# DBPS: Doob-Bridge Posterior Sampler

Code for [*DBPS: Doob-Bridge Posterior Sampler with Balanced Endpoint-Population Control for Unpaired Neurodegenerative Pathology Transport*](https://openreview.net/forum?id=GLIYSpGCJP) (NeurIPS 2026).

DBPS learns a stochastic controller that transports regional tau PET profiles of cognitively normal (CN) subjects to the impaired (ADMCI) population along a graph-diffusion prior. The controller balances a closed-form endpoint Doob bridge and a learned population controller with a weight ρ:

    u_DBPS(x, t; y*) = (1 − ρ) u_end(x, t; y*) + ρ u_θ(x, t, age)

## Installation

```bash
conda env create -f environment.yml
conda activate dbps
```

All commands are run from the repository root.

## Data

ADNI and NACC data cannot be redistributed. Obtain them from [ADNI](https://adni.loni.usc.edu) and [NACC](https://naccdata.org) under their data use agreements:

- ADNI: `UCBERKELEY_TAU_6MM_27Oct2025.csv`, `DXSUM_27Oct2025.csv`, `ADNIMERGE_09Dec2025.csv`
- NACC: `investigator_scan_taupetnpdka_nacc72.csv`, `investigator_nacc72.csv`

The repository includes the two non-restricted inputs: `data/TauRegionList.csv` (the 84 regions of interest) and `data/connectomes/*.csv` (five group-level 84×84 connectomes, averaged into the graph Laplacian). Both are taken unchanged from [multimodal_ndm](https://github.com/ethompson93/multimodal_ndm) ([Thompson et al., 2024](https://doi.org/10.1162/imag_a_00089)); the connectomes are derived from the MICA-MICs dataset ([Royer et al., 2022](https://doi.org/10.1038/s41597-022-01682-y)). Please cite both works if you use these files.

```bash
python -m scripts.prepare_adni_csv --raw_dir <ADNI files> --out_dir data
python -m scripts.prepare_nacc_csv --raw_dir <NACC files> --template data/tau_CN_resolved_with_age.csv --out_dir data
python scripts/preprocess_candidate_groups.py --cohort adni --output-variant normalization
python scripts/preprocess_candidate_groups.py --cohort nacc --output-variant normalization
python -m scripts.preprocess_adni_nacc_union
python -m scripts.build_union_to_adni
```

The ADNI scan list and splits are not distributed. To check that your splits match ours, write the manifest and compare its hash:

```bash
python -m scripts.make_split_manifest --out splits/adni_manifest.csv
sha256sum splits/adni_manifest.csv
# 51c0d53e5650b8f446812fc6e1eb507c401a4ef54ea24702752aa9533e2aaeac
```

## Settings

| Setting | Training data | Evaluation pool | Repeats | Flag |
|---|---|---|---|---|
| ADNI→ADNI | ADNI | ADNI held-out | 250 | `--setting adni` (default) |
| Union→Union | ADNI ∪ NACC | Union held-out | 80 | `--setting union` |
| Union→ADNI | ADNI ∪ NACC | ADNI held-out | 200 | `--setting union_to_adni` (evaluation only) |

Each evaluation repeat draws 100 held-out sources and 100 held-out targets with a generator seeded by `42 + 7919 r`. Checkpoints are selected on the training split (200 rollout steps × 40 repeats).

## DBPS

Train one model per γ ∈ {100, 1000, 10000} and K ∈ {128, 50}; ρ is chosen at sampling time.

```bash
# ADNI→ADNI
for g in 100 1000 10000; do
  python -m training.train_db --config configs/methods/dbps.py     --gamma $g --use_normalization
  python -m training.train_db --config configs/methods/dbps_k50.py --gamma $g --use_normalization
  python -m scripts.run_dbps_eval --config configs/methods/dbps.py     --gamma $g --run_dir outputs/adni/dbps_gamma${g}_normalization
  python -m scripts.run_dbps_eval --config configs/methods/dbps_k50.py --gamma $g --run_dir outputs/adni/dbps_k50_gamma${g}_normalization
done

# Union→Union, then Union→ADNI (same models, ADNI held-out pool, selection reused)
for cfg in dbps_union dbps_union_k50; do
  run_name=$([ $cfg = dbps_union ] && echo dbps || echo dbps_k50)
  for g in 100 1000 10000; do
    python -m training.train_db --config configs/methods/$cfg.py --gamma $g --use_normalization
    run=outputs/union/${run_name}_gamma${g}_normalization
    python -m scripts.run_dbps_eval --config configs/methods/$cfg.py --gamma $g --run_dir $run
    python -m scripts.run_dbps_eval --config configs/methods/$cfg.py --gamma $g --run_dir $run \
        --eval_data data/processed/candidate_groups/union_to_adni/normalization/processed_data.npz \
        --eval_repeat 200 --tag union_to_adni --reuse_selection
  done
done
```

`run_dbps_eval` writes `eval/selection_train.csv`, `eval/final_eval[_tag].csv` (SWD, FD, MMD per ρ) and the per-repeat SWD used by the paired tests.

## Baselines

Analytic baselines (Uncontrolled, Mean-shift, FKPP, Sinkhorn-OT, UniDB with random / age / OT anchors):

```bash
S=adni   # adni | union | union_to_adni
for m in uncontrolled meanshift fkpp sinkhorn sinkhorn_age; do
  python -m sampling.sample_$m --config configs/methods/$m.py --use_normalization --setting $S
  python -m scripts.eval_baseline --workdir outputs/$S/${m}_normalization --setting $S
done
for kind in random age ot; do
  for g in 100 1000 10000; do
    python -m sampling.sample_unidb_$kind --config configs/methods/unidb_$kind.py --gamma $g --use_normalization --setting $S
    python -m scripts.eval_baseline --workdir outputs/$S/unidb_analytic_${kind}_gamma${g}_normalization --setting $S
  done
done
```

Learned baselines (CFM, OT-CFM, SOC-Matching, DSB-lite, each with and without age conditioning; full DSBM):

```bash
S=adni   # adni | union
for m in cfm_random cfm_random_age cfm_ot cfm_ot_age soc_matching soc_matching_age dsb_lite dsb_lite_age; do
  python -m training.train_fm --config configs/methods/$m.py --use_normalization --setting $S
  wd=outputs/$S/${m}_normalization
  ckpt=$(python -m scripts.select_ckpt_trainset --config configs/methods/$m.py --workdir $wd \
           --use_normalization --setting $S | sed -n 's/^TRAIN_SELECTED_CKPT=//p')
  python -m scripts.sample_baseline --config configs/methods/$m.py --source_workdir $wd --ckpt $ckpt --setting $S
  python -m scripts.eval_baseline --workdir $wd --setting $S
  # Union→ADNI: apply the Union checkpoint to the ADNI held-out pool
  # python -m scripts.sample_baseline --config configs/methods/$m.py --source_workdir $wd --ckpt $ckpt --setting union_to_adni
  # python -m scripts.eval_baseline --workdir outputs/union_to_adni/${m}_normalization --setting union_to_adni
done
# DSBM: training.train_dsbm with configs/methods/dsbm.py, then the same selection / sampling / evaluation steps
```

## Paper tables and figures

```bash
python -m scripts.make_paper_tables --outputs outputs --tables tables        # Tables 2, 4, 7, 9, 11, 12
python -m scripts.run_paired_tests  --outputs outputs --tables tables        # Table 10
python -m scripts.make_rho_sweep_figures --sweep tables/dbps_sweep.csv --out figures   # Figures 3-5
python -m scripts.make_demographics_table --adni_raw <ADNI files> --nacc_raw <NACC files>   # Table 6

# Table 3: control energy and endpoint-anchor dispersion of the selected DBPS model (per setting)
python -m scripts.compute_control_costs --config configs/methods/dbps.py --gamma 100 --rho 0.1 \
    --checkpoint outputs/adni/dbps_gamma100_normalization/checkpoints/ckpt_<epoch>.pth --out tables/table3_adni.json
```

Section 5:

```bash
# Braak alignment (Figure 2, Section 5.1, Appendix I.1-I.2)
python -m scripts.braak_alignment --config configs/methods/dbps.py --gamma 100 \
    --checkpoint <checkpoint> --estimand dudage --ages imputed --t 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --out results/braak_adni
python -m scripts.braak_alignment --config configs/methods/dbps_union.py --gamma 10000 --rho 0.3 \
    --checkpoint <checkpoint> --estimand endpoint_shift --ages real --out results/braak_union
python -m scripts.make_braak_figure --per_roi results/braak_union/per_roi_endpoint_shift.csv --out figures/figure2.png

python -m scripts.rank_alignment     --config configs/methods/dbps.py --gamma 100 --rho 0.1 --checkpoint <checkpoint> --out results/rank_adni   # Section 5.2
python -m scripts.age_stratified_gap --config configs/methods/dbps.py --gamma 100 --rho 0.1 --checkpoint <checkpoint> --out results/table5  # Table 5

bash scripts/run_graph_controls.sh   # Table 13: randomized anatomical graphs
```

## Repository layout

```
configs/      base.py (shared defaults), data_process.py (preprocessing), methods/ (DBPS and baselines)
controls/     closed-form endpoint controller and population label
dynamics/     graph-diffusion transition kernels and Euler-Maruyama rollout
models/       controller network
training/     DBPS (train_db), flow-matching baselines (train_fm), DSBM (train_dsbm)
sampling/     DBPS evaluation protocol (sample_db) and analytic baselines
evaluation/   SWD / FD / MMD and Braak-stage analysis
matching/, dataset/   candidate groups and graph Laplacian used in preprocessing
scripts/      data preparation, evaluation drivers, tables and figures
```

## Citation

```bibtex
@inproceedings{guo2026dbps,
  title     = {{DBPS}: Doob-Bridge Posterior Sampler with Balanced Endpoint-Population Control for Unpaired Neurodegenerative Pathology Transport},
  author    = {Guo, Xinyu and Xie, Jiajia and He, Xin and Ye, Christophe and Nursal, Batuhan and Xue, Hongyu and Mitchell, Cassie S.},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=GLIYSpGCJP}
}
```

## License

Code released under the [MIT License](LICENSE). The files in `data/` are third-party data and are not covered by this license. The ADNI and NACC data remain subject to their own data use agreements.
