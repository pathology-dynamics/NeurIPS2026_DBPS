"""Shared default configuration for DBPS and all baselines.

Method configs in `configs/methods/` start with `from configs.base import *`
and override what they need. Configs are plain Python modules (attribute
access, e.g. `config.gamma`); sweep grids are tuples.

Numerics: control-law math runs in torch.float64; networks run in
torch.float32.
"""
from pathlib import Path

# ---- Data ---------------------------------------------------------------

dataset_name = "adni"            # adni | nacc | adni_nacc_union
use_normalization = True
processed_data_root = Path("data/processed/candidate_groups")
# Explicit path to a processed_data.npz; when None the path is derived from
# dataset_name (see utils.config_utils.resolve_data_path).
processed_data_path = None

# Evaluation pool: X_val and X_test are concatenated ("merged"); the target
# reference for SWD/FD/MMD is the merged Y_val + Y_test.
eval_split = "merged"

# ---- Method placeholders (overridden by method configs) -----------------

method = None
controller_type = None
train_neural_controller = False
use_analytic_controller = False

# Endpoint anchor y* and training candidates are drawn uniformly at random
# from Y_train.
candidate_strategy = "random10"

# ---- Sweep grids --------------------------------------------------------

rho_values = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0)
gamma_values = (100, 1000, 10000)

# ---- Dynamics and control -----------------------------------------------

gamma = 1000
rho = 0.5
beta = 0.2
g = 1.0
cov_regularization = 1e-6
num_candidate_sample = 128       # K, candidate-set size for the population label
force_anchor_in_candidates = True

# ---- Training -----------------------------------------------------------

batch_size = 128
epochs = 150
lr = 1e-3
seed = 42
hidden_dim = 2048
num_layers = 3
dropout = 0.0
time_emb_dim = 32
activation = "relu"
time_embedding_type = "raw"      # scalar t concatenated to the input
block_type = "plain"             # Linear -> activation blocks, no skip connections

# Age conditioning: when True, the controller also receives the source
# subject's age through (age - 70) / 10 -> Linear(1, age_emb_dim) -> ReLU.
cond_on_age = False
age_emb_dim = 32

weight_decay = 1e-6
adam_beta1 = 0.7
grad_clip = 0.8

# Checkpoints saved at evenly spaced epochs: floor(epochs / n) * (1, ..., n).
num_checkpoints = 25

# Training times are drawn from U(t_eps, 1 - t_eps).
t_eps = 1e-2

# ---- Evaluation ---------------------------------------------------------

# Per-checkpoint cheap evaluation (flow-matching baselines).
num_rollout_steps_cheap = 100
repeat_cheap = 10

# Checkpoint selection uses the training split only.
select_steps = 200
select_repeat = 40

# Final evaluation: Euler-Maruyama steps and independent sampling repeats.
num_rollout_steps_heavy = 400
repeat_heavy = 250

num_rollout_steps = num_rollout_steps_heavy
repeat = repeat_heavy
use_sde = True
save_trajectories = True
save_controls = True

reeval_top_k = None
reeval_seed = 42

# Source samples drawn per evaluation repeat.
eval_batch_size = 100

# ---- Output -------------------------------------------------------------

workdir_root = Path("outputs/adni")
suffix = "_normalization"

# Each method config sets these.
make_workdir_name = None
make_sampling_workdir_name = None
