"""DSBM baseline -- Diffusion Schroedinger Bridge Matching (Shi et al., 2023).

Full IMF (iterated Markovian fitting): forward + backward drift networks,
alternating bridge-matching fits, and an endpoint coupling REGENERATED from the
current learned process at every outer iteration. Trained by
training/train_dsbm.py.

This config is deliberately a near-clone of dsb_lite.py so that any
measured difference is attributable to the IMF mechanism alone. Everything
inherited from configs.base (ControllerNet width/depth/activation/time embedding,
optimizer family, lr, batch_size, weight_decay, adam_beta1, grad_clip, t_eps,
splits, seed, eval protocol) is untouched.

The ONE deviation from dsb_lite is `g`: dsb_lite regresses a sigma=0.5 bridge
but samples with g=1.0 (the configs.base default), i.e. it simulates a different
path measure than it fit. That is tolerable for a one-pass method and fatal for
IMF -- the reciprocal projection must simulate exactly the diffusion the
Markovian projection regressed, or the iteration has no fixed point. So
g = sigma_bridge here, and train_dsbm.py hard-fails if they disagree.
"""
from configs.base import *

method = "dsbm"
controller_type = "dsbm"
candidate_strategy = "dsbm"
train_neural_controller = True
report_anchor_adherence = False
gamma = None
rho = None

# --- held identical to dsb_lite ---
fm_strategy = "ot"                 # iteration-1 coupling = minibatch Hungarian OT
path_mode = "bridge"               # Brownian bridge interpolation
sigma_bridge = 0.5                 # reference bridge noise
use_sde_fm = True                  # SDE sampling (Euler-Maruyama)

# --- required for IMF consistency (see docstring) ---
g = sigma_bridge

# Regression target for the Markovian projection.
#   "endpoint" -- the net predicts x1; drift = (x1_hat - x_t)/(1 - t) analytically.
#   "drift"    -- the net predicts the drift directly, exactly as dsb_lite does.
# Same network and loss family either way; only the target changes. "drift" is
# what dsb_lite uses and is kept switchable so the two can be compared, but it
# is NOT usable for IMF: the drift at t = 1 - 1/N has magnitude N*||dx||, a
# regime training never covers, so every Markovian projection over-disperses by
# 10-25%, the next half-step is then fit on a wider input law than it is applied
# to, and the two marginals swap within ~10 outer iterations (verified on a
# synthetic Gaussian pair, where it additionally blows up to 1e7 by iteration 8).
dsbm_parameterization = "endpoint"

# Per-element bound on the reconstructed drift during rollout. This is NOT a
# DSBM-specific hyperparameter: it defaults to controls.db_control.U_STAR_CLAMP
# (= 20.0), the identical per-element control clamp this repo already applies to
# DBPS's own u* (controls/db_control.py). It is needed because the endpoint
# parameterisation's drift (x1_hat - x_t)/(1 - t) is unbounded off-manifold.
# Set to 0 to disable.
dsbm_drift_clamp = 20.0

# --- DSBM-specific schedule ---
# Outer IMF iterations. Cap; the reported iteration is chosen from these by
# scripts/select_ckpt_trainset.py on the TRAIN split only.
dsbm_outer_iters = 20
# Gradient epochs per direction per outer iteration. dsb_lite trains 150 epochs
# once; DSBM therefore uses strictly MORE total updates. The exact cumulative
# count is logged every outer iteration in dsbm_trajectory.csv so the budget
# difference is explicit rather than hidden.
dsbm_inner_epochs = 30
# Euler steps used to regenerate the coupling (reciprocal projection). Matched
# to the heavy sampling discretisation so the coupling is not built from a
# coarser process than the one evaluated.
dsbm_coupling_steps = 400
# Cheap protocol for the per-iteration trajectory table only. Final numbers
# always come from scripts/reeval_baseline.py at the heavy protocol.
dsbm_traj_steps = 200
dsbm_traj_repeat = 10

cond_on_age = False


def make_train_workdir_name(c):
    return f"dsbm{c.suffix}"


def make_sampling_workdir_name(c):
    return f"dsbm{c.suffix}"


make_workdir_name = make_train_workdir_name
