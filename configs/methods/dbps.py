"""DBPS: balanced endpoint-population controller (K = 128 candidates).

Network: u_theta(x_t, t, age).
Training label: u_pop = sum_k alpha_k u_end(x_t, t; y_k), with K candidates
drawn uniformly from Y_train.
Sampling: u = (1 - rho) u_end(x_t, t; y*) + rho u_theta(x_t, t, age).
One training run per gamma; rho is chosen at sampling time.
"""
from configs.base import *

method = "dbps"
controller_type = "unconditional_pop"
train_neural_controller = True
report_anchor_adherence = False
candidate_strategy = "random10"
cond_on_age = True
age_emb_dim = 32
sampling_formula = "interp"


def make_workdir_name(c):
    return f"dbps_gamma{c.gamma}{c.suffix}"


def make_sampling_workdir_name(c):
    ckpt_label = getattr(c, "checkpoint_label", "best")
    return f"dbps_gamma{c.gamma}_ckpt{ckpt_label}{c.suffix}"
