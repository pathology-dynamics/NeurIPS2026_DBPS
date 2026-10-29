"""DSB-lite (Bridge Matching) with age embedding."""
from configs.base import *

method = 'dsb_lite_age'
controller_type = 'dsb_lite_age'
candidate_strategy = 'dsb_lite_age'
train_neural_controller = True

fm_strategy = 'ot'
path_mode = 'bridge'
sigma_bridge = 0.5
use_sde_fm = True
cond_on_age = True
age_emb_dim = 32


def make_workdir_name(c):
    return f"dsb_lite_age{c.suffix}"


def make_sampling_workdir_name(c):
    return f"dsb_lite_age{c.suffix}"
