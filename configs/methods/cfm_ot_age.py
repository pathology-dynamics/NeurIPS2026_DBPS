"""OT-CFM with age embedding."""
from configs.base import *

method = 'cfm_ot_age'
controller_type = 'cfm_ot_age'
candidate_strategy = 'cfm_ot_age'
train_neural_controller = True

fm_strategy = 'ot'
path_mode = 'linear'
use_sde_fm = False
cond_on_age = True
age_emb_dim = 32


def make_workdir_name(c):
    return f"cfm_ot_age{c.suffix}"


def make_sampling_workdir_name(c):
    return f"cfm_ot_age{c.suffix}"
