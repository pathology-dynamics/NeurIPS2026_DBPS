"""CFM-random with age embedding."""
from configs.base import *

method = 'cfm_random_age'
controller_type = 'cfm_random_age'
candidate_strategy = 'cfm_random_age'
train_neural_controller = True

fm_strategy = 'random'
path_mode = 'linear'
use_sde_fm = False                       # CFM = ODE rollout
cond_on_age = True
age_emb_dim = 32


def make_workdir_name(c):
    return f"cfm_random_age{c.suffix}"


def make_sampling_workdir_name(c):
    return f"cfm_random_age{c.suffix}"
