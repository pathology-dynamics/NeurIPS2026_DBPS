"""SOC Matching with age embedding."""
from configs.base import *

method = 'soc_matching_age'
controller_type = 'soc_matching_age'
candidate_strategy = 'soc_matching_age'
train_neural_controller = True

fm_strategy = 'ot'
path_mode = 'linear'
use_sde_fm = True
cond_on_age = True
age_emb_dim = 32


def make_workdir_name(c):
    return f"soc_matching_age{c.suffix}"


def make_sampling_workdir_name(c):
    return f"soc_matching_age{c.suffix}"
