"""UniDB-analytic-AGE baseline config: SDE rollout with age-paired y*."""
from configs.base import *

method = 'unidb_analytic_age'
controller_type = 'unidb_analytic_age'
candidate_strategy = 'age_top1'
train_neural_controller = False
use_analytic_controller = True


def make_workdir_name(c):
    return f"unidb_analytic_age_gamma{c.gamma}{c.suffix}"


def make_sampling_workdir_name(c):
    return f"unidb_analytic_age_gamma{c.gamma}{c.suffix}"
