"""Sinkhorn-OT-AGE baseline config: 1-D age cost matrix + Sinkhorn."""
from configs.base import *

method = 'sinkhorn_age'
controller_type = 'sinkhorn_age'
candidate_strategy = 'sinkhorn_age'
train_neural_controller = False
use_analytic_controller = False
sinkhorn_reg = 0.05


def make_workdir_name(c):
    return f"sinkhorn_age{c.suffix}"


def make_sampling_workdir_name(c):
    return f"sinkhorn_age{c.suffix}"
