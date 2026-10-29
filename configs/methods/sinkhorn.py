"""Sinkhorn-OT baseline config: barycentric transport X_eval -> Y_train."""
from configs.base import *

method = 'sinkhorn'
controller_type = 'sinkhorn'
candidate_strategy = 'sinkhorn'
train_neural_controller = False
use_analytic_controller = False
sinkhorn_reg = 0.05


def make_workdir_name(c):
    return f"sinkhorn{c.suffix}"


def make_sampling_workdir_name(c):
    return f"sinkhorn{c.suffix}"
