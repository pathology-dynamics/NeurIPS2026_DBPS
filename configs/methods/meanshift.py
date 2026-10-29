"""Mean-shift baseline config: x_1 = X_eval + (mu_Y - mu_X)."""
from configs.base import *

method = 'meanshift'
controller_type = 'meanshift'
candidate_strategy = 'meanshift'
train_neural_controller = False
use_analytic_controller = False


def make_workdir_name(c):
    return f"meanshift{c.suffix}"


def make_sampling_workdir_name(c):
    return f"meanshift{c.suffix}"
