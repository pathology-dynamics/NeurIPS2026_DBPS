"""SOC Matching baseline (Domingo-Enrich 2024).

Same training as OT-CFM (linear interpolation, velocity target) but
sampling uses SDE Euler-Maruyama with diffusion g, matching the
stochastic-control formulation: dx_t = u_theta(x_t,t) dt + g dW.
This implements the path-matching variant of SOC: the network is
trained on OT-paired velocity targets, and the SDE sampling realizes
the optimal control as an instantaneous drift.
"""
from configs.base import *

method = 'soc_matching'
controller_type = 'soc_matching'
candidate_strategy = 'soc_matching'
train_neural_controller = True

fm_strategy = 'ot'                        # OT-paired (x_0, x_1) within minibatch
path_mode = 'linear'                      # linear interpolation
use_sde_fm = True                         # SDE sampling (key vs OT-CFM)
# g, num_rollout_steps_*, repeat_*, eval_batch_size inherited from configs.base


def make_workdir_name(c):
    return f"soc_matching{c.suffix}"


def make_sampling_workdir_name(c):
    return f"soc_matching{c.suffix}"
