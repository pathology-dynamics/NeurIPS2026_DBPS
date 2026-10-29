"""Conditional Flow Matching (CFM) baseline -- vanilla / I-CFM.

Independent random coupling: each minibatch pairs x_0 ~ X_train with
x_1 ~ Y_train independently (Lipman 2023). No paired GT, no anchor,
no rho.

Velocity field u_theta(x_t, t) trained with linear path target
u_target = x_1 - x_0.
"""
from configs.base import *

method = "cfm_random"
controller_type = None
fm_strategy = "random"
train_neural_controller = True
report_anchor_adherence = False
gamma = None
rho = None


def make_train_workdir_name(c):
    return f"cfm_random{c.suffix}"


def make_sampling_workdir_name(c):
    return f"cfm_random{c.suffix}"


make_workdir_name = make_train_workdir_name
