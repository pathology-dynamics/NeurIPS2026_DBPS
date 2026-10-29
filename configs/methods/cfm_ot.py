"""Conditional Flow Matching (CFM) baseline -- OT-CFM (Tong 2024).

Mini-batch optimal-transport coupling: each minibatch pairs x_0 to
x_1 via Hungarian solve on squared Euclidean cost. Reduces training
variance vs vanilla CFM. Sampling time is identical to vanilla CFM
(ODE; the OT only affects training-time pairing).
"""
from configs.base import *

method = "cfm_ot"
controller_type = None
fm_strategy = "ot"
train_neural_controller = True
report_anchor_adherence = False
gamma = None
rho = None


def make_train_workdir_name(c):
    return f"cfm_ot{c.suffix}"


def make_sampling_workdir_name(c):
    return f"cfm_ot{c.suffix}"


make_workdir_name = make_train_workdir_name
