"""Analytic UniDB endpoint bridge with random anchors (UniDB-random).

The anchor y* is drawn uniformly from Y_train per (eval_sample, repeat).
This is the same endpoint controller as DBPS at rho = 0.
"""
from configs.base import *

method = 'unidb_analytic_random'
controller_type = 'analytic_endpoint_random'
train_neural_controller = False
use_analytic_controller = True
report_anchor_adherence = False


def make_workdir_name(c):
    return f"unidb_analytic_random_gamma{c.gamma}{c.suffix}"
