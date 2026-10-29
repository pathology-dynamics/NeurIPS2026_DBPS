"""Analytic UniDB endpoint bridge with OT-paired anchors (UniDB-OT).

The anchor y* for each evaluation source is the OT top-1 target from
preprocessing (`ot_top_indices_eval[:, 0]`). Sweeps gamma; one workdir per gamma.
"""
from configs.base import *

method = 'unidb_analytic_ot'
controller_type = 'analytic_endpoint_ot'
train_neural_controller = False
use_analytic_controller = True
report_anchor_adherence = False


def make_workdir_name(c):
    return f"unidb_analytic_ot_gamma{c.gamma}{c.suffix}"
