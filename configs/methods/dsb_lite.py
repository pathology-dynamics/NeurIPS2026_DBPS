"""DSB-lite baseline (Bridge Matching variant of De Bortoli 2021 DSB).

Single-pass approximation to Diffusion Schrödinger Bridge: train the
forward drift on Brownian-bridge interpolation between OT-paired
(x_0, x_1), then sample with SDE. Equivalent to one IPF iteration of
the full DSB algorithm, which is enough for a baseline reference.

Path: x_t = (1-t) x_0 + t x_1 + sigma * sqrt(t(1-t)) * eps, eps ~ N(0,I)
Target drift: (x_1 - x_t) / (1 - t)   (Brownian-bridge drift)
Sample: SDE with the trained drift, x_0 ~ X_eval, t: 0 -> 1.
"""
from configs.base import *

method = 'dsb_lite'
controller_type = 'dsb_lite'
candidate_strategy = 'dsb_lite'
train_neural_controller = True

fm_strategy = 'ot'                        # OT-paired pairs
path_mode = 'bridge'                      # Brownian bridge interpolation
sigma_bridge = 0.5                        # bridge noise amplitude
use_sde_fm = True                         # SDE sampling (matches Doob bridge SDE)


def make_workdir_name(c):
    return f"dsb_lite{c.suffix}"


def make_sampling_workdir_name(c):
    return f"dsb_lite{c.suffix}"
