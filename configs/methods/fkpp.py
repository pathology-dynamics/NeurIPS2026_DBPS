"""FKPP baseline (Chaggar 2025): reaction-diffusion PDE on graph."""
from configs.base import *

method = 'fkpp'
controller_type = 'fkpp'
candidate_strategy = 'fkpp'
train_neural_controller = False
use_analytic_controller = False

# Hyperparameters
fkpp_rho = 0.1                           # diffusion coefficient (Chaggar 2025 default range)
fkpp_num_steps = 100                     # Euler integration steps
fkpp_noise_scale = 0.02                  # small Gaussian noise for SWD-friendly per-repeat variance


def make_workdir_name(c):
    return f"fkpp{c.suffix}"


def make_sampling_workdir_name(c):
    return f"fkpp{c.suffix}"
