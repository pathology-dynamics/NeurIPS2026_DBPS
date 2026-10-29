"""Method config for the u=0 baseline (uncontrolled SDE rollout).

No anchor, no candidates -- purely the base dynamics
    dX_t = -beta L X_t dt + sqrt(beta) dW_t
"""
from configs.base import *

method = "uncontrolled"
controller_type = "zero"
train_neural_controller = False
use_analytic_controller = False

make_workdir_name = lambda c: f"uncontrolled{c.suffix}"
