"""DBPS with K = 50 candidates for the population label."""
from configs.methods.dbps import *

method = "dbps_k50"
num_candidate_sample = 50


def make_workdir_name(c):
    return f"dbps_k50_gamma{c.gamma}{c.suffix}"


def make_sampling_workdir_name(c):
    ckpt_label = getattr(c, "checkpoint_label", "best")
    return f"dbps_k50_gamma{c.gamma}_ckpt{ckpt_label}{c.suffix}"
