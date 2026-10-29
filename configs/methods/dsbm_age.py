"""DSBM-AGE -- DSBM with source-age conditioning.

Identical to dsbm_config in every respect except that both the forward and the
backward drift network additionally consume the source subject's age through
ControllerNet's cond_on_age path ((age-70)/10 -> Linear(1,32) -> act), the same
mechanism every other _age learning baseline uses.

Age bookkeeping through the IMF loop:
  * forward fit  -- age of the coupling's source endpoint;
  * backward fit -- age of the real X_train subject the pair was generated from;
  * backward simulation from Y_train produces a SYNTHETIC source subject, which
    is assigned an age drawn from the empirical X_train age distribution, so the
    regenerated coupling still carries a source age;
  * heavy evaluation -- the real X_eval subject ages, exactly as in every other
    _age baseline.

Per the staged protocol, this variant is run at the outer-iteration count
already selected by the non-AGE run; it does not re-search that count.
"""
from configs.methods.dsbm import *

method = "dsbm_age"
controller_type = "dsbm_age"
candidate_strategy = "dsbm_age"
cond_on_age = True
age_emb_dim = 32


def make_train_workdir_name(c):
    return f"dsbm_age{c.suffix}"


def make_sampling_workdir_name(c):
    return f"dsbm_age{c.suffix}"


make_workdir_name = make_train_workdir_name
