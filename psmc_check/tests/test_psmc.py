from ..psmc_check import VALIDATION_LIMITS, \
    HIST_LIMIT, calc_model, model_path
from acis_thermal_check.regression_testing import \
    RegressionTester

atc_kwargs = {"other_telem": ['1dahtbon'],
              "other_map": {'1dahtbon': 'dh_heater'}}

psmc_rt = RegressionTester("1pdeaat", "psmc", model_path, VALIDATION_LIMITS,
                           HIST_LIMIT, calc_model, atc_kwargs=atc_kwargs)

def test_psmc_loads(answer_store):
    psmc_rt.run_test_arrays(answer_store)