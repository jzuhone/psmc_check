from ..psmc_check import model_path, PSMCCheck
from acis_thermal_check.regression_testing import \
    RegressionTester, all_loads
import pytest

psmc_rt = RegressionTester(PSMCCheck, model_path, "psmc_test_spec.json")

# ACIS state builder tests

psmc_rt.run_models(state_builder='sql')

# Prediction tests


@pytest.mark.parametrize('load', all_loads)
def test_prediction(answer_store, load):
    if not answer_store:
        psmc_rt.run_test("prediction", load)
    else:
        pass

# Validation tests


@pytest.mark.parametrize('load', all_loads)
def test_validation(answer_store, load):
    if not answer_store:
        psmc_rt.run_test("validation", load)
    else:
        pass
