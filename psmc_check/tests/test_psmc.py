from ..psmc_check import model_path, PSMCCheck
from acis_thermal_check.regression_testing import \
    RegressionTester, all_loads
import pytest

psmc_rt = RegressionTester(PSMCCheck, model_path)

# Prediction tests


@pytest.mark.parametrize('load', all_loads)
def test_prediction(answer_store, load):
    psmc_rt.run_test("prediction", answer_store, load)

# Validation tests


@pytest.mark.parametrize('load', all_loads)
def test_validation(answer_store, load):
    psmc_rt.run_test("validation", answer_store, load)