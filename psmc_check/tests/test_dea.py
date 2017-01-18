import os
import shutil
import tempfile

from ..psmc_check import psmc_check, model_path
from acis_thermal_check.regression_testing import run_answer_test, \
    run_image_test, run_model

def psmc_test_template(generate_answers, run_start, load_week, 
                       cmd_states_db='sybase'):
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    model_spec = os.path.join(model_path, "psmc_model_spec.json")
    out_dir = run_model("psmc", psmc_check, model_spec, run_start, 
                        load_week, cmd_states_db)
    run_answer_test("psmc", load_week, out_dir, generate_answers)
    run_image_test("1pdeaat", "psmc", load_week, out_dir, generate_answers)
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

def test_psmc_may3016(generate_answers):
    run_start = "2016:122:12:00:00.000"
    load_week = "MAY3016"
    psmc_test_template(generate_answers, run_start, load_week)
