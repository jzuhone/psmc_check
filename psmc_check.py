#!/usr/bin/env python

"""
========================
psmc_check
========================

This code generates backstop load review outputs for checking the ACIS
PSMC temperature 1PDEAAT.  It also generates 1PDEAAT model validation
plots comparing predicted values to telemetry for the previous three weeks.
"""

import sys
import os
import logging
import numpy as np
from numpy import ndarray
import Chandra.cmd_states as cmd_states
import Ska.Table
import Chandra.Time

# Matplotlib setup
# Use Agg backend for command-line (non-interactive) operation
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xija

from model_check import ModelCheck, calc_off_nom_rolls

MSID = dict(psmc='1PDEAAT')
YELLOW = dict(psmc=55.0)
MARGIN = dict(psmc=2.5)
VALIDATION_LIMITS = {'1PDEAAT': [(1, 2.5),
                                 (50, 1.0),
                                 (99, 5.5)],
                     'PITCH': [(1, 3.0),
                               (99, 3.0)],
                     'TSCPOS': [(1, 2.5),
                                (99, 2.5)]
                     }
HIST_LIMIT = [30.,40.]

TASK_DATA = os.path.dirname(__file__)
logger = logging.getLogger('psmc_check')

_versionfile = os.path.join(os.path.dirname(__file__), 'VERSION')
VERSION = open(_versionfile).read().strip()

def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.set_defaults()
    parser.add_option("--outdir",
                      default="out",
                      help="Output directory")
    parser.add_option("--oflsdir",
                       help="Load products OFLS directory")
    parser.add_option("--model-spec",
                      default=os.path.join(TASK_DATA, 'psmc_model_spec.json'),
                       help="PSMC model specification file")
    parser.add_option("--days",
                      type='float',
                      default=21.0,
                      help="Days of validation data (days)")
    parser.add_option("--run-start",
                      help="Reference time to replace run start time "
                           "for regression testing")
    parser.add_option("--traceback",
                      default=True,
                      help='Enable tracebacks')
    parser.add_option("--verbose",
                      type='int',
                      default=1,
                      help="Verbosity (0=quiet, 1=normal, 2=debug)")
    parser.add_option("--ccd-count",
                      type='int',
                      default=6,
                      help="Initial number of CCDs (default=6)")
    parser.add_option("--fep-count",
                      type='int',
                      default=6,
                      help="Initial number of FEPs (default=6)")
    parser.add_option("--vid-board",
                      type='int',
                      default=1,
                      help="Initial state of ACIS vid_board (default=1)")
    parser.add_option("--clocking",
                      type='int',
                      default=1,
                      help="Initial state of ACIS clocking (default=1)")
    parser.add_option("--simpos",
                      default=75616,
                      type='float',
                      help="Starting SIM-Z position (steps)")
    parser.add_option("--pitch",
                      default=150.0,
                      type='float',
                      help="Starting pitch (deg)")
    parser.add_option("--T-psmc",
                      type='float',
                      help="Starting 1PDEAAT temperature (degC)")
    parser.add_option("--T-pin1at",
                      type='float',
                      help="Starting 1PIN1AT temperature (degC)")
    # adding dh_heater
    parser.add_option("--dh_heater",
                      type='int',
                      default=0,
                      help="Starting Detector Housing Heater state")
    parser.add_option("--version",
                      action='store_true',
                      help="Print version")

    opt, args = parser.parse_args()
    return opt, args

def calc_model(model_spec, states, start, stop, T_psmc=None, T_psmc_times=None,
               T_pin1at=None,T_pin1at_times=None,
               dh_heater=None,dh_heater_times=None):
    model = xija.XijaModel('psmc', start=start, stop=stop, model_spec=model_spec)

    #set fetch to quiet if and only if verbose == 0
    if opt.verbose == 0:
        xija.logger.setLevel(100);

    times = np.array([states['tstart'], states['tstop']])
    model.comp['sim_z'].set_data(states['simpos'], times)
    #model.comp['eclipse'].set_data(False)
    model.comp['1pdeaat'].set_data(T_psmc, T_psmc_times)
    model.comp['pin1at'].set_data(T_pin1at,T_pin1at_times)
    model.comp['roll'].set_data(calc_off_nom_rolls(states), times)
    model.comp['eclipse'].set_data(False)

    # for name in ('ccd_count', 'fep_count', 'vid_board', 'clocking', 'pitch', 'dh_heater'):
    for name in ('ccd_count', 'fep_count', 'vid_board', 'clocking', 'pitch'):
        model.comp[name].set_data(states[name], times)

    model.comp['dh_heater'].set_data(dh_heater,dh_heater_times)

    model.make()
    model.calc()
    return model

class PSMCModelCheck(ModelCheck):

    def set_initial_state(self, tlm, db, t_msid):
        state0 = cmd_states.get_state0(tlm['date'][-5], db,
                                           datepar='datestart')
        ok = ((tlm['date'] >= state0['tstart'] - 700) &
              (tlm['date'] <= state0['tstart'] + 700))
        state0.update({t_msid: np.mean(tlm[self.msid][ok])})
        state0.update({'T_pin1at': np.mean(tlm['1pdeaat'][ok]) - 10.0 })
        return state0

    def calc_model_wrapper(self, opt, states, tstart, tstop, t_msid, state0=None):
        # htrbfn='/home/edgar/acis/thermal_models/dhheater_history/dahtbon_history.rdb'                                
        htrbfn='dahtbon_history.rdb'
        logger.info('Reading file of dahtrb commands from file %s' % htrbfn)
        htrb=Ska.Table.read_ascii_table(htrbfn,headerrow=2,headertype='rdb')
        dh_heater_times=Chandra.Time.date2secs(htrb['time'])
        dh_heater=htrb['dahtbon'].astype(bool)
        if state0 is None:
            start_msid = None
            start_pin = None
            dh_heater = None
            dh_heater_times = None
        else:
            start_msid = state0[t_msid]
            start_pin = state0['T_pin1at']
            # htrbfn='/home/edgar/acis/thermal_models/dhheater_history/dahtbon_history.rdb'                     
            htrbfn='dahtbon_history.rdb'
            logger.info('Reading file of dahtrb commands from file %s' % htrbfn)
            htrb=Ska.Table.read_ascii_table(htrbfn,headerrow=2,headertype='rdb')
            dh_heater_times=Chandra.Time.date2secs(htrb['time'])
            dh_heater=htrb['dahtbon'].astype(bool)
        return self.calc_model(opt.model_spec, states, tstart, tstop, T_psmc=start_msid,
                               T_psmc_times=None, T_pin1at=start_pin, T_pin1at_times=None,
                               dh_heater=dh_heater, dh_heater_times=dh_heater_times)

if __name__ == '__main__':
    opt, args = get_options()
    if opt.version:
        print VERSION
        sys.exit(0)

    try:
        psmc_check = PSMCModelCheck("1pdeaat", "psmc", MSID,
                                    YELLOW, MARGIN, VALIDATION_LIMITS,
                                    HIST_LIMIT, calc_model, VERSION,
                                    other_telem=['1dahtbon'],
                                    other_map={'1dahtbon': 'dh_heater'},
                                    other_opts=['T_pin1at','dh_heater'])
        psmc_check.driver(opt)
    except Exception, msg:
        if opt.traceback:
            raise
        else:
            print "ERROR:", msg
            sys.exit(1)
