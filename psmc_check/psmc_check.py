#!/usr/bin/env python

"""
========================
psmc_check
========================

This code generates backstop load review outputs for checking the ACIS
PSMC temperature 1PDEAAT.  It also generates PSMC model validation
plots comparing predicted values to telemetry for the previous three
weeks.
"""
from __future__ import print_function

# Matplotlib setup
# Use Agg backend for command-line (non-interactive) operation
import matplotlib
matplotlib.use('Agg')

from acis_thermal_check import \
    ACISThermalCheck, \
    get_options
import os
import sys

model_path = os.path.abspath(os.path.dirname(__file__))


class PSMCCheck(ACISThermalCheck):
    def __init__(self):
        valid_limits = {'1PDEAAT': [(1, 2.5), (50, 1.0), (99, 5.5)],
                        'PITCH': [(1, 3.0), (99, 3.0)],
                        'TSCPOS': [(1, 2.5), (99, 2.5)]
                       }
        hist_limit = [30., 40.]
        super(PSMCCheck, self).__init__("1pdeaat", "psmc", valid_limits,
                                        hist_limit, other_telem=['1dahtbon'],
                                        other_map={'1dahtbon': 'dh_heater'})


def main():
    args = get_options("psmc", model_path)
    psmc_check = PSMCCheck()
    try:
        psmc_check.run(args)
    except Exception as msg:
        if args.traceback:
            raise
        else:
            print("ERROR:", msg)
            sys.exit(1)


if __name__ == '__main__':
    main()
