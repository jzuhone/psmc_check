#!/usr/bin/env python
from setuptools import setup

entry_points = {'console_scripts': 'psmc_check = psmc_check.psmc_check:main'}

setup(name='psmc_check',
      packages=["psmc_check"],
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='ACIS Thermal Model for 1PDEAAT',
      author='John ZuHone',
      author_email='jzuhone@gmail.com',
      url='http://github.com/acisops/psmc_check',
      include_package_data=True,
      entry_points=entry_points,
      )
