# pylint: disable=invalid-name
"""Setup xgboost package."""
from __future__ import absolute_import
import sys
from setuptools import setup
sys.path.insert(0, '.')
import xgboost

LIB_PATH = xgboost.core.find_lib_path()

setup(name='xgboost',
      version=xgboost.__version__,
      description=xgboost.__doc__,
      install_requires=[
          'numpy',
          'scipy',
      ],
      zip_safe=False,
      packages=['xgboost'],
      data_files=[('xgboost', [LIB_PATH[0]])],
      url='https://github.com/dmlc/xgboost')
