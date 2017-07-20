# pylint: disable=invalid-name, exec-used, no-self-use, missing-docstring
"""Setup xgboost package."""
from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages, Distribution
# import subprocess
sys.path.insert(0, '.')

# this script is for packing and shipping pip installation
# it builds xgboost code on the fly and packs for pip
# please don't use this file for installing from github

if os.name != 'nt':     # if not windows, compile and install
    os.system('sh ./xgboost/build-python.sh')
else:
    print('Windows users please use github installation.')
    sys.exit()

CURRENT_DIR = os.path.dirname(__file__)


class BinaryDistribution(Distribution):
    """Auxilliary class necessary to inform setuptools that this is a
    non-generic, platform-specific package."""
    def has_ext_modules(self):
        return True


# We can not import `xgboost.libpath` in setup.py directly since xgboost/__init__.py
# import `xgboost.core` and finally will import `numpy` and `scipy` which are setup
# `install_requires`. That's why we're using `exec` here.
libpath_py = os.path.join(CURRENT_DIR, 'xgboost/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

LIB_PATH = libpath['find_lib_path']()

# to deploy to pip, please use
# make pythonpack
# python setup.py register sdist upload
# and be sure to test it firstly using "python setup.py register sdist upload -r pypitest"
setup(name='xgboost',
      # version=open(os.path.join(CURRENT_DIR, 'xgboost/VERSION')).read().strip(),
      version='0.6a2',
      description='XGBoost Python Package',
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Hongliang Liu',
      maintainer_email='phunter.lau@gmail.com',
      zip_safe=False,
      packages=find_packages(),
      # don't need this and don't use this, give everything to MANIFEST.in
      # package_dir = {'':'xgboost'},
      # package_data = {'': ['*.txt','*.md','*.sh'],
      #               }
      # this will use MANIFEST.in during install where we specify additional files,
      # this is the golden line
      include_package_data=True,
      # !!! don't use data_files for creating pip installation,
      # otherwise install_data process will copy it to
      # root directory for some machines, and cause confusions on building
      # data_files=[('xgboost', LIB_PATH)],
      distclass=BinaryDistribution,
      url='https://github.com/dmlc/xgboost')
