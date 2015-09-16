# pylint: disable=invalid-name
"""Setup xgboost package."""
from __future__ import absolute_import
import sys
from setuptools import setup, find_packages
import subprocess
sys.path.insert(0, '.')

import os
#build on the fly if install in pip
#otherwise, use build.sh in the parent directory

if 'pip' in __file__:
    if not os.name == 'nt': #if not windows
        build_sh = subprocess.Popen(['sh', 'xgboost/build-python.sh'])
        build_sh.wait()
        output = build_sh.communicate()
        print(output)

import xgboost

LIB_PATH = xgboost.core.find_lib_path()
#print LIB_PATH

#to deploy to pip, please use
#make pythonpack
#python setup.py register sdist upload
#and be sure to test it firstly using "python setup.py register sdist upload -r pypitest"
setup(name='xgboost',
      version=xgboost.__version__,
      #version='0.4a13',
      description=xgboost.__doc__,
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Hongliang Liu',
      maintainer_email='phunter.lau@gmail.com',
      zip_safe=False,
      packages=find_packages(),
      #don't need this and don't use this, give everything to MANIFEST.in
      #package_dir = {'':'xgboost'},
      #package_data = {'': ['*.txt','*.md','*.sh'],
      #               }
      #this will use MANIFEST.in during install where we specify additional files,
      #this is the golden line
      include_package_data=True,
      data_files=[('xgboost', LIB_PATH)],
      url='https://github.com/dmlc/xgboost')
