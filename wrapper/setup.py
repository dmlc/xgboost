import os

from setuptools import setup


class XGBoostLibraryNotFound(Exception):
    pass


cur_dir = os.path.dirname(os.path.abspath(__file__))

if os.name == 'nt':
    dll_path = os.path.join(cur_dir,
                            '../windows/x64/Release/xgboost_wrapper.dll')
else:
    dll_path = os.path.join(cur_dir, 'libxgboostwrapper.so')

if not os.path.exists(dll_path):
    raise XGBoostLibraryNotFound("XGBoost library not found. Did you run "
                                 "../make?")

setup(name="xgboost",
      version="0.32",
      description="Python wrappers for XGBoost: eXtreme Gradient Boosting",
      zip_safe=False,
      py_modules=['xgboost'],
      data_files=[dll_path],
      url="https://github.com/dmlc/xgboost")
