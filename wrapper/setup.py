import os
import platform
from setuptools import setup


class XGBoostLibraryNotFound(Exception):
    pass


curr_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = [curr_dir]

if os.name == 'nt':
    if platform.architecture()[0] == '64bit':
        dll_path.append(os.path.join(curr_dir, '../windows/x64/Release/'))
    else:
        dll_path.append(os.path.join(curr_dir, '../windows/Release/'))
        

if os.name == 'nt':
    dll_path = [os.path.join(p, 'xgboost_wrapper.dll') for p in dll_path]
else:
    dll_path = [os.path.join(p, 'libxgboostwrapper.so') for p in dll_path]

lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

if len(lib_path) == 0:
    raise XGBoostLibraryNotFound("XGBoost library not found. Did you run "
                                 "../make?")
setup(name="xgboost",
      version="0.40",
      description="Python wrappers for XGBoost: eXtreme Gradient Boosting",
      zip_safe=False,
      py_modules=['xgboost'],
      data_files=[('.', [lib_path[0]])],
      url="https://github.com/dmlc/xgboost")
