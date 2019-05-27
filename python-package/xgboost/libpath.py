# coding: utf-8
"""Find the path to xgboost dynamic library files."""

import os
import platform
import sys


class XGBoostLibraryNotFound(Exception):
    """Error thrown by when xgboost is not found"""


def find_lib_path():
    """Find the path to xgboost dynamic library files.

    Returns
    -------
    lib_path: list(string)
       List of all found library path to xgboost
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # make pythonpack hack: copy this directory one level upper for setup.py
    dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
                os.path.join(curr_path, './lib/'),
                os.path.join(sys.prefix, 'xgboost')]
    if sys.platform == 'win32':
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
        else:
            dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/Release/'))
        dll_path = [os.path.join(p, 'xgboost.dll') for p in dll_path]
    elif sys.platform.startswith('linux') or sys.platform.startswith('freebsd'):
        dll_path = [os.path.join(p, 'libxgboost.so') for p in dll_path]
    elif sys.platform == 'darwin':
        dll_path = [os.path.join(p, 'libxgboost.dylib') for p in dll_path]
    elif sys.platform == 'cygwin':
        dll_path = [os.path.join(p, 'cygxgboost.dll') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    # From github issues, most of installation errors come from machines w/o compilers
    if not lib_path and not os.environ.get('XGBOOST_BUILD_DOC', False):
        raise XGBoostLibraryNotFound(
            'Cannot find XGBoost Library in the candidate path, ' +
            'did you install compilers and run build.sh in root path?\n'
            'List of candidates:\n' + ('\n'.join(dll_path)))
    return lib_path
