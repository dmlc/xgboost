# coding: utf-8
"""Find the path to xgboost dynamic library files."""

import os
import platform


class XGBoostLibraryNotFound(Exception):
    """Error throwed by when xgboost is not found"""
    pass


def find_lib_path():
    """Load find the path to xgboost dynamic library files.

    Returns
    -------
    lib_path: list(string)
       List of all found library path to xgboost
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # make pythonpack hack: copy this directory one level upper for setup.py
    dll_path = [curr_path, os.path.join(curr_path, '../../wrapper/'),
                os.path.join(curr_path, './wrapper/')]
    if os.name == 'nt':
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
        else:
            dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/Release/'))
    if os.name == 'nt':
        dll_path = [os.path.join(p, 'xgboost_wrapper.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'libxgboostwrapper.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    #From github issues, most of installation errors come from machines w/o compilers
    if len(lib_path) == 0 and not os.environ.get('XGBOOST_BUILD_DOC', False):
        raise XGBoostLibraryNotFound(
            'Cannot find XGBoost Libarary in the candicate path, ' +
            'did you install compilers and run build.sh in root path?\n'
            'List of candidates:\n' + ('\n'.join(dll_path)))
    return lib_path
