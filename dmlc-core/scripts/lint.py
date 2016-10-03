#!/usr/bin/env python
# pylint: disable=protected-access, unused-variable, locally-disabled, redefined-variable-type
"""Lint helper to generate lint summary of source.

Copyright by Contributors
"""
from __future__ import print_function
import codecs
import sys
import re
import os
import cpplint
from cpplint import _cpplint_state
from pylint import epylint

CXX_SUFFIX = set(['cc', 'c', 'cpp', 'h', 'cu', 'hpp'])
PYTHON_SUFFIX = set(['py'])

class LintHelper(object):
    """Class to help runing the lint and records summary"""

    @staticmethod
    def _print_summary_map(strm, result_map, ftype):
        """Print summary of certain result map."""
        if len(result_map) == 0:
            return 0
        npass = len([x for k, x in result_map.iteritems() if len(x) == 0])
        strm.write('=====%d/%d %s files passed check=====\n' % (npass, len(result_map), ftype))
        for fname, emap in result_map.iteritems():
            if len(emap) == 0:
                continue
            strm.write('%s: %d Errors of %d Categories map=%s\n' % (
                fname, sum(emap.values()), len(emap), str(emap)))
        return len(result_map) - npass

    def __init__(self):
        self.project_name = None
        self.cpp_header_map = {}
        self.cpp_src_map = {}
        self.python_map = {}
        pylint_disable = ['superfluous-parens',
                          'too-many-instance-attributes',
                          'too-few-public-methods']
        # setup pylint
        self.pylint_opts = ['--extension-pkg-whitelist=numpy',
                            '--disable=' + ','.join(pylint_disable)]

        self.pylint_cats = set(['error', 'warning', 'convention', 'refactor'])
        # setup cpp lint
        cpplint_args = ['.', '--extensions=' + (','.join(CXX_SUFFIX))]
        _ = cpplint.ParseArguments(cpplint_args)
        cpplint._SetFilters(','.join(['-build/c++11',
                                      '-build/namespaces',
                                      '-build/include,',
                                      '+build/include_what_you_use',
                                      '+build/include_order']))
        cpplint._SetCountingStyle('toplevel')
        cpplint._line_length = 100

    def process_cpp(self, path, suffix):
        """Process a cpp file."""
        _cpplint_state.ResetErrorCounts()
        cpplint.ProcessFile(str(path), _cpplint_state.verbose_level)
        _cpplint_state.PrintErrorCounts()
        errors = _cpplint_state.errors_by_category.copy()

        if suffix == 'h':
            self.cpp_header_map[str(path)] = errors
        else:
            self.cpp_src_map[str(path)] = errors

    def process_python(self, path):
        """Process a python file."""
        (pylint_stdout, pylint_stderr) = epylint.py_run(
            ' '.join([str(path)] + self.pylint_opts), return_std=True)
        emap = {}
        print(pylint_stderr.read())
        for line in pylint_stdout:
            sys.stderr.write(line)
            key = line.split(':')[-1].split('(')[0].strip()
            if key not in self.pylint_cats:
                continue
            if key not in emap:
                emap[key] = 1
            else:
                emap[key] += 1
        sys.stderr.write('\n')
        self.python_map[str(path)] = emap

    def print_summary(self, strm):
        """Print summary of lint."""
        nerr = 0
        nerr += LintHelper._print_summary_map(strm, self.cpp_header_map, 'cpp-header')
        nerr += LintHelper._print_summary_map(strm, self.cpp_src_map, 'cpp-soruce')
        nerr += LintHelper._print_summary_map(strm, self.python_map, 'python')
        if nerr == 0:
            strm.write('All passed!\n')
        else:
            strm.write('%d files failed lint\n' % nerr)
        return nerr

# singleton helper for lint check
_HELPER = LintHelper()

def get_header_guard_dmlc(filename):
    """Get Header Guard Convention for DMLC Projects.

    For headers in include, directly use the path
    For headers in src, use project name plus path

    Examples: with project-name = dmlc
        include/dmlc/timer.h -> DMLC_TIMTER_H_
        src/io/libsvm_parser.h -> DMLC_IO_LIBSVM_PARSER_H_
    """
    fileinfo = cpplint.FileInfo(filename)
    file_path_from_root = fileinfo.RepositoryName()
    inc_list = ['include', 'api', 'wrapper']

    if file_path_from_root.find('src/') != -1 and _HELPER.project_name is not None:
        idx = file_path_from_root.find('src/')
        file_path_from_root = _HELPER.project_name +  file_path_from_root[idx + 3:]
    else:
        for spath in inc_list:
            prefix = spath + os.sep
            if file_path_from_root.startswith(prefix):
                file_path_from_root = re.sub('^' + prefix, '', file_path_from_root)
                break
    return re.sub(r'[-./\s]', '_', file_path_from_root).upper() + '_'

cpplint.GetHeaderGuardCPPVariable = get_header_guard_dmlc

def process(fname, allow_type):
    """Process a file."""
    fname = str(fname)
    arr = fname.rsplit('.', 1)
    if fname.find('#') != -1 or arr[-1] not in allow_type:
        return
    if arr[-1] in CXX_SUFFIX:
        _HELPER.process_cpp(fname, arr[-1])
    if arr[-1] in PYTHON_SUFFIX:
        _HELPER.process_python(fname)

def main():
    """Main entry function."""
    if len(sys.argv) < 3:
        print('Usage: <project-name> <filetype> <list-of-path to traverse>')
        print('\tfiletype can be python/cpp/all')
        exit(-1)
    _HELPER.project_name = sys.argv[1]
    file_type = sys.argv[2]
    allow_type = []
    if file_type == 'python' or file_type == 'all':
        allow_type += [x for x in PYTHON_SUFFIX]
    if file_type == 'cpp' or file_type == 'all':
        allow_type += [x for x in CXX_SUFFIX]
    allow_type = set(allow_type)
    if sys.version_info.major == 2 and os.name != 'nt':
        sys.stderr = codecs.StreamReaderWriter(sys.stderr,
                                               codecs.getreader('utf8'),
                                               codecs.getwriter('utf8'),
                                               'replace')
    for path in sys.argv[3:]:
        if os.path.isfile(path):
            process(path, allow_type)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    process(os.path.join(root, name), allow_type)

    nerr = _HELPER.print_summary(sys.stderr)
    sys.exit(nerr > 0)

if __name__ == '__main__':
    main()
