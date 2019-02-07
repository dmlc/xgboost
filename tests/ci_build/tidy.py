#!/usr/bin/env python
import subprocess
import yaml
import json
from multiprocessing import Pool, cpu_count
import shutil
import os
import sys
import re
import argparse


def call(args):
    '''Subprocess run wrapper.'''
    completed = subprocess.run(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    error_msg = completed.stdout.decode('utf-8')
    matched = re.match('.*src.*warning.*', error_msg,
                       re.MULTILINE | re.DOTALL)
    if matched is None:
        return_code = 0
    else:
        return_code = 1
    return (completed.returncode, return_code, error_msg)


class ClangTidy(object):
    '''
    clang tidy wrapper.
    Args:
      gtest_path: Full path of Google Test library.
      cpp_lint: Run linter on C++ source code.
      cuda_lint: Run linter on CUDA source code.
    '''
    def __init__(self, gtest_path, cpp_lint, cuda_lint):
        self.gtest_path = gtest_path
        self.cpp_lint = cpp_lint
        self.cuda_lint = cuda_lint
        print('Using Google Test from {}'.format(self.gtest_path))
        print('Run linter on CUDA: ', self.cuda_lint)
        print('Run linter on C++:', self.cpp_lint)
        if not self.cpp_lint and not self.cuda_lint:
            raise ValueError('Both --cpp and --cuda are set to 0.')
        self.root_path = os.path.abspath(os.path.curdir)
        print('Project root:', self.root_path)
        self.cdb_path = os.path.join(self.root_path, 'cdb')

    def __enter__(self):
        if os.path.exists(self.cdb_path):
            shutil.rmtree(self.cdb_path)
        self._generate_cdb()
        return self

    def __exit__(self, *args):
        if os.path.exists(self.cdb_path):
            shutil.rmtree(self.cdb_path)

    def _generate_cdb(self):
        '''Run CMake to generate compilation database.'''
        os.mkdir(self.cdb_path)
        os.chdir(self.cdb_path)
        cmake_args = ['cmake', '..', '-DGENERATE_COMPILATION_DATABASE=ON',
                      '-DGOOGLE_TEST=ON', '-DGTEST_ROOT={}'.format(
                          self.gtest_path)]
        if self.cuda_lint:
            cmake_args.extend(['-DUSE_CUDA=ON', '-DUSE_NCCL=ON'])
        subprocess.run(cmake_args)
        os.chdir(self.root_path)

    def _configure_flags(self, path, command):
        common_args = ['clang-tidy',
                       "-header-filter='(xgboost\\/src|xgboost\\/include)'",
                       '-config='+str(self.clang_tidy)]
        common_args.append(path)
        common_args.append('--')

        command = command.split()[1:]  # remove clang/c++/g++
        # filter out not used flags
        if '-fuse-ld=gold' in command:
            command.remove('-fuse-ld=gold')
        if '-rdynamic' in command:
            command.remove('-rdynamic')
        if '-Xcompiler=-fPIC' in command:
            command.remove('-Xcompiler=-fPIC')
        if '-Xcompiler=-fPIE' in command:
            command.remove('-Xcompiler=-fPIE')
        if '-c' in command:
            index = command.index('-c')
            del command[index+1]
            command.remove('-c')
        if '-o' in command:
            index = command.index('-o')
            del command[index+1]
            command.remove('-o')

        common_args.extend(command)

        # Two passes, one for device code another for host code.
        if path.endswith('cu'):
            args = [common_args.copy(), common_args.copy()]
            args[0].append('--cuda-host-only')
            args[1].append('--cuda-device-only')
        else:
            args = [common_args.copy()]
        for a in args:
            a.append('-Wno-unused-command-line-argument')
        return args

    def _configure(self):
        '''Load and configure compile_commands and clang_tidy.'''

        def should_lint(path):
            if not self.cpp_lint and path.endswith('.cc'):
                return False
            isxgb = path.find('rabit') == -1
            isxgb = isxgb and path.find('dmlc-core') == -1
            if isxgb:
                return True

        cdb_file = os.path.join(self.cdb_path, 'compile_commands.json')
        with open(cdb_file, 'r') as fd:
            self.compile_commands = json.load(fd)
        tidy_file = os.path.join(self.root_path, '.clang-tidy')
        with open(tidy_file) as fd:
            self.clang_tidy = yaml.load(fd)
        all_files = []
        for entry in self.compile_commands:
            path = entry['file']
            if should_lint(path):
                print(path)
                args = self._configure_flags(path, entry['command'])
                all_files.extend(args)
        return all_files

    def run(self):
        '''Run clang-tidy.'''
        all_files = self._configure()
        passed = True
        BAR = '-'*32
        with Pool(cpu_count()) as pool:
            results = pool.map(call, all_files)
            for (process_status, tidy_status, msg) in results:
                # Don't enforce clang-tidy to pass for now due to namespace
                # for cub in thrust is not correct.
                if tidy_status == 1:
                    passed = False
                    print(BAR, '\n'
                          'Process return code:', process_status, ', ',
                          'Tidy result code:', tidy_status, ', ',
                          'Message:\n', msg,
                          BAR, '\n')
        if not passed:
            print('Please correct clang-tidy warnings.')
        return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run clang-tidy.')
    parser.add_argument('--cpp', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--gtest-path', required=True,
                        help='Full path of Google Test library directory')
    args = parser.parse_args()
    with ClangTidy(args.gtest_path, args.cpp, args.cuda) as linter:
        passed = linter.run()
    if not passed:
        sys.exit(1)
