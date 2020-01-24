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
    matched = re.search('(src|tests)/.*warning:', error_msg,
                        re.MULTILINE)
    if matched is None:
        return_code = 0
    else:
        return_code = 1
    return (completed.returncode, return_code, error_msg)


class ClangTidy(object):
    ''' clang tidy wrapper.
    Args:
      args:  Command line arguments.
          cpp_lint: Run linter on C++ source code.
          cuda_lint: Run linter on CUDA source code.
          use_dmlc_gtest: Whether to use gtest bundled in dmlc-core.
    '''
    def __init__(self, args):
        self.cpp_lint = args.cpp
        self.cuda_lint = args.cuda
        self.use_dmlc_gtest = args.use_dmlc_gtest

        print('Run linter on CUDA: ', self.cuda_lint)
        print('Run linter on C++:', self.cpp_lint)
        print('Use dmlc gtest:', self.use_dmlc_gtest)

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
        cmake_args = ['cmake', '..', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                      '-DGOOGLE_TEST=ON']
        if self.use_dmlc_gtest:
            cmake_args.append('-DUSE_DMLC_GTEST=ON')
        else:
            cmake_args.append('-DUSE_DMLC_GTEST=OFF')

        if self.cuda_lint:
            cmake_args.extend(['-DUSE_CUDA=ON', '-DUSE_NCCL=ON'])
        subprocess.run(cmake_args)
        os.chdir(self.root_path)

    def convert_nvcc_command_to_clang(self, command):
        '''Convert nvcc flags to corresponding clang flags.'''
        components = command.split()
        compiler: str = components[0]
        if compiler.find('nvcc') != -1:
            compiler = 'clang++'
            components[0] = compiler
        # check each component in a command
        converted_components = [compiler]

        for i in range(len(components)):
            if components[i] == '-lineinfo':
                continue
            elif components[i] == '-fuse-ld=gold':
                continue
            elif components[i] == '-rdynamic':
                continue
            elif (components[i] == '-x' and
                  components[i+1] == 'cu'):
                # -x cu -> -x cuda
                converted_components.append('-x')
                converted_components.append('cuda')
                components[i+1] = ''
                continue
            elif components[i].find('-Xcompiler') != -1:
                continue
            elif components[i].find('--expt') != -1:
                continue
            elif components[i].find('-ccbin') != -1:
                continue
            elif components[i].find('--generate-code') != -1:
                keyword = 'code=sm'
                pos = components[i].find(keyword)
                capability = components[i][pos + len(keyword) + 1:
                                           pos + len(keyword) + 3]
                if pos != -1:
                    converted_components.append(
                        '--cuda-gpu-arch=sm_' + capability)
            elif components[i].find('--std=c++11') != -1:
                converted_components.append('-std=c++11')
            elif components[i].startswith('-isystem='):
                converted_components.extend(components[i].split('='))
            else:
                converted_components.append(components[i])

        command = ''
        for c in converted_components:
            command = command + ' ' + c
        command = command.strip()
        return command

    def _configure_flags(self, path, command):
        common_args = ['clang-tidy',
                       "-header-filter='(xgboost\\/src|xgboost\\/include)'",
                       '-config='+self.clang_tidy]
        common_args.append(path)
        common_args.append('--')
        command = self.convert_nvcc_command_to_clang(command)

        command = command.split()[1:]  # remove clang/c++/g++
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
            isxgb = isxgb and (not path.startswith(self.cdb_path))
            if isxgb:
                return True

        cdb_file = os.path.join(self.cdb_path, 'compile_commands.json')
        with open(cdb_file, 'r') as fd:
            self.compile_commands = json.load(fd)
        tidy_file = os.path.join(self.root_path, '.clang-tidy')
        with open(tidy_file) as fd:
            self.clang_tidy = yaml.safe_load(fd)
            self.clang_tidy = str(self.clang_tidy)
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


def test_tidy():
    '''See if clang-tidy and our regex is working correctly.  There are
many subtleties we need to be careful.  For instances:

    * Is the string re-directed to pipe encoded as UTF-8? or is it
bytes?

    * On Jenkins there's no 'xgboost' directory, are we catching the
right keywords?

    * Should we use re.DOTALL?

    * Should we use re.MULTILINE?

    Tests here are not thorough, at least we want to guarantee tidy is
    not missing anything on Jenkins.

    '''
    root_path = os.path.abspath(os.path.curdir)
    tidy_file = os.path.join(root_path, '.clang-tidy')
    test_file_path = os.path.join(root_path,
                                  'tests', 'ci_build', 'test_tidy.cc')

    with open(tidy_file) as fd:
        tidy_config = fd.read()
        tidy_config = str(tidy_config)
    tidy_config = '-config='+tidy_config
    args = ['clang-tidy', tidy_config, test_file_path]
    (proc_code, tidy_status, error_msg) = call(args)
    assert proc_code == 0
    assert tidy_status == 1
    print('clang-tidy is working.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run clang-tidy.')
    parser.add_argument('--cpp', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--use-dmlc-gtest', type=int, default=1,
                        help='Whether to use gtest bundled in dmlc-core.')
    args = parser.parse_args()

    test_tidy()

    with ClangTidy(args) as linter:
        passed = linter.run()
    if not passed:
        sys.exit(1)
