#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from time import time

import yaml


def call(args):
    '''Subprocess run wrapper.'''
    completed = subprocess.run(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    error_msg = completed.stdout.decode('utf-8')
    # `workspace` is a name used in Jenkins CI.  Normally we should keep the
    # dir as `xgboost`.
    matched = re.search('(workspace|xgboost)/.*(src|tests|include)/.*warning:',
                        error_msg,
                        re.MULTILINE)
    if matched is None:
        return_code = 0
    else:
        return_code = 1
    return (completed.returncode, return_code, error_msg, args)


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
        self.use_dmlc_gtest: bool = args.use_dmlc_gtest
        self.cuda_archs = args.cuda_archs.copy() if args.cuda_archs else []

        if args.tidy_version:
            self.exe = 'clang-tidy-' + str(args.tidy_version)
        else:
            self.exe = 'clang-tidy'

        print('Run linter on CUDA: ', self.cuda_lint)
        print('Run linter on C++:', self.cpp_lint)
        print('Use dmlc gtest:', self.use_dmlc_gtest)
        print('CUDA archs:', ' '.join(self.cuda_archs))

        if not self.cpp_lint and not self.cuda_lint:
            raise ValueError('Both --cpp and --cuda are set to 0.')
        self.root_path = os.path.abspath(os.path.curdir)
        print('Project root:', self.root_path)
        self.cdb_path = os.path.join(self.root_path, 'cdb')

    def __enter__(self):
        self.start = time()
        if os.path.exists(self.cdb_path):
            shutil.rmtree(self.cdb_path)
        self._generate_cdb()
        return self

    def __exit__(self, *args):
        if os.path.exists(self.cdb_path):
            shutil.rmtree(self.cdb_path)
        self.end = time()
        print('Finish running clang-tidy:', self.end - self.start)

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
            if self.cuda_archs:
                arch_list = ';'.join(self.cuda_archs)
                cmake_args.append(f'-DGPU_COMPUTE_VER={arch_list}')
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

        for i in range(1, len(components)):
            if components[i] == '-lineinfo':
                continue
            elif components[i] == '-fuse-ld=gold':
                continue
            elif components[i] == '-rdynamic':
                continue
            elif components[i] == "-Xfatbin=-compress-all":
                continue
            elif components[i] == "-forward-unknown-to-host-compiler":
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
            elif components[i].find('--std=c++14') != -1:
                converted_components.append('-std=c++14')
            elif components[i].startswith('-isystem='):
                converted_components.extend(components[i].split('='))
            else:
                converted_components.append(components[i])

        converted_components.append('-isystem /usr/local/cuda/include/')

        command = ''
        for c in converted_components:
            command = command + ' ' + c
        command = command.strip()
        return command

    def _configure_flags(self, path, command):
        src = os.path.join(self.root_path, 'src')
        src = src.replace('/', '\\/')
        include = os.path.join(self.root_path, 'include')
        include = include.replace('/', '\\/')

        header_filter = '(' + src + '|' + include + ')'
        common_args = [self.exe,
                       "-header-filter=" + header_filter,
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
                print(path)
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
            for i, (process_status, tidy_status, msg, args) in enumerate(results):
                # Don't enforce clang-tidy to pass for now due to namespace
                # for cub in thrust is not correct.
                if tidy_status == 1:
                    passed = False
                    print(BAR, '\n'
                          'Command args:', ' '.join(args), ', ',
                          'Process return code:', process_status, ', ',
                          'Tidy result code:', tidy_status, ', ',
                          'Message:\n', msg,
                          BAR, '\n')
        if not passed:
            print('Errors in `thrust` namespace can be safely ignored.',
                  'Please address rest of the clang-tidy warnings.')
        return passed


def test_tidy(args):
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
    if not args.tidy_version:
        tidy = 'clang-tidy'
    else:
        tidy = 'clang-tidy-' + str(args.tidy_version)
    args = [tidy, tidy_config, test_file_path]
    (proc_code, tidy_status, error_msg, _) = call(args)
    assert proc_code == 0
    assert tidy_status == 1
    print('clang-tidy is working.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clang-tidy.")
    parser.add_argument("--cpp", type=int, default=1)
    parser.add_argument(
        "--tidy-version",
        type=int,
        default=None,
        help="Specify the version of preferred clang-tidy.",
    )
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument(
        "--use-dmlc-gtest",
        action="store_true",
        help="Whether to use gtest bundled in dmlc-core.",
    )
    parser.add_argument(
        "--cuda-archs", action="append", help="List of CUDA archs to build"
    )
    args = parser.parse_args()

    test_tidy(args)

    with ClangTidy(args) as linter:
        passed = linter.run()
    if not passed:
        sys.exit(1)
