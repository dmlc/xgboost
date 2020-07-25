import argparse
import os
import subprocess

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir,
                 os.path.pardir))
r_package = os.path.join(ROOT, 'R-package')


class DirectoryExcursion:
    def __init__(self, path: os.PathLike):
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.curdir)


def get_mingw_bin():
    return os.path.join('c:/rtools40/mingw64/', 'bin')


def test_with_autotools(args):
    with DirectoryExcursion(r_package):
        if args.compiler == 'mingw':
            mingw_bin = get_mingw_bin()
            CXX = os.path.join(mingw_bin, 'g++.exe')
            CC = os.path.join(mingw_bin, 'gcc.exe')
            cmd = ['R.exe', 'CMD', 'INSTALL', str(os.path.curdir)]
            env = os.environ.copy()
            env.update({'CC': CC, 'CXX': CXX})
            subprocess.check_call(cmd, env=env)
        elif args.compiler == 'msvc':
            cmd = ['R.exe', 'CMD', 'INSTALL', str(os.path.curdir)]
            env = os.environ.copy()
            # autotool favor mingw by default.
            env.update({'CC': 'cl.exe', 'CXX': 'cl.exe'})
            subprocess.check_call(cmd, env=env)
        else:
            raise ValueError('Wrong compiler')
        subprocess.check_call([
            'R.exe', '-q', '-e',
            "library(testthat); setwd('tests'); source('testthat.R')"
        ])


def test_with_cmake(args):
    os.mkdir('build')
    with DirectoryExcursion('build'):
        if args.compiler == 'mingw':
            mingw_bin = get_mingw_bin()
            CXX = os.path.join(mingw_bin, 'g++.exe')
            CC = os.path.join(mingw_bin, 'gcc.exe')
            env = os.environ.copy()
            env.update({'CC': CC, 'CXX': CXX})
            subprocess.check_call([
                'cmake', os.path.pardir, '-DUSE_OPENMP=ON', '-DR_LIB=ON',
                '-DCMAKE_CONFIGURATION_TYPES=Release', '-G', 'Unix Makefiles',
            ],
                                  env=env)
            subprocess.check_call(['make', '-j', 'install'])
        elif args.compiler == 'msvc':
            subprocess.check_call([
                'cmake', os.path.pardir, '-DUSE_OPENMP=ON', '-DR_LIB=ON',
                '-DCMAKE_CONFIGURATION_TYPES=Release', '-A', 'x64'
            ])
            subprocess.check_call([
                'cmake', '--build', os.path.curdir, '--target', 'install',
                '--config', 'Release'
            ])
        else:
            raise ValueError('Wrong compiler')
    with DirectoryExcursion(r_package):
        subprocess.check_call([
            'R.exe', '-q', '-e',
            "library(testthat); setwd('tests'); source('testthat.R')"
        ])


def main(args):
    if args.build_tool == 'autotools':
        test_with_autotools(args)
    else:
        test_with_cmake(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler',
                        type=str,
                        choices=['mingw', 'msvc'],
                        help='Compiler used for compiling CXX code.')
    parser.add_argument(
        '--build-tool',
        type=str,
        choices=['cmake', 'autotools'],
        help='Build tool for compiling CXX code and install R package.')
    args = parser.parse_args()
    main(args)
