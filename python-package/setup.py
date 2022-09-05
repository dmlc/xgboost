"""Setup xgboost package."""
import os
import shutil
import subprocess
import logging
from typing import Optional, List
import sys
from platform import system
from setuptools import setup, find_packages, Extension
from setuptools.command import build_ext, sdist, install_lib, install

# You can't use `pip install .` as pip copies setup.py to a temporary
# directory, parent directory is no longer reachable (isolated build) .
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, CURRENT_DIR)

# Options only effect `python setup.py install`, building `bdist_wheel`
# requires using CMake directly.
USER_OPTIONS = {
    # libxgboost options.
    'use-openmp': (None, 'Build with OpenMP support.', 1),
    'use-cuda':   (None, 'Build with GPU acceleration.', 0),
    'use-nccl':   (None, 'Build with NCCL to enable distributed GPU support.', 0),
    'build-with-shared-nccl': (None, 'Build with shared NCCL library.', 0),
    'hide-cxx-symbols':       (None, 'Hide all C++ symbols during build.', 1),
    'use-hdfs':   (None, 'Build with HDFS support', 0),
    'use-azure':  (None, 'Build with AZURE support.', 0),
    'use-s3':     (None, 'Build with S3 support', 0),
    'plugin-dense-parser': (None, 'Build dense parser plugin.', 0),
    # Python specific
    'use-system-libxgboost': (None, 'Use libxgboost.so in system path.', 0)
}

NEED_CLEAN_TREE = set()
NEED_CLEAN_FILE = set()
BUILD_TEMP_DIR = None


def lib_name() -> str:
    '''Return platform dependent shared object name.'''
    if system() == 'Linux' or system().upper().endswith('BSD'):
        name = 'libxgboost.so'
    elif system() == 'Darwin':
        name = 'libxgboost.dylib'
    elif system() == 'Windows':
        name = 'xgboost.dll'
    elif system() == 'OS400':
        name = 'libxgboost.so'
    return name


def copy_tree(src_dir: str, target_dir: str) -> None:
    '''Copy source tree into build directory.'''
    def clean_copy_tree(src: str, dst: str) -> None:
        shutil.copytree(src, dst)
        NEED_CLEAN_TREE.add(os.path.abspath(dst))

    def clean_copy_file(src: str, dst: str) -> None:
        shutil.copy(src, dst)
        NEED_CLEAN_FILE.add(os.path.abspath(dst))

    src = os.path.join(src_dir, 'src')
    inc = os.path.join(src_dir, 'include')
    dmlc_core = os.path.join(src_dir, 'dmlc-core')
    gputreeshap = os.path.join(src_dir, "gputreeshap")
    rabit = os.path.join(src_dir, 'rabit')
    cmake = os.path.join(src_dir, 'cmake')
    plugin = os.path.join(src_dir, 'plugin')

    clean_copy_tree(src, os.path.join(target_dir, 'src'))
    clean_copy_tree(inc, os.path.join(target_dir, 'include'))
    clean_copy_tree(dmlc_core, os.path.join(target_dir, 'dmlc-core'))
    clean_copy_tree(gputreeshap, os.path.join(target_dir, "gputreeshap"))
    clean_copy_tree(rabit, os.path.join(target_dir, 'rabit'))
    clean_copy_tree(cmake, os.path.join(target_dir, 'cmake'))
    clean_copy_tree(plugin, os.path.join(target_dir, 'plugin'))

    cmake_list = os.path.join(src_dir, 'CMakeLists.txt')
    clean_copy_file(cmake_list, os.path.join(target_dir, 'CMakeLists.txt'))
    lic = os.path.join(src_dir, 'LICENSE')
    clean_copy_file(lic, os.path.join(target_dir, 'LICENSE'))


def clean_up() -> None:
    '''Removed copied files.'''
    for path in NEED_CLEAN_TREE:
        shutil.rmtree(path)
    for path in NEED_CLEAN_FILE:
        os.remove(path)


class CMakeExtension(Extension):  # pylint: disable=too-few-public-methods
    '''Wrapper for extension'''
    def __init__(self, name: str) -> None:
        super().__init__(name=name, sources=[])


class BuildExt(build_ext.build_ext):  # pylint: disable=too-many-ancestors
    '''Custom build_ext command using CMake.'''

    logger = logging.getLogger('XGBoost build_ext')

    # pylint: disable=too-many-arguments
    def build(
        self,
        src_dir: str,
        build_dir: str,
        generator: str,
        build_tool: Optional[str] = None,
        use_omp: int = 1,
    ) -> None:
        '''Build the core library with CMake.'''
        cmake_cmd = ['cmake', src_dir, generator]

        for k, v in USER_OPTIONS.items():
            arg = k.replace('-', '_').upper()
            value = str(v[2])
            if arg == 'USE_SYSTEM_LIBXGBOOST':
                continue
            if arg == 'USE_OPENMP' and use_omp == 0:
                cmake_cmd.append("-D" + arg + "=0")
                continue
            cmake_cmd.append('-D' + arg + '=' + value)

        # Flag for cross-compiling for Apple Silicon
        # We use environment variable because it's the only way to pass down custom flags
        # through the cibuildwheel package, which otherwise calls `python setup.py bdist_wheel`
        # command.
        if 'CIBW_TARGET_OSX_ARM64' in os.environ:
            cmake_cmd.append("-DCMAKE_OSX_ARCHITECTURES=arm64")

        self.logger.info('Run CMake command: %s', str(cmake_cmd))
        subprocess.check_call(cmake_cmd, cwd=build_dir)

        if system() != 'Windows':
            nproc = os.cpu_count()
            assert build_tool is not None
            subprocess.check_call([build_tool, '-j' + str(nproc)],
                                  cwd=build_dir)
        else:
            subprocess.check_call(['cmake', '--build', '.',
                                   '--config', 'Release'], cwd=build_dir)

    def build_cmake_extension(self) -> None:
        '''Configure and build using CMake'''
        if USER_OPTIONS['use-system-libxgboost'][2]:
            self.logger.info('Using system libxgboost.')
            return

        build_dir = self.build_temp
        global BUILD_TEMP_DIR  # pylint: disable=global-statement
        BUILD_TEMP_DIR = build_dir
        libxgboost = os.path.abspath(
            os.path.join(CURRENT_DIR, os.path.pardir, 'lib', lib_name()))

        if os.path.exists(libxgboost):
            self.logger.info('Found shared library, skipping build.')
            return

        src_dir = 'xgboost'
        try:
            copy_tree(os.path.join(CURRENT_DIR, os.path.pardir),
                      os.path.join(self.build_temp, src_dir))
        except Exception:  # pylint: disable=broad-except
            copy_tree(src_dir, os.path.join(self.build_temp, src_dir))

        self.logger.info('Building from source. %s', libxgboost)
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        if shutil.which('ninja'):
            build_tool = 'ninja'
        else:
            build_tool = 'make'
        if sys.platform.startswith('os400'):
            build_tool = 'make'

        if system() == 'Windows':
            # Pick up from LGB, just test every possible tool chain.
            for vs in (
                "-GVisual Studio 17 2022",
                '-GVisual Studio 16 2019',
                '-GVisual Studio 15 2017',
                '-GVisual Studio 14 2015',
                '-GMinGW Makefiles',
            ):
                try:
                    self.build(src_dir, build_dir, vs)
                    self.logger.info(
                        '%s is used for building Windows distribution.', vs)
                    break
                except subprocess.CalledProcessError:
                    shutil.rmtree(build_dir)
                    os.mkdir(build_dir)
                    continue
        else:
            gen = '-GNinja' if build_tool == 'ninja' else '-GUnix Makefiles'
            try:
                self.build(src_dir, build_dir, gen, build_tool, use_omp=1)
            except subprocess.CalledProcessError:
                self.logger.warning('Disabling OpenMP support.')
                self.build(src_dir, build_dir, gen, build_tool, use_omp=0)

    def build_extension(self, ext: Extension) -> None:
        '''Override the method for dispatching.'''
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension()
        else:
            super().build_extension(ext)

    def copy_extensions_to_source(self) -> None:
        '''Dummy override.  Invoked during editable installation.  Our binary
        should available in `lib`.

        '''
        if not os.path.exists(
                os.path.join(CURRENT_DIR, os.path.pardir, 'lib', lib_name())):
            raise ValueError('For using editable installation, please ' +
                             'build the shared object first with CMake.')


class Sdist(sdist.sdist):       # pylint: disable=too-many-ancestors
    '''Copy c++ source into Python directory.'''
    logger = logging.getLogger('xgboost sdist')

    def run(self) -> None:
        copy_tree(os.path.join(CURRENT_DIR, os.path.pardir),
                  os.path.join(CURRENT_DIR, 'xgboost'))
        libxgboost = os.path.join(
            CURRENT_DIR, os.path.pardir, 'lib', lib_name())
        if os.path.exists(libxgboost):
            self.logger.warning(
                'Found shared library, removing to avoid being included in source distribution.'
            )
            os.remove(libxgboost)
        super().run()


class InstallLib(install_lib.install_lib):
    '''Copy shared object into installation directory.'''
    logger = logging.getLogger('xgboost install_lib')

    def install(self) -> List[str]:
        outfiles = super().install()

        if USER_OPTIONS['use-system-libxgboost'][2] != 0:
            self.logger.info('Using system libxgboost.')
            lib_path = os.path.join(sys.prefix, 'lib')
            msg = 'use-system-libxgboost is specified, but ' + lib_name() + \
                ' is not found in: ' + lib_path
            assert os.path.exists(os.path.join(lib_path, lib_name())), msg
            return []

        lib_dir = os.path.join(self.install_dir, 'xgboost', 'lib')
        if not os.path.exists(lib_dir):
            os.mkdir(lib_dir)
        dst = os.path.join(self.install_dir, 'xgboost', 'lib', lib_name())

        libxgboost_path = lib_name()

        assert BUILD_TEMP_DIR is not None
        dft_lib_dir = os.path.join(CURRENT_DIR, os.path.pardir, 'lib')
        build_dir = os.path.join(BUILD_TEMP_DIR, 'xgboost', 'lib')

        if os.path.exists(os.path.join(dft_lib_dir, libxgboost_path)):
            # The library is built by CMake directly
            src = os.path.join(dft_lib_dir, libxgboost_path)
        else:
            # The library is built by setup.py
            src = os.path.join(build_dir, libxgboost_path)
        self.logger.info('Installing shared library: %s', src)
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)
        return outfiles


class Install(install.install):  # pylint: disable=too-many-instance-attributes
    '''An interface to install command, accepting XGBoost specific
    arguments.

    '''
    user_options = install.install.user_options + list(
        (k, v[0], v[1]) for k, v in USER_OPTIONS.items())

    def initialize_options(self) -> None:
        super().initialize_options()
        self.use_openmp = 1
        self.use_cuda = 0
        self.use_nccl = 0
        self.build_with_shared_nccl = 0
        self.hide_cxx_symbols = 1

        self.use_hdfs = 0
        self.use_azure = 0
        self.use_s3 = 0

        self.plugin_dense_parser = 0

        self.use_system_libxgboost = 0

    def run(self) -> None:
        # setuptools will configure the options according to user supplied command line
        # arguments, then here we propagate them into `USER_OPTIONS` for visibility to
        # other sub-commands like `build_ext`.
        for k, v in USER_OPTIONS.items():
            arg = k.replace('-', '_')
            if hasattr(self, arg):
                USER_OPTIONS[k] = (v[0], v[1], getattr(self, arg))
        super().run()


if __name__ == '__main__':
    # Supported commands:
    # From internet:
    # - pip install xgboost
    # - pip install --no-binary :all: xgboost

    # From source tree `xgboost/python-package`:
    # - python setup.py build
    # - python setup.py build_ext
    # - python setup.py install
    # - python setup.py sdist       && pip install <sdist-name>
    # - python setup.py bdist_wheel && pip install <wheel-name>

    # When XGBoost is compiled directly with CMake:
    # - pip install -e .
    # - python setup.py develop   # same as above
    logging.basicConfig(level=logging.INFO)

    with open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8') as fd:
        description = fd.read()
    with open(os.path.join(CURRENT_DIR, 'xgboost/VERSION'), encoding="ascii") as fd:
        version = fd.read().strip()

    setup(name='xgboost',
          version=version,
          description="XGBoost Python Package",
          long_description=description,
          long_description_content_type="text/x-rst",
          install_requires=[
              'numpy',
              'scipy',
          ],
          ext_modules=[CMakeExtension('libxgboost')],
          # error: expected "str": "Type[Command]"
          cmdclass={
              'build_ext': BuildExt,     # type: ignore
              'sdist': Sdist,            # type: ignore
              'install_lib': InstallLib,  # type: ignore
              'install': Install          # type: ignore
          },
          extras_require={
              'pandas': ['pandas'],
              'scikit-learn': ['scikit-learn'],
              'dask': ['dask', 'pandas', 'distributed'],
              'datatable': ['datatable'],
              'plotting': ['graphviz', 'matplotlib'],
              "pyspark": ["pyspark", "scikit-learn", "cloudpickle"],
          },
          maintainer='Hyunsu Cho',
          maintainer_email='chohyu01@cs.washington.edu',
          zip_safe=False,
          packages=find_packages(),
          include_package_data=True,
          license='Apache-2.0',
          classifiers=['License :: OSI Approved :: Apache Software License',
                       'Development Status :: 5 - Production/Stable',
                       'Operating System :: OS Independent',
                       'Programming Language :: Python',
                       'Programming Language :: Python :: 3',
                       'Programming Language :: Python :: 3.8',
                       'Programming Language :: Python :: 3.9',
                       'Programming Language :: Python :: 3.10'],
          python_requires=">=3.8",
          url='https://github.com/dmlc/xgboost')

    clean_up()
