Installation Guide
==================

This page gives instructions of how to build and install the xgboost package from
scratch on various systems. It consists of two steps:

1. Fist build the shared library from the C++ codes (`libxgboost.so` for linux/osx and `libxgboost.dll` for windows).
   - Exception: for R-package installation please directly refer to the R package section.
2. Then install the language packages (e.g. Python Package).

Please refer to [Installation FAQ](#frequently-asked-questions) first if you had any problem
during installation. If the instructions do not work for you, please feel free
to ask questions at [xgboost/issues](https://github.com/dmlc/xgboost/issues), or
even better to send pull request if you can fix the problem.

## Contents
- [Build the Shared Library](#build-the-shared-library)
  - [Prerequisites](#prerequisites)
  - [Building on Ubuntu/Debian](#building-on-ubuntu-debian)
  - [Building on OSX](#building-on-osx)
  - [Building on Windows](#building-on-windows)
  - [Customized Building](#customized-building)
- [Python Package Installation](#python-package-installation)
- [R Package Installation](#r-package-installation)
- [Frequently asked questions](#frequently-asked-questions)

## Build the Shared Library

Our goal is to build the shared library:
- On Linux/OSX the target library is ```libxgboost.so```
- On Windows the target libary is ```libxgboost.dll```

The minimal building requirement is

- A recent c++ compiler supporting C++ 11 (g++-4.6 or higher)

We can edit `make/config.mk` to change the compile options, and then build by
`make`. If everything goes well, we can go the specific language installation section.

### Building on Ubuntu/Debian

On Ubuntu, one build xgboost by

Then build xgboost
```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
```

### Building on OSX

On Ubuntu OSX, one build xgboost by

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/minimum.mk ./config.mk; make -j4
```

This build xgboost without multi-threading, because by default clang in OSX does not come with open-mp.
See the following paragraph for OpenMP enabled xgboost.


Here is the complete solution to use OpenMP-enabled compilers to install XGBoost.
Obtain gcc-5.x.x with openmp support by `brew install gcc --without-multilib`. (`brew` is the de facto standard of `apt-get` on OS X. So installing [HPC](http://hpc.sourceforge.net/) separately is not recommended, but it should work.)

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/config.mk ./config.mk; make -j4
```

### Building on Windows

XGBoost support both build by MSVC or MinGW. Here is how you can build xgboost library using MinGW.

Build with mingw
```bash
cp make/mingw64.mk config.mk; make -j4
```

The MSVC build for new version is not yet updated.


### Customized Building

The configuration of xgboost can be modified by ```config.mk```
- modify configuration on various distributed filesystem such as HDFS/Amazon S3/...
- First copy [make/config.mk](../make/config.mk) to the project root, on which
  any local modification will be ignored by git, then modify the according flags.



## Python Package Installation

The python package is located at [python-package](../python-package).
There are several ways to install the package:

1. Install system-widely, which requires root permission

   ```bash
   cd python; sudo python setup.py install
   ```

   You will however need Python `distutils` module for this to
   work. It is often part of the core python package or it can be installed using your
   package manager, e.g. in Debian use

   ```bash
   sudo apt-get install python-setuptools
   ```

   *NOTE: If you recompiled xgboost, then you need to reinstall it again to
    make the new library take effect*

2. Only set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `xgboost` on the home directory
   `~`. then we can added the following line in `~/.bashrc`
   It is ***recommended for developers*** who may change the codes. The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call ```setup``` again)

    ```bash
    export PYTHONPATH=~/xgboost/python-package
    ```

3. Install only for the current user.

    ```bash
    cd python; python setup.py develop --user
    ```

## R Package Installation

You can install R package using devtools

```r
devtools::install_git('git://github.com/dmlc/xgboost',subdir='R-package')

```

For OSX users, single threaded version will be installed, to install multi-threaded version.
First follow [Building on OSX](#building-on-osx) to get the OpenMP enabled compiler, then:

- Set the `Makevars` file in highest piority for R.

  The point is, there are three `Makevars` : `~/.R/Makevars`, `xgboost/R-package/src/Makevars`, and `/usr/local/Cellar/r/3.2.0/R.framework/Resources/etc/Makeconf` (the last one obtained by running `file.path(R.home("etc"), "Makeconf")` in R), and `SHLIB_OPENMP_CXXFLAGS` is not set by default!! After trying, it seems that the first one has highest piority (surprise!).

  Then inside R, run

  ```R
  install.packages('xgboost/R-package/', repos=NULL, type='source')
  ```

  Or

  ```R
  devtools::install_local('xgboost/', subdir = 'R-package') # you may use devtools
  ```

## Frequently Asked Questions

1. **Compile failed after `git pull`**

   Please first update the submodules, clean all and recompile:

   ```bash
   git submodule update && make clean_all && make -j4
   ```

2. **Compile failed after `config.mk` is modified**
   Need to clean all first:

    ```bash
    make clean_all && make -j4
    ```
