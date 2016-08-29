Installation Guide
==================

This page gives instructions on how to build and install the xgboost package from
scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (`libxgboost.so` for linux/osx and `libxgboost.dll` for windows).
   - Exception: for R-package installation please directly refer to the R package section.
2. Then install the language packages (e.g. Python Package).

***Important*** the newest version of xgboost uses submodule to maintain packages. So when you clone the repo, remember to use the recursive option as follows.
```bash
git clone --recursive https://github.com/dmlc/xgboost
```
For windows users who use github tools, you can open the git shell, and type the following command.
```bash
git submodule init
git submodule update
```

Please refer to [Trouble Shooting Section](#trouble-shooting) first if you had any problem
during installation. If the instructions do not work for you, please feel free
to ask questions at [xgboost/issues](https://github.com/dmlc/xgboost/issues), or
even better to send pull request if you can fix the problem.

## Contents
- [Build the Shared Library](#build-the-shared-library)
  - [Building on Ubuntu/Debian](#building-on-ubuntu-debian)
  - [Building on OSX](#building-on-osx)
  - [Building on Windows](#building-on-windows)
  - [Customized Building](#customized-building)
- [Python Package Installation](#python-package-installation)
- [R Package Installation](#r-package-installation)
- [Trouble Shooting](#trouble-shooting)

## Build the Shared Library

Our goal is to build the shared library:
- On Linux/OSX the target library is `libxgboost.so`
- On Windows the target library is `libxgboost.dll`

The minimal building requirement is

- A recent c++ compiler supporting C++ 11 (g++-4.6 or higher)

We can edit `make/config.mk` to change the compile options, and then build by
`make`. If everything goes well, we can go to the specific language installation section.

### Building on Ubuntu/Debian

On Ubuntu, one builds xgboost by

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
```

### Building on OSX

On OSX, one builds xgboost by

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/minimum.mk ./config.mk; make -j4
```

This builds xgboost without multi-threading, because by default clang in OSX does not come with open-mp.
See the following paragraph for OpenMP enabled xgboost.


Here is the complete solution to use OpenMP-enabled compilers to install XGBoost.
Obtain gcc-6.x.x with openmp support by `brew install gcc --without-multilib`. (`brew` is the de facto standard of `apt-get` on OS X. So installing [HPC](http://hpc.sourceforge.net/) separately is not recommended, but it should work.). Installation of `gcc` can take a while (~ 30 minutes)

Now, clone the repository

```bash
git clone --recursive https://github.com/dmlc/xgboost
```

and build using the following commands

```bash
cd ..; cp make/config.mk ./config.mk; make -j4
```

NOTE:
If you use OSX El Capitan, brew installs gcc the latest version gcc-6. So you may need to modify Makefile#L46 and change gcc-5 to gcc-6. After that change gcc-5/g++-5 to gcc-6/g++-6 in make/config.mk then build using the following commands

```bash
cd ..; cp make/config.mk ./config.mk; make -j4
```

### Building on Windows
You need to first clone the xgboost repo with recursive option clone the submodules.
If you are using github tools, you can open the git-shell, and type the following command.
We recommend using [Git for Windows](https://git-for-windows.github.io/)
because it brings a standard bash shell. This will highly ease the installation process.

```bash
git submodule init
git submodule update
```

XGBoost support both build by MSVC or MinGW. Here is how you can build xgboost library using MinGW.

After installing [Git for Windows](https://git-for-windows.github.io/), you should have a shortcut `Git Bash`.
All the following steps are in the `Git Bash`.

In MinGW, `make` command comes with the name `mingw32-make`. You can add the following line into the `.bashrc` file.

```bash
alias make='mingw32-make'
```

To build with MinGW

```bash
cp make/mingw64.mk config.mk; make -j4
```

To build with Visual Studio 2013 use cmake. Make sure you have a recent version of cmake added to your path and then from the xgboost directory:

```bash
mkdir build
cd build
cmake .. -G"Visual Studio 12 2013 Win64"
```

This specifies an out of source build using the MSVC 12 64 bit generator. Open the .sln file in the build directory and build with Visual Studio. To use the Python module you can copy libxgboost.dll into python-package\xgboost.

Other versions of Visual Studio may work but are untested.

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
   cd python-package; sudo python setup.py install
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
   `~`. then we can added the following line in `~/.bashrc`.
    It is ***recommended for developers*** who may change the codes. The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call ```setup``` again)

    ```bash
    export PYTHONPATH=~/xgboost/python-package
    ```

3. Install only for the current user.

    ```bash
    cd python-package; python setup.py develop --user
    ```

4. If you are installing the latest xgboost version which requires compilation, add MinGW to the system PATH:

    ```python
    import os
    os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
    ```

## R Package Installation

You can install R package from cran just like other packages, or you can install from our weekly updated drat repo:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
```

If you would like to use the latest xgboost version and already compiled xgboost, use `library(devtools); install('xgboost/R-package')` to install manually xgboost package (change the path accordingly to where you compiled xgboost).

For OSX users, single threaded version will be installed, to install multi-threaded version.
First follow [Building on OSX](#building-on-osx) to get the OpenMP enabled compiler, then:

- Set the `Makevars` file in highest piority for R.

  The point is, there are three `Makevars` : `~/.R/Makevars`, `xgboost/R-package/src/Makevars`, and `/usr/local/Cellar/r/3.2.0/R.framework/Resources/etc/Makeconf` (the last one obtained by running `file.path(R.home("etc"), "Makeconf")` in R), and `SHLIB_OPENMP_CXXFLAGS` is not set by default!! After trying, it seems that the first one has highest piority (surprise!).

  Then inside R, run

  ```R
  install.packages("drat", repos="https://cran.rstudio.com")
  drat:::addRepo("dmlc")
  install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
  ```

Due to the usage of submodule, `install_github` is no longer support to install the
latest version of R package. To install the latest version run the following bash script,

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git submodule init
git submodule update
alias make='mingw32-make'
cd dmlc-core
make -j4
cd ../rabit
make lib/librabit_empty.a -j4
cd ..
cp make/mingw64.mk config.mk
make -j4
```

## Trouble Shooting

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


3. **Makefile: dmlc-core/make/dmlc.mk: No such file or directory**

   We need to recursively clone the submodule, you can do:

    ```bash
    git submodule init
    git submodule update
    ```
    Alternatively, do another clone
    ```bash
    git clone https://github.com/dmlc/xgboost --recursive
    ```
