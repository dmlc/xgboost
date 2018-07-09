Installation Guide
==================

**NOTE**. If you are planning to use Python on a Linux system, consider installing XGBoost from a pre-built binary wheel. The wheel is available from Python Package Index (PyPI). You may download and install it by running
```bash
# Ensure that you are downloading xgboost-{version}-py2.py3-none-manylinux1_x86_64.whl
pip3 install xgboost
```
* This package will support GPU algorithms (`gpu_exact`, `gpu_hist`) on machines with NVIDIA GPUs.
* Currently, PyPI has a binary wheel only for 64-bit Linux.

# Building XGBoost from source
This page gives instructions on how to build and install the xgboost package from
scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (`libxgboost.so` for Linux/OSX and `xgboost.dll` for Windows).
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
  - [Building on macOS](#building-on-macos)
  - [Building on Windows](#building-on-windows)
  - [Building with GPU support](#building-with-gpu-support)
  - [Windows Binaries](#windows-binaries)
  - [Customized Building](#customized-building)
- [Python Package Installation](#python-package-installation)
- [R Package Installation](#r-package-installation)
- [Trouble Shooting](#trouble-shooting)

## Build the Shared Library

Our goal is to build the shared library:
- On Linux/OSX the target library is `libxgboost.so`
- On Windows the target library is `xgboost.dll`

The minimal building requirement is

- A recent c++ compiler supporting C++ 11 (g++-4.8 or higher)

We can edit `make/config.mk` to change the compile options, and then build by
`make`. If everything goes well, we can go to the specific language installation section.

### Building on Ubuntu/Debian

On Ubuntu, one builds xgboost by

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
```

### Building on macOS

**Install with pip - simple method**

First, make sure you obtained *gcc-5* (newer version does not work with this method yet). Note: installation of `gcc` can take a while (~ 30 minutes)

```bash
brew install gcc5
```

You might need to run the following command with `sudo` if you run into some permission errors:

```bash
pip install xgboost
```

**Build from the source code - advanced method**

First, obtain gcc-7.x.x with brew (https://brew.sh/) if you want multi-threaded version, otherwise, Clang is ok if OpenMP / multi-threaded is not required. Note: installation of `gcc` can take a while (~ 30 minutes)

```bash
brew install gcc
```

Now, clone the repository

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/config.mk ./config.mk
```

Open config.mk and uncomment these two lines

```config.mk
export CC = gcc
export CXX = g++
```

and replace these two lines into(5 or 6 or 7; depending on your gcc-version)

```config.mk
export CC = gcc-7
export CXX = g++-7
```

To find your gcc version

```bash
gcc-version
```

and build using the following commands

```bash
make -j4
```
head over to `Python Package Installation` for the next steps

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
(On 64-bit Windows, you should get [mingw64](https://sourceforge.net/projects/mingw-w64/) instead.) Make sure
that the path to MinGW is in the system PATH.

To build with MinGW, type:

```bash
cp make/mingw64.mk config.mk; make -j4
```

To build with Visual Studio 2013 use cmake. Make sure you have a recent version of cmake added to your path and then from the xgboost directory:

```bash
mkdir build
cd build
cmake .. -G"Visual Studio 12 2013 Win64"
```

This specifies an out of source build using the MSVC 12 64 bit generator. Open the .sln file in the build directory and build with Visual Studio. To use the Python module you can copy `xgboost.dll` into python-package\xgboost.

Other versions of Visual Studio may work but are untested.

### Building with GPU support

XGBoost can be built with GPU support for both Linux and Windows using cmake. GPU support works with the Python package as well as the CLI version. See [Installing R package with GPU support](#installing-r-package-with-gpu-support) for special instructions for R.

An up-to-date version of the CUDA toolkit is required.

From the command line on Linux starting from the xgboost directory:

```bash
$ mkdir build
$ cd build
$ cmake .. -DUSE_CUDA=ON
$ make -j
```
**Windows requirements** for GPU build: only Visual C++ 2015 or 2013 with CUDA v8.0 were fully tested. Either install Visual C++ 2015 Build Tools separately, or as a part of Visual Studio 2015. If you already have Visual Studio 2017, the Visual C++ 2015 Toolchain componenet has to be installed using the VS 2017 Installer. Likely, you would need to use the VS2015 x64 Native Tools command prompt to run the cmake commands given below. In some situations, however, things run just fine from MSYS2 bash command line.

On Windows, using cmake, see what options for Generators you have for cmake, and choose one with [arch] replaced by Win64:
```bash
cmake -help
```
Then run cmake as:
```bash
$ mkdir build
$ cd build
$ cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON
```
To speed up compilation, compute version specific to your GPU could be passed to cmake as, e.g., `-DGPU_COMPUTE_VER=50`.
The above cmake configuration run will create an xgboost.sln solution file in the build directory. Build this solution in release mode as a x64 build, either from Visual studio or from command line:
```
cmake --build . --target xgboost --config Release
```
If build seems to use only a single process, you might try to append an option like ` -- /m:6` to the above command.

### Windows Binaries

After the build process successfully ends, you will find a `xgboost.dll` library file inside `./lib/` folder, copy this file to the the API package folder like `python-package/xgboost` if you are using *python* API. And you are good to follow the below instructions.

Unofficial windows binaries and instructions on how to use them are hosted on [Guido Tapia's blog](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/)

### Building with Multi-GPU support
Multi-GPU support requires the [NCCL](https://developer.nvidia.com/nccl) library. With NCCL installed, run cmake as:
```bash
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DNCCL_ROOT="<NCCL_DIRECTORY>"
export LD_LIBRARY_PATH=<NCCL_DIRECTORY>/lib:$LD_LIBRARY_PATH
```
One can also pass NCCL_ROOT as an environment variable, in which case, this takes precedence over the cmake variable NCCL_ROOT.

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

### Installing pre-packaged version

You can install xgboost from CRAN just like any other R package:

```r
install.packages("xgboost")
```

Or you can install it from our weekly updated drat repo:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
```

For OSX users, single threaded version will be installed. To install multi-threaded version,
first follow [Building on OSX](#building-on-osx) to get the OpenMP enabled compiler, then:

- Set the `Makevars` file in highest piority for R.

  The point is, there are three `Makevars` : `~/.R/Makevars`, `xgboost/R-package/src/Makevars`, and `/usr/local/Cellar/r/3.2.0/R.framework/Resources/etc/Makeconf` (the last one obtained by running `file.path(R.home("etc"), "Makeconf")` in R), and `SHLIB_OPENMP_CXXFLAGS` is not set by default!! After trying, it seems that the first one has highest piority (surprise!).

  Then inside R, run

  ```R
  install.packages("drat", repos="https://cran.rstudio.com")
  drat:::addRepo("dmlc")
  install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
  ```

### Installing the development version

Make sure you have installed git and a recent C++ compiler supporting C++11 (e.g., g++-4.8 or higher).
On Windows, Rtools must be installed, and its bin directory has to be added to PATH during the installation.
And see the previous subsection for an OSX tip.

Due to the use of git-submodules, `devtools::install_github` can no longer be used to install the latest version of R package.
Thus, one has to run git to check out the code first:

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git submodule init
git submodule update
cd R-package
R CMD INSTALL .
```

If the last line fails because of "R: command not found", it means that R was not set up to run from command line.
In this case, just start R as you would normally do and run the following:

```r
setwd('wherever/you/cloned/it/xgboost/R-package/')
install.packages('.', repos = NULL, type="source")
```

The package could also be built and installed with cmake (and Visual C++ 2015 on Windows) using instructions from the next section, but without GPU support (omit the `-DUSE_CUDA=ON` cmake parameter).

If all fails, try [building the shared library](#build-the-shared-library) to see whether a problem is specific to R package or not.

### Installing R package with GPU support

The procedure and requirements are similar as in [Building with GPU support](#building-with-gpu-support), so make sure to read it first.

On Linux, starting from the xgboost directory:

```bash
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DR_LIB=ON
make install -j
```
When default target is used, an R package shared library would be built in the `build` area.
The `install` target, in addition, assembles the package files with this shared library under `build/R-package`, and runs `R CMD INSTALL`.

On Windows, cmake with Visual C++ Build Tools (or Visual Studio) has to be used to build an R package with GPU support. Rtools must also be installed (perhaps, some other MinGW distributions with `gendef.exe` and `dlltool.exe` would work, but that was not tested).
```bash
mkdir build
cd build
cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON -DR_LIB=ON
cmake --build . --target install --config Release
```
When `--target xgboost` is used, an R package dll would be built under `build/Release`.
The `--target install`, in addition, assembles the package files with this dll under `build/R-package`, and runs `R CMD INSTALL`.

If cmake can't find your R during the configuration step, you might provide the location of its executable to cmake like this: `-DLIBR_EXECUTABLE="C:/Program Files/R/R-3.4.1/bin/x64/R.exe"`.

If on Windows you get a "permission denied" error when trying to write to ...Program Files/R/... during the package installation, create a `.Rprofile` file in your personal home directory (if you don't already have one in there), and add a line to it which specifies the location of your R packages user library, like the following:
```r
.libPaths( unique(c("C:/Users/USERNAME/Documents/R/win-library/3.4", .libPaths())))
```
You might find the exact location by running `.libPaths()` in R GUI or RStudio.

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
