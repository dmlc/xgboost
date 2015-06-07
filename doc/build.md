Build XGBoost
====
* Run ```bash build.sh``` (you can also type make)
* If you have C++11 compiler, it is recommended to type ```make cxx11=1```
  - C++11 is not used by default
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
* You may get a error: -lgomp is not found
  - You can type ```make no_omp=1```, this will get you single thread xgboost
  - Alternatively, you can upgrade your compiler to compile multi-thread version
* Windows(VS 2010): see [../windows](../windows) folder
  - In principle, you put all the cpp files in the Makefile to the project, and build
* OS X with multi-threading support: see [next section](#openmp-for-os-x)

Build XGBoost in OS X with OpenMP
====
Here is the complete solution to use OpenMp-enabled compilers to install XGBoost.

1. Install dependencies.

  [Homebrew](http://brew.sh/) is the de facto standard of `apt-get` on OS X. We install OpenMP runtime and compiler (gcc or clang) using `brew`:

  ```bash
  # recommended
  # the clang shpped with OS X does not support OpenMP
  # so we need to install a special version of clang
  brew install clang-omp
  brew install libiomp && brew unlink libiomp && brew link libiomp
  ```

  or

  ```bash
  # not recommended, 
  # it may take more than an hour to compile gcc itself!
  brew install gcc --without-multilib
  brew install libiomp && brew unlink libiomp && brew link libiomp
  ```

  BTW, installing [HPC](http://hpc.sourceforge.net/) separately without using homebrew is not recommended, but it should work.


2. Download source code and Set up Makefile

  ```bash
  git clone https://github.com/dmlc/xgboost.git
  cd xgboost/
  ```

  Consider the lines inside `ifeq ($(UNAME), Darwin)` near line 13 and line 14. If you decide to use gcc, change the lines to these:

  ```Makefile
  export CC  = gcc-5
  export CXX = g++-5
  ```

  If you decide to use clang, no changes are needed.
  
  Then compile it:

  ```bash
  bash build.sh # or make
  ```

  You should have built xgboost successfully now.

3. Install python version.

  Like what you would do in Linux.

  ```bash
  # you are in xgboost/ now
  cd wrapper/
  python setup.py install
  ```


4. Set the `Makevars` file in highest piority for R. 

  The point is, there are three `Makevars` inside the machine: `~/.R/Makevars`, `xgboost/R-package/src/Makevars`, and `/usr/local/Cellar/r/3.2.0/R.framework/Resources/etc/Makeconf` (the last one obtained by runing `file.path(R.home("etc"), "Makeconf")` in R), and `SHLIB_OPENMP_CXXFLAGS` is not set by default!! After trying, it seems that the first one has highest piority (surprise!).

  So, **add** or **change** `~/.R/Makevars` to the following lines:

  ```Makefile
  CC=gcc-5
  CXX=g++-5
  SHLIB_OPENMP_CFLAGS = -fopenmp
  SHLIB_OPENMP_CXXFLAGS = -fopenmp
  SHLIB_OPENMP_FCFLAGS = -fopenmp
  SHLIB_OPENMP_FFLAGS = -fopenmp
  ```

  Or

  ```Makefile
  CC=clang-omp
  CXX=clang-omp++
  SHLIB_OPENMP_CFLAGS = -fopenmp
  SHLIB_OPENMP_CXXFLAGS = -fopenmp
  SHLIB_OPENMP_FCFLAGS = -fopenmp
  SHLIB_OPENMP_FFLAGS = -fopenmp
  ```

  Then inside R, set the working directory as the parent of xgboost folder, run 

  ```R
  install.packages('xgboost/R-package/', repos=NULL, type='source')
  ```
  
  Or
  
  ```R
  # you may use devtools
  devtools::install_local('xgboost/', subdir = 'R-package')
  ```


Build with HDFS and S3 Support
=====
* To build xgboost use with HDFS/S3 support and distributed learnig. It is recommended to build with dmlc, with the following steps
  - ```git clone https://github.com/dmlc/dmlc-core```
  - Follow instruction in dmlc-core/make/config.mk to compile libdmlc.a
  - In root folder of xgboost, type ```make dmlc=dmlc-core```
* This will allow xgboost to directly load data and save model from/to hdfs and s3
  - Simply replace the filename with prefix s3:// or hdfs://
* This xgboost that can be used for distributed learning
