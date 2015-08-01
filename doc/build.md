Build XGBoost
=============
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
---------------------------------
Here is the complete solution to use OpenMp-enabled compilers to install XGBoost.

1. Obtain gcc with openmp support by `brew install gcc --without-multilib` **or** clang with openmp by `brew install clang-omp`. The clang one is recommended because the first method requires us compiling gcc inside the machine (more than an hour in mine)! (BTW, `brew` is the de facto standard of `apt-get` on OS X. So installing [HPC](http://hpc.sourceforge.net/) separately is not recommended, but it should work.)

2. **if you are planing to use clang-omp** - in step 3 and/or 4, change line 9 in `xgboost/src/utils/omp.h` to

  ```C++
  #include <libiomp/omp.h> /* instead of #include <omp.h> */`
  ```

  to make it work, otherwise you might get this error

  `src/tree/../utils/omp.h:9:10: error: 'omp.h' file not found...`



3. Set the `Makefile` correctly for compiling cpp version xgboost then python version xgboost.

  ```Makefile
  export CC  = gcc-4.9
  export CXX = g++-4.9
  ```

  Or

  ```Makefile
  export CC = clang-omp
  export CXX = clang-omp++
  ```

  Remember to change `header` (mentioned in step 2) if using clang-omp.

  Then `cd xgboost` then `bash build.sh` to compile XGBoost. And go to `wrapper` sub-folder to install python version.

4. Set the `Makevars` file in highest piority for R.

  The point is, there are three `Makevars` : `~/.R/Makevars`, `xgboost/R-package/src/Makevars`, and `/usr/local/Cellar/r/3.2.0/R.framework/Resources/etc/Makeconf` (the last one obtained by running `file.path(R.home("etc"), "Makeconf")` in R), and `SHLIB_OPENMP_CXXFLAGS` is not set by default!! After trying, it seems that the first one has highest piority (surprise!).

  So, **add** or **change** `~/.R/Makevars` to the following lines:

  ```Makefile
  CC=gcc-4.9
  CXX=g++-4.9
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

  Again, remember to change `header` if using clang-omp.

  Then inside R, run

  ```R
  install.packages('xgboost/R-package/', repos=NULL, type='source')
  ```

  Or

  ```R
  devtools::install_local('xgboost/', subdir = 'R-package') # you may use devtools
  ```


Build with HDFS and S3 Support
------------------------------
* To build xgboost use with HDFS/S3 support and distributed learnig. It is recommended to build with dmlc, with the following steps
  - ```git clone https://github.com/dmlc/dmlc-core```
  - Follow instruction in dmlc-core/make/config.mk to compile libdmlc.a
  - In root folder of xgboost, type ```make dmlc=dmlc-core```
* This will allow xgboost to directly load data and save model from/to hdfs and s3
  - Simply replace the filename with prefix s3:// or hdfs://
* This xgboost that can be used for distributed learning
