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

1. Obtain gcc-5.x.x with openmp support by `brew install gcc --without-multilib`. (`brew` is the de facto standard of `apt-get` on OS X. So installing [HPC](http://hpc.sourceforge.net/) separately is not recommended, but it should work.)

2. `cd xgboost` then `bash build.sh` to compile XGBoost.

3. Install xgboost package for Python and R

- For Python: go to `python-package` sub-folder to install python version with `python setup.py install` (or `sudo python setup.py install`).
- For R: Set the `Makevars` file in highest piority for R.

  The point is, there are three `Makevars` : `~/.R/Makevars`, `xgboost/R-package/src/Makevars`, and `/usr/local/Cellar/r/3.2.0/R.framework/Resources/etc/Makeconf` (the last one obtained by running `file.path(R.home("etc"), "Makeconf")` in R), and `SHLIB_OPENMP_CXXFLAGS` is not set by default!! After trying, it seems that the first one has highest piority (surprise!).

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
