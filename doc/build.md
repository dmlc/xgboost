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

OpenMP for OS X
====
* For users who want OpenMP support using [Homebrew](http://brew.sh/), run ```brew update``` (ensures that you install gcc-4.9 or above) and ```brew install gcc --without-multilib```. Once it is installed, edit [../Makefile](../Makefile) by replacing:
  ```bash
  export CC  = gcc
  export CXX = g++
  ```
  with
  ```bash
  export CC  = gcc-4.9
  export CXX = g++-4.9
  ```
  Then run ```bash build.sh``` normally.
  
* For users who want to use [High Performance Computing for Mac OS X](http://hpc.sourceforge.net/), download the GCC 4.9 binary tar ball and follow the installation guidance to install them under `/usr/local`. Then edit [../Makefile](../Makefile) by replacing:
  ```
  export CC  = gcc
  export CXX = g++
  ```
  with
  ```
  export CC  = /usr/local/bin/gcc
  export CXX = /usr/local/bin/g++
  ```
  Then run ```bash build.sh``` normally. This solution is given by [Phil Culliton](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost/68308#post68308).

Build with HDFS and S3 Support
=====
* To build xgboost use with HDFS/S3 support and distributed learnig. It is recommended to build with dmlc, with the following steps
  - ```git clone https://github.com/dmlc/dmlc-core```
  - Follow instruction in dmlc-core/make/config.mk to compile libdmlc.a
  - In root folder of xgboost, type ```make dmlc=dmlc-core```
* This will allow xgboost to directly load data and save model from/to hdfs and s3
  - Simply replace the filename with prefix s3:// or hdfs://
* This xgboost that can be used for distributed learning
