# xgboost4j
this is a java wrapper for xgboost 

the structure of this wrapper is almost the same as the official python wrapper.

core of this wrapper is two classes:

* DMatrix: for handling data

* Booster: for train and predict

## usage:
  please refer to [xgboost4j.md](doc/xgboost4j.md) for more information.

  besides, simple examples could be found in [xgboost4j-demo](xgboost4j-demo/README.md)
 

## build native library

for windows: open the xgboost.sln in "../windows" folder, you will found the xgboost4j project, you should do the following steps to build wrapper library:
 * Select x64/win32 and Release in build
 * (if you have setted `JAVA_HOME` properly in windows environment variables, escape this step) right click on xgboost4j project -> choose "Properties" -> click on "C/C++" in the window -> change the "Additional Include Directories" to fit your jdk install path.
 * rebuild all
 * double click "create_wrap.bat" to set library to proper place

for linux: 
 * make sure you have installed jdk and `JAVA_HOME` has been setted properly
 * run "create_wrap.sh"

for osx:
 * make sure you have installed jdk
 * for single thread xgboost, simply run "create_wrap.sh"
 * for build with openMP, please refer to [build.md](../doc/build.md) to get openmp supported compiler first, and change the line "dis_omp=1" to "dis_omp=0" in "create_wrap.sh", then run "create_wrap.sh"