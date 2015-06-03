# xgboost4j
this is a java wrapper for xgboost 

the structure of this wrapper is almost the same as the official python wrapper.

core of this wrapper is two classes:

* DMatrix: for handling data

* Booster: for train and predict

## usage:
  
  simple examples could be found in test package:

  * Simple Train Example: org.dmlc.xgboost4j.example.TrainMultiClassifierExample.java
  
  * Simple Predict Example: org.dmlc.xgboost4j.example.PredictExample.java
  
  * Cross Validation Example: org.dmlc.xgboost4j.example.CVExample.java
 

## build native library

for windows: open the xgboost.sln in windows folder, you will found the xgboostjavawrapper project, you should do the following steps first before build:
 * Select x64/win32 and Release in build
 * right click on xgboostjavawrapper project -> choose "Properties" -> click on "C/C++" in the window -> change the "Additional Include Directories" to fit your jdk install path.
 * rebuild all
 * move the dll "xgboostjavawrapper.dll" to "xgboost4j/src/main/resources/lib/"

for linux: 
 * modify the "export JAVAINCFLAGS" line in "Makefile.config" to fit your environment (you may change nothing if you have proper JAVA_HOME setting beforehand)
 * run "create_wrap.sh"