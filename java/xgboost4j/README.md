# xgboost4j
this is a java wrapper for xgboost (https://github.com/dmlc/xgboost)  
the structure of this wrapper is almost the same as the official python wrapper.
core of this wrapper is two classes:

* DMatrix： for handling data

* Booster: for train and predict

## usage:
  
  simple examples could be found in test package:

  * Simple Train Example: org.dmlc.xgboost4j.TrainExample.java
  
  * Simple Predict Example: org.dmlc.xgboost4j.PredictExample.java
  
  * Cross Validation Example: org.dmlc.xgboost4j.example.CVExample.java
  
## native library:
  
  only 64-bit linux/windows is supported now, if you want to build native wrapper library yourself, please refer to 
  https://github.com/yanqingmen/xgboost-java, and put your native library to the "./src/main/resources/lib" folder and replace the originals. (either "libxgboostjavawrapper.so" for linux or "xgboostjavawrapper.dll" for windows) 
