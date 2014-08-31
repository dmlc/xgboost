This is a Fork of XGBoost from https://github.com/tqchen/xgboost

In the main repo you already find 2 windows projects for the porting of the executable and the python library.

Here you have:

  1) a c# dll wrapper, meaning the passage from unmanaged to managed code, in https://github.com/giuliohome/xgboost/tree/master/windows/xgboost_sharp_wrapper
  
  2) the c# Higgs Kaggle demo, instead of the python one (actually you will get a higher score with the c# version, due to some changes I've made) in https://github.com/giuliohome/xgboost/tree/master/windows/kaggle_higgs_demo
  

next steps:

  I will upload a c# cv implementation for the demo very soon
