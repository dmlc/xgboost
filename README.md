This is a Fork of XGBoost from https://github.com/tqchen/xgboost

In the main repo you already find 2 windows projects for the porting of the executable and the python library.

Here you have:

  1) a c# dll wrapper, meaning the passage from unmanaged to managed code, in https://github.com/giuliohome/xgboost/tree/master/windows/xgboost_sharp_wrapper
  
  2) the c# Higgs Kaggle demo, instead of the python one (actually you will get a higher score with the c# version, due to some changes I've made) in https://github.com/giuliohome/xgboost/tree/master/windows/kaggle_higgs_demo
  
  Start the demo from the root folder like this: 
  
  bin\x64\Debug\kaggle_higgs_demo.exe training_path.csv test_path.csv sharp_pred.csv NFoldCV
  
  NFoldCV: 0 => no cv , 5 = 5-fold-cv, 10 = 10-fold-cv :-)
  
  3) 5 fold cv implementation in c# for the demo: you see inline cv ams while training (of course on a completely separate set)
