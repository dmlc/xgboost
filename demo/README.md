XGBoost Code Examples
=====================
This folder contains all the code examples using xgboost.

* Contribution of examples, benchmarks is more than welcome!
* If you like to share how you use xgboost to solve your problem, send a pull request:)

Features Walkthrough
--------------------
This is a list of short codes introducing different functionalities of xgboost packages.
* Basic walkthrough of packages
  [python](guide-python/basic_walkthrough.py)
  [R](../R-package/demo/basic_walkthrough.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/basic_walkthrough.jl)
* Customize loss function, and evaluation metric
  [python](guide-python/custom_objective.py)
  [R](../R-package/demo/custom_objective.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/custom_objective.jl)
* Boosting from existing prediction
  [python](guide-python/boost_from_prediction.py)
  [R](../R-package/demo/boost_from_prediction.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/boost_from_prediction.jl)
* Predicting using first n trees
  [python](guide-python/predict_first_ntree.py)
  [R](../R-package/demo/predict_first_ntree.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/predict_first_ntree.jl)
* Generalized Linear Model
  [python](guide-python/generalized_linear_model.py)
  [R](../R-package/demo/generalized_linear_model.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/generalized_linear_model.jl)
* Cross validation
  [python](guide-python/cross_validation.py)
  [R](../R-package/demo/cross_validation.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/cross_validation.jl)
* Predicting leaf indices
  [python](guide-python/predict_leaf_indices.py)
  [R](../R-package/demo/predict_leaf_indices.R)

Basic Examples by Tasks
-----------------------
Most of examples in this section are based on CLI or python version.
However, the parameter settings can be applied to all versions
* [Binary classification](binary_classification)
* [Multiclass classification](multiclass_classification)
* [Regression](regression)
* [Learning to Rank](rank)
* [Distributed Training](distributed-training)

Benchmarks
----------
* [Starter script for Kaggle Higgs Boson](kaggle-higgs)
* [Kaggle Tradeshift winning solution by daxiongshu](https://github.com/daxiongshu/kaggle-tradeshift-winning-solution)

Machine Learning Challenge Winning Solutions
--------------------------------------------
* XGBoost helps Vlad Mironov, Alexander Guschin to win the [CERN LHCb experiment Flavour of Physics competition](https://www.kaggle.com/c/flavours-of-physics). Check out the [interview from Kaggle](http://blog.kaggle.com/2015/11/30/flavour-of-physics-technical-write-up-1st-place-go-polar-bears/).
* XGBoost helps Mario Filho, Josef Feigl, Lucas, Gilberto to win the [Caterpillar Tube Pricing competition](https://www.kaggle.com/c/caterpillar-tube-pricing). Check out the [interview from Kaggle](http://blog.kaggle.com/2015/09/22/caterpillar-winners-interview-1st-place-gilberto-josef-leustagos-mario/).
* XGBoost helps Halla Yang to win the [Recruit Coupon Purchase Prediction Challenge](https://www.kaggle.com/c/coupon-purchase-prediction). Check out the [interview from Kaggle](http://blog.kaggle.com/2015/10/21/recruit-coupon-purchase-winners-interview-2nd-place-halla-yang/).
