<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Awesome XGBoost](#awesome-xgboost)
  - [Contributing](#contributing)
  - [Examples](#examples)
    - [Features Walkthrough](#features-walkthrough)
    - [Basic Examples by Tasks](#basic-examples-by-tasks)
    - [Benchmarks](#benchmarks)
  - [Machine Learning Challenge Winning Solutions](#machine-learning-challenge-winning-solutions)
  - [Tutorials](#tutorials)
  - [Tools with XGBoost](#tools-with-xgboost)
  - [Services Powered by XGBoost](#services-powered-by-xgboost)
  - [Awards](#awards)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

Awesome XGBoost 
======
Welcome to the wonderland of XGBoost. This page contains a curated list of awesome XGBoost examples, tutorials and blogs. It is inspired by [awesome-MXNet](https://github.com/dmlc/mxnet/blob/master/example/README.md), [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning).

Contributing
----
* Contribution of examples, benchmarks is more than welcome!
* If you like to share how you use xgboost to solve your problem, send a pull request:)
* If you want to contribute to this list and the examples, please open a new pull request.

Examples
----
### Features Walkthrough

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

### Basic Examples by Tasks

Most of examples in this section are based on CLI or python version.
However, the parameter settings can be applied to all versions

* [Binary classification](binary_classification)
* [Multiclass classification](multiclass_classification)
* [Regression](regression)
* [Learning to Rank](rank)
* [Distributed Training](distributed-training)

### Benchmarks

* [Starter script for Kaggle Higgs Boson](kaggle-higgs)
* [Kaggle Tradeshift winning solution by daxiongshu](https://github.com/daxiongshu/kaggle-tradeshift-winning-solution)

## Machine Learning Challenge Winning Solutions

"Over the last six months, a new algorithm has come up on Kaggle __winning every single competition__ in this category, it is an algorithm called __XGBoost__." -- Anthony Goldbloom, Founder & CEO of Kaggle (from his presentation "What Is Winning on Kaggle?" [youtube link](https://youtu.be/GTs5ZQ6XwUM?t=7m7s))

XGBoost has helped on these winning solutions:

* Marios Michailidis, Mathias Müller and HJ van Veen, 1st place of the [Dato Truely Native? competition](https://www.kaggle.com/c/dato-native). Link to [the Kaggle interview](http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors/).
* Vlad Mironov, Alexander Guschin, 1st place of the [CERN LHCb experiment Flavour of Physics competition](https://www.kaggle.com/c/flavours-of-physics). Link to [the Kaggle interview](http://blog.kaggle.com/2015/11/30/flavour-of-physics-technical-write-up-1st-place-go-polar-bears/).
* Josef Slavicek, 3rd place of the [CERN LHCb experiment Flavour of Physics competition](https://www.kaggle.com/c/flavours-of-physics). Link to [the Kaggle interview](http://blog.kaggle.com/2015/11/23/flavour-of-physics-winners-interview-3rd-place-josef-slavicek/).
* Mario Filho, Josef Feigl, Lucas, Gilberto, 1st place of the [Caterpillar Tube Pricing competition](https://www.kaggle.com/c/caterpillar-tube-pricing). Link to [the Kaggle interview](http://blog.kaggle.com/2015/09/22/caterpillar-winners-interview-1st-place-gilberto-josef-leustagos-mario/).
* Qingchen Wang, 1st place of the [Liberty Mutual Property Inspection](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction). Link to [the Kaggle interview] (http://blog.kaggle.com/2015/09/28/liberty-mutual-property-inspection-winners-interview-qingchen-wang/).
* Chenglong Chen, 1st place of the [Crowdflower Search Results Relevance](https://www.kaggle.com/c/crowdflower-search-relevance). [Link to the winning solution](https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/15186/1st-place-winner-solution-chenglong-chen/).
* Alexandre Barachant (“Cat”) and Rafał Cycoń (“Dog”), 1st place of the [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection). Link to [the Kaggle interview](http://blog.kaggle.com/2015/10/12/grasp-and-lift-eeg-winners-interview-1st-place-cat-dog/).
* Halla Yang, 2nd place of the [Recruit Coupon Purchase Prediction Challenge](https://www.kaggle.com/c/coupon-purchase-prediction). Link to [the Kaggle interview](http://blog.kaggle.com/2015/10/21/recruit-coupon-purchase-winners-interview-2nd-place-halla-yang/).
* Owen Zhang, 1st place of the [Avito Context Ad Clicks competition](https://www.kaggle.com/c/avito-context-ad-clicks). Link to [the Kaggle interview](http://blog.kaggle.com/2015/08/26/avito-winners-interview-1st-place-owen-zhang/).

There are many other great winning solutions and interviews, but this list is [too small](https://en.wikipedia.org/wiki/Fermat%27s_Last_Theorem) to put all of them here. Please send pull requests if important ones appear.


## Tutorials

* "[Open Source Tools & Data Science Competitions](http://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1)" by Owen Zhang - XGBoost parameter tuning tips
* "[Tips for data science competitions](http://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions)" by Owen Zhang - Page 14
* "[XGBoost - eXtreme Gradient Boosting](http://www.slideshare.net/ShangxuanZhang/xgboost)" by Tong He
* "[How to use XGBoost algorithm in R in easy steps](http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/)" by TAVISH SRIVASTAVA ([Chinese Translation 中文翻译](https://segmentfault.com/a/1190000004421821) by [HarryZhu](https://segmentfault.com/u/harryprince))
* "[Kaggle Solution: What’s Cooking ? (Text Mining Competition)](http://www.analyticsvidhya.com/blog/2015/12/kaggle-solution-cooking-text-mining-competition/)" by MANISH SARASWAT
* "Better Optimization with Repeated Cross Validation and the XGBoost model - Machine Learning with R)" by Manuel Amunategui ([Youtube Link](https://www.youtube.com/watch?v=Og7CGAfSr_Y)) ([Github Link](https://github.com/amunategui/BetterCrossValidation))
* "[XGBoost Rossman Parameter Tuning](https://www.kaggle.com/khozzy/rossmann-store-sales/xgboost-parameter-tuning-template/run/90168/notebook)" by [Norbert Kozlowski](https://www.kaggle.com/khozzy)
* "[Featurizing log data before XGBoost](http://www.slideshare.net/DataRobot/featurizing-log-data-before-xgboost)" by Xavier Conort, Owen Zhang etc
* "[West Nile Virus Competition Benchmarks & Tutorials](http://blog.kaggle.com/2015/07/21/west-nile-virus-competition-benchmarks-tutorials/)" by [Anna Montoya](http://blog.kaggle.com/author/annamontoya/)
* "[Ensemble Decision Tree with XGBoost](https://www.kaggle.com/binghsu/predict-west-nile-virus/xgboost-starter-code-python-0-69)" by [Bing Xu](https://www.kaggle.com/binghsu)
* "[Notes on eXtreme Gradient Boosting](http://startup.ml/blog/xgboost)" by ARSHAK NAVRUZYAN ([iPython Notebook](https://github.com/startupml/koan/blob/master/eXtreme%20Gradient%20Boosting.ipynb))

## Tools with XGBoost

* [BayesBoost](https://github.com/mpearmain/BayesBoost) - Bayesian Optimization using xgboost and sklearn API

## Services Powered by XGBoost

* [Seldon predictive service powered by XGBoost](http://docs.seldon.io/iris-demo.html)
* [ODPS by Alibaba](https://yq.aliyun.com/articles/6355) (in Chinese)

## Awards

* [John Chambers Award](http://stat-computing.org/awards/jmc/winners.html) - 2016 Winner: XGBoost, by Tong He (Simon Fraser University) and Tianqi Chen (University of Washington)
