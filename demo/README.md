Awesome XGBoost
===============
This page contains a curated list of examples, tutorials, blogs about XGBoost usecases.
It is inspired by [awesome-MXNet](https://github.com/dmlc/mxnet/blob/master/example/README.md),
[awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning).

Please send a pull request if you find things that belongs to here.

Contents
--------
- [Code Examples](#code-examples)
  - [Features Walkthrough](#features-walkthrough)
  - [Basic Examples by Tasks](#basic-examples-by-tasks)
  - [Benchmarks](#benchmarks)
- [Machine Learning Challenge Winning Solutions](#machine-learning-challenge-winning-solutions)
- [Tutorials](#tutorials)
- [Usecases](#usecases)
- [Tools using XGBoost](#tools-using-xgboost)
- [Awards](#awards)
- [Windows Binaries](#windows-binaries)

Code Examples
-------------
### Features Walkthrough

This is a list of short codes introducing different functionalities of xgboost packages.

* Basic walkthrough of packages
  [python](guide-python/basic_walkthrough.py)
  [R](../R-package/demo/basic_walkthrough.R)
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/basic_walkthrough.jl)
  [PHP](https://github.com/bpachev/xgboost-php/blob/master/demo/titanic_demo.php)
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

- [Binary classification](binary_classification)
- [Multiclass classification](multiclass_classification)
- [Regression](regression)
- [Learning to Rank](rank)

### Benchmarks

- [Starter script for Kaggle Higgs Boson](kaggle-higgs)
- [Kaggle Tradeshift winning solution by daxiongshu](https://github.com/daxiongshu/kaggle-tradeshift-winning-solution)
- [Benchmarking the most commonly used open source tools for binary classification](https://github.com/szilard/benchm-ml#boosting-gradient-boosted-treesgradient-boosting-machines)


## Machine Learning Challenge Winning Solutions

XGBoost is extensively used by machine learning practitioners to create state of art data science solutions,
this is a list of machine learning winning solutions with XGBoost.
Please send pull requests if you find ones that are missing here.

- Maksims Volkovs, Guangwei Yu and Tomi Poutanen, 1st place of the [2017 ACM RecSys challenge](http://2017.recsyschallenge.com/). Link to [paper](http://www.cs.toronto.edu/~mvolkovs/recsys2017_challenge.pdf).
- Vlad Sandulescu, Mihai Chiru, 1st place of the [KDD Cup 2016 competition](https://kddcup2016.azurewebsites.net). Link to [the arxiv paper](http://arxiv.org/abs/1609.02728).
- Marios Michailidis, Mathias Müller and HJ van Veen, 1st place of the [Dato Truely Native? competition](https://www.kaggle.com/c/dato-native). Link to [the Kaggle interview](http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors/).
- Vlad Mironov, Alexander Guschin, 1st place of the [CERN LHCb experiment Flavour of Physics competition](https://www.kaggle.com/c/flavours-of-physics). Link to [the Kaggle interview](http://blog.kaggle.com/2015/11/30/flavour-of-physics-technical-write-up-1st-place-go-polar-bears/).
- Josef Slavicek, 3rd place of the [CERN LHCb experiment Flavour of Physics competition](https://www.kaggle.com/c/flavours-of-physics). Link to [the Kaggle interview](http://blog.kaggle.com/2015/11/23/flavour-of-physics-winners-interview-3rd-place-josef-slavicek/).
- Mario Filho, Josef Feigl, Lucas, Gilberto, 1st place of the [Caterpillar Tube Pricing competition](https://www.kaggle.com/c/caterpillar-tube-pricing). Link to [the Kaggle interview](http://blog.kaggle.com/2015/09/22/caterpillar-winners-interview-1st-place-gilberto-josef-leustagos-mario/).
- Qingchen Wang, 1st place of the [Liberty Mutual Property Inspection](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction). Link to [the Kaggle interview](http://blog.kaggle.com/2015/09/28/liberty-mutual-property-inspection-winners-interview-qingchen-wang/).
- Chenglong Chen, 1st place of the [Crowdflower Search Results Relevance](https://www.kaggle.com/c/crowdflower-search-relevance). Link to [the winning solution](https://www.kaggle.com/c/crowdflower-search-relevance/forums/t/15186/1st-place-winner-solution-chenglong-chen/).
- Alexandre Barachant (“Cat”) and Rafał Cycoń (“Dog”), 1st place of the [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection). Link to [the Kaggle interview](http://blog.kaggle.com/2015/10/12/grasp-and-lift-eeg-winners-interview-1st-place-cat-dog/).
- Halla Yang, 2nd place of the [Recruit Coupon Purchase Prediction Challenge](https://www.kaggle.com/c/coupon-purchase-prediction). Link to [the Kaggle interview](http://blog.kaggle.com/2015/10/21/recruit-coupon-purchase-winners-interview-2nd-place-halla-yang/).
- Owen Zhang, 1st place of the [Avito Context Ad Clicks competition](https://www.kaggle.com/c/avito-context-ad-clicks). Link to [the Kaggle interview](http://blog.kaggle.com/2015/08/26/avito-winners-interview-1st-place-owen-zhang/).
- Keiichi Kuroyanagi, 2nd place of the [Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings). Link to [the Kaggle interview](http://blog.kaggle.com/2016/03/17/airbnb-new-user-bookings-winners-interview-2nd-place-keiichi-kuroyanagi-keiku/).
- Marios Michailidis, Mathias Müller and Ning Situ, 1st place [Homesite Quote Conversion](https://www.kaggle.com/c/homesite-quote-conversion). Link to [the Kaggle interview](http://blog.kaggle.com/2016/04/08/homesite-quote-conversion-winners-write-up-1st-place-kazanova-faron-clobber/).

## Talks
- [XGBoost: A Scalable Tree Boosting System](http://datascience.la/xgboost-workshop-and-meetup-talk-with-tianqi-chen/) (video+slides) by Tianqi Chen at the Los Angeles Data Science meetup

## Tutorials

- [Machine Learning with XGBoost on Qubole Spark Cluster](https://www.qubole.com/blog/machine-learning-xgboost-qubole-spark-cluster/)
- [XGBoost Official RMarkdown Tutorials](https://xgboost.readthedocs.org/en/latest/R-package/index.html#tutorials)
- [An Introduction to XGBoost R Package](http://dmlc.ml/rstats/2016/03/10/xgboost.html) by Tong He
- [Open Source Tools & Data Science Competitions](http://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1) by Owen Zhang - XGBoost parameter tuning tips
* [Feature Importance Analysis with XGBoost in Tax audit](http://fr.slideshare.net/MichaelBENESTY/feature-importance-analysis-with-xgboost-in-tax-audit)
* [Winning solution of Kaggle Higgs competition: what a single model can do](http://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/)
- [XGBoost - eXtreme Gradient Boosting](http://www.slideshare.net/ShangxuanZhang/xgboost) by Tong He
- [How to use XGBoost algorithm in R in easy steps](http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/) by TAVISH SRIVASTAVA ([Chinese Translation 中文翻译](https://segmentfault.com/a/1190000004421821) by [HarryZhu](https://segmentfault.com/u/harryprince))
- [Kaggle Solution: What’s Cooking ? (Text Mining Competition)](http://www.analyticsvidhya.com/blog/2015/12/kaggle-solution-cooking-text-mining-competition/) by MANISH SARASWAT
- Better Optimization with Repeated Cross Validation and the XGBoost model - Machine Learning with R) by Manuel Amunategui ([Youtube Link](https://www.youtube.com/watch?v=Og7CGAfSr_Y)) ([Github Link](https://github.com/amunategui/BetterCrossValidation))
- [XGBoost Rossman Parameter Tuning](https://www.kaggle.com/khozzy/rossmann-store-sales/xgboost-parameter-tuning-template/run/90168/notebook) by [Norbert Kozlowski](https://www.kaggle.com/khozzy)
- [Featurizing log data before XGBoost](http://www.slideshare.net/DataRobot/featurizing-log-data-before-xgboost) by Xavier Conort, Owen Zhang etc
- [West Nile Virus Competition Benchmarks & Tutorials](http://blog.kaggle.com/2015/07/21/west-nile-virus-competition-benchmarks-tutorials/) by [Anna Montoya](http://blog.kaggle.com/author/annamontoya/)
- [Ensemble Decision Tree with XGBoost](https://www.kaggle.com/binghsu/predict-west-nile-virus/xgboost-starter-code-python-0-69) by [Bing Xu](https://www.kaggle.com/binghsu)
- [Notes on eXtreme Gradient Boosting](http://startup.ml/blog/xgboost) by ARSHAK NAVRUZYAN ([iPython Notebook](https://github.com/startupml/koan/blob/master/eXtreme%20Gradient%20Boosting.ipynb))
- [Complete Guide to Parameter Tuning in XGBoost](http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/) by Aarshay Jain
- [Practical XGBoost in Python online course](http://education.parrotprediction.teachable.com/courses/practical-xgboost-in-python) by Parrot Prediction
- [Spark and XGBoost using Scala](http://www.elenacuoco.com/2016/10/10/scala-spark-xgboost-classification/) by Elena Cuoco
## Usecases
If you have particular usecase of xgboost that you would like to highlight.
Send a PR to add a one sentence description:)

- XGBoost is used in [Kaggle Script](https://www.kaggle.com/scripts) to solve data science challenges.
- [Seldon predictive service powered by XGBoost](http://docs.seldon.io/iris-demo.html)
- XGBoost Distributed is used in [ODPS Cloud Service by Alibaba](https://yq.aliyun.com/articles/6355) (in Chinese)
- XGBoost is incoporated as part of [Graphlab Create](https://dato.com/products/create/) for scalable machine learning.
- [Hanjing Su](https://www.52cs.org) from Tencent data platform team: "We use distributed XGBoost for click through prediction in wechat shopping and lookalikes. The problems involve hundreds millions of users and thousands of features. XGBoost is cleanly designed and can be easily integrated into our production environment, reducing our cost in developments."
- [CNevd](https://github.com/CNevd) from autohome.com ad platform team: "Distributed XGBoost is used for click through rate prediction in our display advertising, XGBoost is highly efficient and flexible and can be easily used on our distributed platform, our ctr made a great improvement with hundred millions samples and millions features due to this awesome XGBoost"



## Tools using XGBoost

- [BayesBoost](https://github.com/mpearmain/BayesBoost) - Bayesian Optimization using xgboost and sklearn API
- [gp_xgboost_gridsearch](https://github.com/vatsan/gp_xgboost_gridsearch) - In-database parallel grid-search for XGBoost on [Greenplum](https://github.com/greenplum-db/gpdb) using PL/Python
- [tpot](https://github.com/rhiever/tpot) - A Python tool that automatically creates and optimizes machine learning pipelines using genetic programming.

## Awards
- [John Chambers Award](http://stat-computing.org/awards/jmc/winners.html) - 2016 Winner: XGBoost R Package, by Tong He (Simon Fraser University) and Tianqi Chen (University of Washington)

## Windows Binaries
Unofficial windows binaries and instructions on how to use them are hosted on [Guido Tapia's blog](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/)
