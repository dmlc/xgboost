XGBoost Documentation
=====================
This is document of xgboost library.
XGBoost is short for eXtreme gradient boosting. This is a library that is designed, and optimized for boosted (tree) algorithms.
The goal of this library is to push the extreme of the computation limits of machines to provide a ***scalable***, ***portable*** and ***accurate***
for large scale tree boosting.


This document is hosted at http://xgboost.readthedocs.org/. You can also browse most of the documents in github directly.

How to Get Started
------------------
The best way to get started to learn xgboost is by the examples. There are three types of examples you can find in xgboost.
* [Tutorials](#tutorials) are self-conatained tutorials on a complete data science tasks.
* [XGBoost Code Examples](../demo/) are collections of code and benchmarks of xgboost.
  - There is a walkthrough section in this to walk you through specific API features.
* [Highlight Solutions](#highlight-solutions) are presentations using xgboost to solve real world problems.
  - These examples are usually more advanced. You can usually find state-of-art solutions to many problems and challenges in here.

After you gets familiar with the interface, checkout the following additional resources
* [Frequently Asked Questions](faq.md)
* [Learning what is in Behind: Introduction to Boosted Trees](model.md)
* [User Guide](#user-guide) contains comprehensive list of documents of xgboost.
* [Developer Guide](dev-guide/contribute.md)

Tutorials
---------
Tutorials are self contained materials that teaches you how to achieve a complete data science task with xgboost, these
are great resources to learn xgboost by real examples. If you think you have something that belongs to here, send a pull request.
* [Binary classification using XGBoost Command Line](../demo/binary_classification/) (CLI)
  - This tutorial introduces the basic usage of CLI version of xgboost
* [Introduction of XGBoost in Python](python/python_intro.md) (python)
  - This tutorial introduces the python package of xgboost
* [Introduction to XGBoost in R](../R-package/vignettes/xgboostPresentation.Rmd) (R package)
  - This is a general presentation about xgboost in R.
* [Discover your data with XGBoost in R](../R-package/vignettes/discoverYourData.Rmd) (R package)
  - This tutorial explaining feature analysis in xgboost.
* [Understanding XGBoost Model on Otto Dataset](../demo/kaggle-otto/understandingXGBoostModel.Rmd) (R package)
  - This tutorial teaches you how to use xgboost to compete kaggle otto challenge.


Highlight Solutions
-------------------
This section is about blogposts, presentation and videos discussing how to use xgboost to solve your interesting problem. If you think something belongs to here, send a pull request.
* [Kaggle CrowdFlower winner's solution by Chenglong Chen](https://github.com/ChenglongChen/Kaggle_CrowdFlower)
* [Kaggle Malware Prediction winner's solution](https://github.com/xiaozhouwang/kaggle_Microsoft_Malware)
* [Kaggle Tradeshift winning solution by daxiongshu](https://github.com/daxiongshu/kaggle-tradeshift-winning-solution)
* [Feature Importance Analysis with XGBoost in Tax audit](http://fr.slideshare.net/MichaelBENESTY/feature-importance-analysis-with-xgboost-in-tax-audit)
* Video tutorial: [Better Optimization with Repeated Cross Validation and the XGBoost model](https://www.youtube.com/watch?v=Og7CGAfSr_Y)
* [Winning solution of Kaggle Higgs competition: what a single model can do](http://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/)

User Guide
----------
* [Frequently Asked Questions](faq.md)
* [Introduction to Boosted Trees](model.md)
* [Using XGBoost in Python](python/python_intro.md)
* [Using XGBoost in R](../R-package/vignettes/xgboostPresentation.Rmd)
* [Learning to use XGBoost by Example](../demo)
* [External Memory Version](external_memory.md)
* [Text input format](input_format.md)
* [Build Instruction](build.md)
* [Parameters](parameter.md)
* [Notes on Parameter Tunning](param_tuning.md)

Developer Guide
---------------
* [Developer Guide](dev-guide/contribute.md)

API Reference
-------------
* [Python API Reference](python/python_api.rst)
