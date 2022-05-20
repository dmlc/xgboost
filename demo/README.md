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
- [Integrations with 3rd party software](#integrations-with-3rd-party-software)
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

- [Binary classification](CLI/binary_classification)
- [Multiclass classification](multiclass_classification)
- [Regression](CLI/regression)
- [Learning to Rank](rank)

### Benchmarks

- [Starter script for Kaggle Higgs Boson](kaggle-higgs)
- [Kaggle Tradeshift winning solution by daxiongshu](https://github.com/daxiongshu/kaggle-tradeshift-winning-solution)
- [Benchmarking the most commonly used open source tools for binary classification](https://github.com/szilard/benchm-ml#boosting-gradient-boosted-treesgradient-boosting-machines)


## Machine Learning Challenge Winning Solutions

XGBoost is extensively used by machine learning practitioners to create state of art data science solutions,
this is a list of machine learning winning solutions with XGBoost.
Please send pull requests if you find ones that are missing here.

- Bishwarup Bhattacharjee, 1st place winner of [Allstate Claims Severity](https://www.kaggle.com/competitions/allstate-claims-severity/overview) conducted on December 2016. Link to [discussion](https://www.kaggle.com/competitions/allstate-claims-severity/discussion/26416)
- Benedikt Schifferer, Gilberto Titericz, Chris Deotte, Christof Henkel, Kazuki Onodera, Jiwei Liu, Bojan Tunguz, Even Oldridge, Gabriel De Souza Pereira Moreira and Ahmet Erdem, 1st place winner of [Twitter RecSys Challenge 2020](https://recsys-twitter.com/) conducted from June,20-August,20. [GPU Accelerated Feature Engineering and Training for Recommender Systems](https://medium.com/rapids-ai/winning-solution-of-recsys2020-challenge-gpu-accelerated-feature-engineering-and-training-for-cd67c5a87b1f)
- Eugene Khvedchenya,Jessica Fridrich, Jan Butora, Yassine Yousfi 1st place winner in [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis/overview). Link to [discussion](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168546)
- Dan Ofer, Seffi Cohen, Noa Dagan, Nurit, 1st place in WiDS Datathon 2020. Link to [discussion](https://www.kaggle.com/c/widsdatathon2020/discussion/133189)
- Chris Deotte, Konstantin Yakovlev 1st place in [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/overview). Link to [discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308)
- Giba, Lucasz, 1st place winner in [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge) organized on August,2018. Solution [discussion](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/65272) and [code](https://www.kaggle.com/titericz/winner-model-giba-single-xgb-lb0-5178/comments)
- Beluga, 2nd place and Evgeny Nekrasov, 3rd place winner in Statoil/C-CORE Iceberg Classifier Challenge'2018. Link to [discussion](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48294)
- Radek Osmulski, 1st place of the [iMaterialist Challenge (Fashion) at FGVC5](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/overview). Link to [the winning solution](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/discussion/57944).
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
- Gilberto Titericz, Stanislav Semenov, 1st place in challenge to classify products into the correct category organized by Otto Group in 2015. Link to [challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge). Link to [kaggle winning solution](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335)
- Darius Barušauskas, 1st place winner in [Predicting Red Hat Business Value](https://www.kaggle.com/c/predicting-red-hat-business-value). Link to [interview](https://medium.com/kaggle-blog/red-hat-business-value-competition-1st-place-winners-interview-darius-baru%C5%A1auskas-646692a2841b). Link to [discussion](https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23786)
- David Austin, Weimin Wang, 1st place winner in [Iceberg-classifier-challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/leaderboard) Link to [discussion](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/48241)
- Kazuki Onodera, Kazuki Fujikawa, 2nd place winner in [OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction](https://www.kaggle.com/c/stanford-covid-vaccine/overview) Link to [Discussion](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709)
- Prarthana Bhat, 2nd place winner in [DYD Competition](https://datahack.analyticsvidhya.com/contest/date-your-data/). Link to [Solution](https://github.com/analyticsvidhya/DateYourData/blob/master/Prathna_Bhat_Model.R).

## Talks
- [XGBoost: A Scalable Tree Boosting System](http://datascience.la/xgboost-workshop-and-meetup-talk-with-tianqi-chen/) (video+slides) by Tianqi Chen at the Los Angeles Data Science meetup

## Tutorials

- [XGBoost Training with Dask, using Saturn Cloud](https://www.saturncloud.io/docs/tutorials/xgboost/)
- [Machine Learning with XGBoost on Qubole Spark Cluster](https://www.qubole.com/blog/machine-learning-xgboost-qubole-spark-cluster/)
- [XGBoost Official RMarkdown Tutorials](https://xgboost.readthedocs.org/en/latest/R-package/index.html#tutorials)
- [An Introduction to XGBoost R Package](http://dmlc.ml/rstats/2016/03/10/xgboost.html) by Tong He
- [Open Source Tools & Data Science Competitions](http://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1) by Owen Zhang - XGBoost parameter tuning tips
* [Feature Importance Analysis with XGBoost in Tax audit](http://fr.slideshare.net/MichaelBENESTY/feature-importance-analysis-with-xgboost-in-tax-audit)
* [Winning solution of Kaggle Higgs competition: what a single model can do](http://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/)
- [XGBoost - eXtreme Gradient Boosting](http://www.slideshare.net/ShangxuanZhang/xgboost) by Tong He
- [How to use XGBoost algorithm in R in easy steps](http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/) by TAVISH SRIVASTAVA ([Chinese Translation 中文翻译](https://segmentfault.com/a/1190000004421821) by [HarryZhu](https://segmentfault.com/u/harryprince))
- [Kaggle Solution: What’s Cooking ? (Text Mining Competition)](http://www.analyticsvidhya.com/blog/2015/12/kaggle-solution-cooking-text-mining-competition/) by MANISH SARASWAT
- Better Optimization with Repeated Cross Validation and the XGBoost model - Machine Learning with R) by Manuel Amunategui ([Youtube Link](https://www.youtube.com/watch?v=Og7CGAfSr_Y)) ([GitHub Link](https://github.com/amunategui/BetterCrossValidation))
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
- Distribute XGBoost as Rest API server from Jupyter notebook with [BentoML](https://github.com/bentoml/bentoml). [Link to notebook](https://github.com/bentoml/BentoML/blob/master/examples/xgboost-predict-titanic-survival/XGBoost-titanic-survival-prediction.ipynb)
- [Seldon predictive service powered by XGBoost](https://docs.seldon.io/projects/seldon-core/en/latest/servers/xgboost.html)
- XGBoost Distributed is used in [ODPS Cloud Service by Alibaba](https://yq.aliyun.com/articles/6355) (in Chinese)
- XGBoost is incoporated as part of [Graphlab Create](https://dato.com/products/create/) for scalable machine learning.
- [Hanjing Su](https://www.52cs.org) from Tencent data platform team: "We use distributed XGBoost for click through prediction in wechat shopping and lookalikes. The problems involve hundreds millions of users and thousands of features. XGBoost is cleanly designed and can be easily integrated into our production environment, reducing our cost in developments."
- [CNevd](https://github.com/CNevd) from autohome.com ad platform team: "Distributed XGBoost is used for click through rate prediction in our display advertising, XGBoost is highly efficient and flexible and can be easily used on our distributed platform, our ctr made a great improvement with hundred millions samples and millions features due to this awesome XGBoost"

## Tools using XGBoost

- [BayesBoost](https://github.com/mpearmain/BayesBoost) - Bayesian Optimization using xgboost and sklearn API
- [FLAML](https://github.com/microsoft/FLAML) - An open source AutoML library 
designed to automatically produce accurate machine learning models with low computational cost. FLAML includes [XGBoost as one of the default learners](https://github.com/microsoft/FLAML/blob/main/flaml/model.py) and can also be used as a fast hyperparameter tuning tool for XGBoost ([code example](https://microsoft.github.io/FLAML/docs/Examples/AutoML-for-XGBoost)).
- [gp_xgboost_gridsearch](https://github.com/vatsan/gp_xgboost_gridsearch) - In-database parallel grid-search for XGBoost on [Greenplum](https://github.com/greenplum-db/gpdb) using PL/Python
- [tpot](https://github.com/rhiever/tpot) - A Python tool that automatically creates and optimizes machine learning pipelines using genetic programming.

## Integrations with 3rd party software
Open source integrations with XGBoost:
* [Neptune.ai](http://neptune.ai/) - Experiment management and collaboration tool for ML/DL/RL specialists. Integration has a form of the [XGBoost callback](https://docs.neptune.ai/integrations/xgboost.html) that automatically logs training and evaluation metrics, as well as saved model (booster), feature importance chart and visualized trees.
* [Optuna](https://optuna.org/) - An open source hyperparameter optimization framework to automate hyperparameter search. Optuna integrates with XGBoost in the [XGBoostPruningCallback](https://optuna.readthedocs.io/en/stable/reference/integration.html#optuna.integration.XGBoostPruningCallback) that let users easily prune unpromising trials.
* [dtreeviz](https://github.com/parrt/dtreeviz) - A python library for decision tree visualization and model interpretation. Starting from version 1.0, dtreeviz is able to visualize tree ensembles produced by XGBoost.

## Awards
- [John Chambers Award](http://stat-computing.org/awards/jmc/winners.html) - 2016 Winner: XGBoost R Package, by Tong He (Simon Fraser University) and Tianqi Chen (University of Washington)
- [InfoWorld’s 2019 Technology of the Year Award](https://www.infoworld.com/article/3336072/application-development/infoworlds-2019-technology-of-the-year-award-winners.html)

## Windows Binaries
Unofficial windows binaries and instructions on how to use them are hosted on [Guido Tapia's blog](http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/)
