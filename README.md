<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png width=135/>  eXtreme Gradient Boosting
===========
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](https://xgboost.readthedocs.org)
[![CRAN Status Badge](http://www.r-pkg.org/badges/version/xgboost)](http://cran.r-project.org/web/packages/xgboost)
[![Gitter chat for developers at https://gitter.im/dmlc/xgboost](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/xgboost?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

An optimized general purpose gradient boosting library. The library is parallelized, and also provides an optimized distributed version.

It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework, including [Generalized Linear Model](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLM) and [Gradient Boosted Decision Trees](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting) (GBDT). XGBoost can also be [distributed](#features) and scale to Terascale data

XGBoost is part of [Distributed Machine Learning Common](http://dmlc.github.io/) <img src=https://avatars2.githubusercontent.com/u/11508361?v=3&s=20> projects

Contents
--------
* [What's New](#whats-new)
* [Version](#version)
* [Documentation](doc/index.md)
* [Build Instruction](doc/build.md)
* [Features](#features)
* [Distributed XGBoost](multi-node)
* [Usecases](doc/index.md#highlight-links)
* [Bug Reporting](#bug-reporting)
* [Contributing to XGBoost](#contributing-to-xgboost)
* [Committers and Contributors](CONTRIBUTORS.md)
* [License](#license)
* [XGBoost in Graphlab Create](#xgboost-in-graphlab-create)

What's New
----------

* XGBoost helps Owen Zhang to win the [Avito Context Ad Click competition](https://www.kaggle.com/c/avito-context-ad-clicks). Check out the [interview from Kaggle](http://blog.kaggle.com/2015/08/26/avito-winners-interview-1st-place-owen-zhang/).
* XGBoost helps Chenglong Chen to win [Kaggle CrowdFlower Competition](https://www.kaggle.com/c/crowdflower-search-relevance)
  Check out the [winning solution](https://github.com/ChenglongChen/Kaggle_CrowdFlower)
* XGBoost-0.4 release, see [CHANGES.md](CHANGES.md#xgboost-04)
* XGBoost helps three champion teams to win [WWW2015  Microsoft Malware Classification Challenge (BIG 2015)](http://www.kaggle.com/c/malware-classification/forums/t/13490/say-no-to-overfitting-approaches-sharing)
  Check out the [winning solution](doc/README.md#highlight-links)
* [External Memory Version](doc/external_memory.md)

Version
-------

* Current version xgboost-0.4
  - [Change log](CHANGES.md)
  - This version is compatible with 0.3x versions

Features
--------
* Easily accessible through CLI, [python](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/basic_walkthrough.py),
  [R](https://github.com/dmlc/xgboost/blob/master/R-package/demo/basic_walkthrough.R),
  [Julia](https://github.com/antinucleon/XGBoost.jl/blob/master/demo/basic_walkthrough.jl)
* Its fast! Benchmark numbers comparing xgboost, H20, Spark, R - [benchm-ml numbers](https://github.com/szilard/benchm-ml)
* Memory efficient - Handles sparse matrices, supports external memory
* Accurate prediction, and used extensively by data scientists and kagglers - [highlight links](https://github.com/dmlc/xgboost/blob/master/doc/README.md#highlight-links)
* Distributed version runs on Hadoop (YARN), MPI, SGE etc., scales to billions of examples.

Bug Reporting
-------------

* For reporting bugs please use the [xgboost/issues](https://github.com/dmlc/xgboost/issues) page.
* For generic questions or to share your experience using xgboost please use the [XGBoost User Group](https://groups.google.com/forum/#!forum/xgboost-user/)


Contributing to XGBoost
-----------------------

XGBoost has been developed and used by a group of active community members. Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.
* Check out [Feature Wish List](https://github.com/dmlc/xgboost/labels/Wish-List) to see what can be improved, or open an issue if you want something.
* Contribute to the [documents and examples](https://github.com/dmlc/xgboost/blob/master/doc/) to share your experience with other users.
* Please add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md) after your patch has been merged.

License
-------
© Contributors, 2015. Licensed under an [Apache-2](https://github.com/dmlc/xgboost/blob/master/LICENSE) license.

XGBoost in Graphlab Create
--------------------------
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful python toolkit that allows you to do data manipulation, graph processing, hyper-parameter search, and visualization of TeraBytes scale data in one framework. Try the [Graphlab Create](http://graphlab.com/products/create/quick-start-guide.html)
* Nice [blogpost](http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand) by Jay Gu about using GLC boosted tree to solve kaggle bike sharing challenge:
