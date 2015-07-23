DMLC/XGBoost
==================================

[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)  [![Gitter chat for developers at https://gitter.im/dmlc/xgboost](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/xgboost?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

An optimized general purpose gradient boosting library. The library is parallelized, and also provides an optimized distributed version.
It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework, including [Generalized Linear Model](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLM) and [Gradient Boosted Decision Trees](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting) (GBDT). XGBoost can also be [distributed](#features) and scale to Terascale data

Check out our [Committers and Contributors](CONTRIBUTORS.md) who help make xgboost better.

Documentation: [Documentation of dmlc/xgboost](doc/README.md)

Issue Tracker: [https://github.com/dmlc/xgboost/issues](https://github.com/dmlc/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Please join [XGBoost User Group](https://groups.google.com/forum/#!forum/xgboost-user/) to ask questions and share your experience on xgboost.
  - Use issue tracker for bug reports, feature requests etc.
  - Use the user group to post your experience, ask questions about general usages.

Distributed Version: [Distributed XGBoost](multi-node)

Highlights of Usecases: [Highlight Links](doc/README.md#highlight-links)

XGBoost is part of [Distributed Machine Learning Common](http://dmlc.github.io/) projects

What's New
==========
* XGBoost helps Chenglong Chen to win [Kaggle CrowdFlower Competition](https://www.kaggle.com/c/crowdflower-search-relevance)
  - Check out the winning solution at [Highlight links](doc/README.md#highlight-links)
* XGBoost-0.4 release, see [CHANGES.md](CHANGES.md#xgboost-04)
* XGBoost helps three champion teams to win [WWW2015  Microsoft Malware Classification Challenge (BIG 2015)](http://www.kaggle.com/c/malware-classification/forums/t/13490/say-no-to-overfitting-approaches-sharing)
  - Check out the winning solution at [Highlight links](doc/README.md#highlight-links)
* [External Memory Version](doc/external_memory.md)

Contributing to XGBoost
=========
XGBoost has been developed and used by a group of active community members. Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.
* Check out [Feature Wish List](https://github.com/dmlc/xgboost/labels/Wish-List) to see what can be improved, or open an issue if you want something.
* Contribute to the [documents and examples](https://github.com/dmlc/xgboost/blob/master/doc/) to share your experience with other users.
* Please add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md) after your patch has been merged.

Features
========
* Easily accessible in python, R, Julia, CLI
* Fast speed and memory efficient
  - Can be more than 10 times faster than GBM in sklearn and R
  - Handles sparse matrices, support external memory
* Accurate prediction, and used extensively by data scientists and kagglers
  - See [highlight links](https://github.com/dmlc/xgboost/blob/master/doc/README.md#highlight-links)
* Distributed and Portable
  - The distributed version runs on Hadoop (YARN), MPI, SGE etc.
  - Scales to billions of examples and beyond

Build
=======
* Run ```bash build.sh``` (you can also type make)
  - Normally it gives what you want
  - See [Build Instruction](doc/build.md) for more information

Version
=======
* Current version xgboost-0.4, a lot improvment has been made since 0.3
  - Change log in [CHANGES.md](CHANGES.md)
  - This version is compatible with 0.3x versions

License
=======
© Contributors, 2015. Licensed under an [Apache-2](https://github.com/dmlc/xgboost/blob/master/LICENSE) license.

XGBoost in Graphlab Create
==========================
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful python toolkit that allows you to do data manipulation, graph processing, hyper-parameter search, and visualization of TeraBytes scale data in one framework. Try the Graphlab Create in http://graphlab.com/products/create/quick-start-guide.html
* Nice blogpost by Jay Gu about using GLC boosted tree to solve kaggle bike sharing challenge: http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand
