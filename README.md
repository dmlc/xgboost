XGBoost: eXtreme Gradient Boosting 
==================================

An optimized general purpose gradient boosting library. The library is parallelized, and also provides an optimized distributed version.
It implements machine learning algorithm under gradient boosting framework, including generalized linear model and gradient boosted regression tree (GBDT). XGBoost can also also distributed and scale to Terascale data

Contributors: https://github.com/dmlc/xgboost/graphs/contributors

Documentations: [Documentation of xgboost](doc/README.md)

Issues Tracker: [https://github.com/dmlc/xgboost/issues](https://github.com/dmlc/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Please join [XGBoost User Group](https://groups.google.com/forum/#!forum/xgboost-user/) to ask questions and share your experience on xgboost.
  - Use issue tracker for bug reports, feature requests etc.
  - Use the user group to post your experience, ask questions about general usages.

Gitter for developers [![Gitter chat for developers at https://gitter.im/dmlc/xgboost](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/xgboost?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Distributed Version: [Distributed XGBoost](multi-node)

Highlights of Usecases: [Highlight Links](doc/README.md#highlight-links)

What's New
==========
* XGBoost-0.4 release, see [CHANGES.md](CHANGES.md#xgboost-04)
* XGBoost wins [WWW2015  Microsoft Malware Classification Challenge (BIG 2015)](http://www.kaggle.com/c/malware-classification/forums/t/13490/say-no-to-overfitting-approaches-sharing)
  - Checkout the winning solution at [Highlight links](doc/README.md#highlight-links)
* [External Memory Version](doc/external_memory.md)

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

XGBoost in Graphlab Create
==========================
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful python toolkit that allows you to data manipulation, graph processing, hyper-parameter search, and visualization of TeraBytes scale data in one framework. Try the Graphlab Create in http://graphlab.com/products/create/quick-start-guide.html
* Nice blogpost by Jay Gu using GLC boosted tree to solve kaggle bike sharing challenge: http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand
