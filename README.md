XGBoost: eXtreme Gradient Boosting 
==================================

[![Join the chat at https://gitter.im/dmlc/xgboost](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/xgboost?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
An optimized general purpose gradient boosting library. The library is parallelized, and also provides an optimized distributed version.
It implements machine learning algorithm under gradient boosting framework, including generalized linear model and gradient boosted regression tree (GBDT). XGBoost can also also distributed and scale to Terascale data

Contributors: https://github.com/dmlc/xgboost/graphs/contributors

Documentations: [Documentation of xgboost](doc/README.md)

Issues Tracker: [https://github.com/dmlc/xgboost/issues](https://github.com/dmlc/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Please join [XGBoost User Group](https://groups.google.com/forum/#!forum/xgboost-user/) to ask questions and share your experience on xgboost.
  - Use issue tracker for bug reports, feature requests etc.
  - Use the user group to post your experience, ask questions about general usages.

Distributed Version: [Distributed XGBoost](multi-node)

Highlights of Usecases: [Highlight Links](doc/README.md#highlight-links)

What's New
==========
* [External Memory Version](doc/external_memory.md)
* XGBoost wins [WWW2015  Microsoft Malware Classification Challenge (BIG 2015)](http://www.kaggle.com/c/malware-classification/forums/t/13490/say-no-to-overfitting-approaches-sharing)
* XGBoost now support HDFS and S3
* [Distributed XGBoost now runs on YARN](https://github.com/dmlc/wormhole/tree/master/learn/xgboost)
* [xgboost user group](https://groups.google.com/forum/#!forum/xgboost-user/) for tracking changes, sharing your experience on xgboost
* New features in the lastest changes :)
  - Distributed version that scale xgboost to even larger problems with cluster
  - Feature importance visualization in R module, thanks to Michael Benesty
  - Predict leaf index, see [demo/guide-python/predict_leaf_indices.py](demo/guide-python/predict_leaf_indices.py)  
* XGBoost wins [Tradeshift Text Classification](https://kaggle2.blob.core.windows.net/forum-message-attachments/60041/1813/TradeshiftTextClassification.pdf?sv=2012-02-12&se=2015-01-02T13%3A55%3A16Z&sr=b&sp=r&sig=5MHvyjCLESLexYcvbSRFumGQXCS7MVmfdBIY3y01tMk%3D)
* XGBoost wins [HEP meets ML Award in Higgs Boson Challenge](http://atlas.ch/news/2014/machine-learning-wins-the-higgs-challenge.html)

Features
========
* Sparse feature format:
  - Sparse feature format allows easy handling of missing values, and improve computation efficiency.
* Push the limit on single machine:
  - Efficient implementation that optimizes memory and computation.
* Speed: XGBoost is very fast
  - IN [demo/higgs/speedtest.py](demo/kaggle-higgs/speedtest.py), kaggle higgs data it is faster(on our machine 20 times faster using 4 threads) than sklearn.ensemble.GradientBoostingClassifier
* Layout of gradient boosting algorithm to support user defined objective
* Distributed and portable
  - The distributed version of xgboost is highly portable and can be used in different platforms
  - It inheritates all the optimizations made in single machine mode, maximumly utilize the resources using both multi-threading and distributed computing.

Build
=====
* Run ```bash build.sh``` (you can also type make)
* If you have C++11 compiler, it is recommended to type ```make cxx11=1```
  - C++11 is not used by default
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
* You may get a error: -lgomp is not found
  - You can type ```make no_omp=1```, this will get you single thread xgboost
  - Alternatively, you can upgrade your compiler to compile multi-thread version
* Windows(VS 2010): see [windows](windows) folder
  - In principle, you put all the cpp files in the Makefile to the project, and build
* OS X:
  - For users who want OpenMP support using [Homebrew](http://brew.sh/), run ```brew update``` (ensures that you install gcc-4.9 or above) and ```brew install gcc --without-multilib```. Once it is installed, edit [Makefile](Makefile/) by replacing:
  ```
  export CC  = gcc
  export CXX = g++
  ```
  with
  ```
  export CC  = gcc-4.9
  export CXX = g++-4.9
  ```
  Then run ```bash build.sh``` normally.
  
  - For users who want to use [High Performance Computing for Mac OS X](http://hpc.sourceforge.net/), download the GCC 4.9 binary tar ball and follow the installation guidance to install them under `/usr/local`. Then edit [Makefile](Makefile/) by replacing:
  ```
  export CC  = gcc
  export CXX = g++
  ```
  with
  ```
  export CC  = /usr/local/bin/gcc
  export CXX = /usr/local/bin/g++
  ```
  Then run ```bash build.sh``` normally. This solution is given by [Phil Culliton](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost/68308#post68308).

Build with HDFS and S3 Support
=====
* To build xgboost use with HDFS/S3 support and distributed learnig. It is recommended to build with dmlc, with the following steps
  - ```git clone https://github.com/dmlc/dmlc-core```
  - Follow instruction in dmlc-core/make/config.mk to compile libdmlc.a
  - In root folder of xgboost, type ```make dmlc=dmlc-core```
* This will allow xgboost to directly load data and save model from/to hdfs and s3
  - Simply replace the filename with prefix s3:// or hdfs://
* This xgboost that can be used for distributed learning

Version
=======
* This version xgboost-0.3, the code has been refactored from 0.2x to be cleaner and more flexibility
* This version of xgboost is not compatible with 0.2x, due to huge amount of changes in code structure
  - This means the model and buffer file of previous version can not be loaded in xgboost-3.0
* For legacy 0.2x code, refer to [Here](https://github.com/tqchen/xgboost/releases/tag/v0.22)
* Change log in [CHANGES.md](CHANGES.md)

XGBoost in Graphlab Create
==========================
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful python toolkit that allows you to data manipulation, graph processing, hyper-parameter search, and visualization of TeraBytes scale data in one framework. Try the Graphlab Create in http://graphlab.com/products/create/quick-start-guide.html
* Nice blogpost by Jay Gu using GLC boosted tree to solve kaggle bike sharing challenge: http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand
