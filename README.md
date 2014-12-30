xgboost: eXtreme Gradient Boosting 
======
An optimized general purpose gradient boosting library. The library is parallelized using OpenMP. It implements machine learning algorithm under gradient boosting framework, including generalized linear model and gradient boosted regression tree. 

Contributors: https://github.com/tqchen/xgboost/graphs/contributors

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki

Questions and Issues: [https://github.com/tqchen/xgboost/issues](https://github.com/tqchen/xgboost/issues?q=is%3Aissue+label%3Aquestion)

Examples Code: [Learning to use xgboost by examples](demo)

Notes on the Code: [Code Guide](src)

Learning about the model: [Introduction to Boosted Trees](http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
* This slide is made by Tianqi Chen to introduce gradient boosting in a statistical view.
* It present boosted tree learning as formal functional space optimization of defined objective.
* The model presented is used by xgboost for boosted trees

What's New
=====
* XGBoost wins [Tradeshift Text Classification](https://kaggle2.blob.core.windows.net/forum-message-attachments/60041/1813/TradeshiftTextClassification.pdf?sv=2012-02-12&se=2015-01-02T13%3A55%3A16Z&sr=b&sp=r&sig=5MHvyjCLESLexYcvbSRFumGQXCS7MVmfdBIY3y01tMk%3D)
* XGBoost wins [HEP meets ML Award in Higgs Boson Challenge](http://atlas.ch/news/2014/machine-learning-wins-the-higgs-challenge.html)
* Thanks to Bing Xu, [XGBoost.jl](https://github.com/antinucleon/XGBoost.jl) allows you to use xgboost from Julia
* See the updated [demo folder](demo) for feature walkthrough
* Thanks to Tong He, the new [R package](R-package) is available

Features
======
* Sparse feature format:
  - Sparse feature format allows easy handling of missing values, and improve computation efficiency.
* Push the limit on single machine:
  - Efficient implementation that optimizes memory and computation.
* Speed: XGBoost is very fast
  - IN [demo/higgs/speedtest.py](demo/kaggle-higgs/speedtest.py), kaggle higgs data it is faster(on our machine 20 times faster using 4 threads) than sklearn.ensemble.GradientBoostingClassifier
* Layout of gradient boosting algorithm to support user defined objective

Build
=====
* Run ```bash build.sh``` (you can also type make)
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
* You may get a error: -lgomp is not found
  - You can type ```make no_omp=1```, this will get you single thread xgboost
  - Alternatively, you can upgrade your compiler to compile multi-thread version
* Windows(VS 2010): see [windows](windows) folder
  - In principle, you put all the cpp files in the Makefile to the project, and build
* OS X:
  - For users who want OpenMP support using [Homebrew](http://brew.sh/), run ```brew update``` (ensures that you install gcc-4.9 or above) and ```brew install gcc```. Once it is installed, edit [Makefile](Makefile/) by replacing:
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

Version
======
* This version xgboost-0.3, the code has been refactored from 0.2x to be cleaner and more flexibility
* This version of xgboost is not compatible with 0.2x, due to huge amount of changes in code structure
  - This means the model and buffer file of previous version can not be loaded in xgboost-3.0
* For legacy 0.2x code, refer to [Here](https://github.com/tqchen/xgboost/releases/tag/v0.22)
* Change log in [CHANGES.md](CHANGES.md)

XGBoost in Graphlab Create
======
* XGBoost is adopted as part of boosted tree toolkit in Graphlab Create (GLC). Graphlab Create is a powerful python toolkit that allows you to data manipulation, graph processing, hyper-parameter search, and visualization of TeraBytes scale data in one framework. Try the Graphlab Create in http://graphlab.com/products/create/quick-start-guide.html
* Nice blogpost by Jay Gu using GLC boosted tree to solve kaggle bike sharing challenge: http://blog.graphlab.com/using-gradient-boosted-trees-to-predict-bike-sharing-demand
