xgboost: eXtreme Gradient Boosting 
=======
An optimized general purpose gradient boosting (tree) library.

Contributors: https://github.com/tqchen/xgboost/graphs/contributors

Turorial and Documentation: https://github.com/tqchen/xgboost/wiki

Questions and Issues: [https://github.com/tqchen/xgboost/issues](https://github.com/tqchen/xgboost/issues?q=is%3Aissue+label%3Aquestion)

xgboost-unity
=======
experimental branch(not usable yet): refactor xgboost, cleaner code, more flexibility

Build
======
* Simply type make
* If your compiler does not come with OpenMP support, it will fire an warning telling you that the code will compile into single thread mode, and you will get single thread xgboost
* You may get a error: -lgomp is not found, you can remove -fopenmp flag in Makefile to get single thread xgboost, or upgrade your compiler to compile multi-thread version

Project Logical Layout
=======
* Dependency order: io->learner->gbm->tree
  - All module depends on data.h
* tree are implementations of tree construction algorithms.
* gbm is gradient boosting interface, that takes trees and other base learner to do boosting.
  - gbm only takes gradient as sufficient statistics, it does not compute the gradient.
* learner is learning module that computes gradient for specific object, and pass it to GBM

File Naming Convention
======= 
* The project is templatized, to make it easy to adjust input data structure.
* .h files are data structures and interface, which are needed to use functions in that layer.
* -inl.hpp files are implementations of interface, like cpp file in most project.
  - You only need to understand the interface file to understand the usage of that layer
