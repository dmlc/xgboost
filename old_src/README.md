Coding Guide
======
This file is intended to be notes about code structure in xgboost

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
* .h files are data structures and interface, which are needed to use functions in that layer.
* -inl.hpp files are implementations of interface, like cpp file in most project.
  - You only need to understand the interface file to understand the usage of that layer
* In each folder, there can be a .cpp file, that compiles the module of that layer

How to Hack the Code
======
* Add objective function: add to learner/objective-inl.hpp and register it in learner/objective.h ```CreateObjFunction``` 
  - You can also directly do it in python
* Add new evaluation metric: add to learner/evaluation-inl.hpp and register it in learner/evaluation.h ```CreateEvaluator``` 
* Add wrapper for a new language, most likely you can do it by taking the functions in python/xgboost_wrapper.h, which is purely C based, and call these C functions to use xgboost
