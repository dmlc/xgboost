Linear and Logistic Regression
====
* input format: LibSVM
* Local Example: [run-linear.sh](run-linear.sh)
* Runnig on YARN: [run-yarn.sh](run-yarn.sh)
  - You will need to have YARN 
  - Modify  ```../make/config.mk``` to set USE_HDFS=1
  - Run build.sh on [../../yarn](../../yarn) on to build yarn jar file
    
Parameters
===
All the parameters can be set by param=value

#### Important Parameters
* objective [default = logistic]
  - can be linear or logistic
* base_score [default = 0.5]
  - global bias, recommended set to mean value of label
* reg_L1 [default = 0]
  - l1 regularization co-efficient
* reg_L2 [default = 1]
  - l2 regularization co-efficient
* lbfgs_stop_tol [default = 1e-5]
  - relative tolerance level of loss reduction with respect to initial loss
* max_lbfgs_iter [default = 500]
  - maximum number of lbfgs iterations

### Optimization Related parameters
* min_lbfgs_iter [default = 5]
  - minimum number of lbfgs iterations
* max_linesearch_iter [default = 100] 
  - maximum number of iterations in linesearch
* linesearch_c1 [default = 1e-4] 
  - c1 co-efficient in backoff linesearch
* linesarch_backoff [default = 0.5]
  - backoff ratio in linesearch
 