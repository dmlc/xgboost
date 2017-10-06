XGBoost Parameters
==================
Before running XGboost, we must set three types of parameters: general parameters, booster parameters and task parameters.
- General parameters relates to which booster we are using to do boosting, commonly tree or linear model
- Booster parameters depends on which booster you have chosen
- Learning Task parameters that decides on the learning scenario, for example, regression tasks may use different parameters with ranking tasks.
- Command line parameters that relates to behavior of CLI version of xgboost.

Parameters in R Package
-----------------------
In R-package, you can use .(dot) to replace underscore in the parameters, for example, you can use max.depth as max_depth. The underscore parameters are also valid in R.

General Parameters
------------------
* booster [default=gbtree]
  - which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
* silent [default=0]
  - 0 means printing running messages, 1 means silent mode.
* nthread [default to maximum number of threads available if not set]
  - number of parallel threads used to run xgboost
* num_pbuffer [set automatically by xgboost, no need to be set by user]
  - size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.
* num_feature [set automatically by xgboost, no need to be set by user]
  - feature dimension used in boosting, set to maximum dimension of the feature

Parameters for Tree Booster
---------------------------
* eta [default=0.3, alias: learning_rate]
  - step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
  - range: [0,1]
* gamma [default=0, alias: min_split_loss]
  - minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
  - range: [0,∞]
* max_depth [default=6]
  - maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting. 0 indicates no limit, limit is required for depth-wise grow policy.
  - range: [0,∞]
* min_child_weight [default=1]
  - minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
  - range: [0,∞]
* max_delta_step [default=0]
  - Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
  - range: [0,∞]
* subsample [default=1]
  - subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
  - range: (0,1]
* colsample_bytree [default=1]
  - subsample ratio of columns when constructing each tree.
  - range: (0,1]
* colsample_bylevel [default=1]
  - subsample ratio of columns for each split, in each level.
  - range: (0,1]
* lambda [default=1, alias: reg_lambda]
  - L2 regularization term on weights, increase this value will make model more conservative.
* alpha [default=0, alias: reg_alpha]
  - L1 regularization term on weights, increase this value will make model more conservative.
* tree_method, string [default='auto']
  - The tree construction algorithm used in XGBoost(see description in the [reference paper](http://arxiv.org/abs/1603.02754))
  - Distributed and external memory version only support approximate algorithm.
  - Choices: {'auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'}
    - 'auto': Use heuristic to choose faster one.
      - For small to medium dataset, exact greedy will be used.
      - For very large-dataset, approximate algorithm will be chosen.
      - Because old behavior is always use exact greedy in single machine,
        user will get a message when approximate algorithm is chosen to notify this choice.
    - 'exact': Exact greedy algorithm.
    - 'approx': Approximate greedy algorithm using sketching and histogram.
    - 'hist': Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
	- 'gpu_exact': GPU implementation of exact algorithm. 
	- 'gpu_hist': GPU implementation of hist algorithm. 
* sketch_eps, [default=0.03]
  - This is only used for approximate greedy algorithm.
  - This roughly translated into ```O(1 / sketch_eps)``` number of bins.
    Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy.
  - Usually user does not have to tune this.
    but consider setting to a lower number for more accurate enumeration.
  - range: (0, 1)
* scale_pos_weight, [default=1]
  - Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative  cases) / sum(positive cases) See [Parameters Tuning](how_to/param_tuning.md) for more discussion. Also see Higgs Kaggle competition demo for examples: [R](../demo/kaggle-higgs/higgs-train.R ), [py1](../demo/kaggle-higgs/higgs-numpy.py ), [py2](../demo/kaggle-higgs/higgs-cv.py ), [py3](../demo/guide-python/cross_validation.py)
* updater, [default='grow_colmaker,prune']
  - A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updater plugins exist:
    - 'grow_colmaker': non-distributed column-based construction of trees.
    - 'distcol': distributed tree construction with column-based data splitting mode.
    - 'grow_histmaker': distributed tree construction with row-based data splitting based on global proposal of histogram counting.
    - 'grow_local_histmaker': based on local histogram counting.
    - 'grow_skmaker': uses the approximate sketching algorithm.
    - 'sync': synchronizes trees in all distributed nodes.
    - 'refresh': refreshes tree's statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
    - 'prune': prunes the splits where loss < min_split_loss (or gamma).
  - In a distributed setting, the implicit updater sequence value would be adjusted as follows:
    - 'grow_histmaker,prune' when  dsplit='row' (or default) and prob_buffer_row == 1 (or default); or when data has multiple sparse pages
    - 'grow_histmaker,refresh,prune' when  dsplit='row' and prob_buffer_row < 1
    - 'distcol' when dsplit='col'
* refresh_leaf, [default=1]
  - This is a parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as well as tree nodes' stats are updated. When it is false, only node stats are updated.
* process_type, [default='default']
  - A type of boosting process to run.
  - Choices: {'default', 'update'}
    - 'default': the normal boosting process which creates new trees.
    - 'update': starts from an existing model and only updates its trees. In each boosting iteration, a tree from the initial model is taken, a specified sequence of updater plugins is run for that tree, and a modified tree is added to the new model. The new model would have either the same or smaller number of trees, depending on the number of boosting iteratons performed. Currently, the following built-in updater plugins could be meaningfully used with this process type: 'refresh', 'prune'. With 'update', one cannot use updater plugins that create new nrees.
* grow_policy, string [default='depthwise']
  - Controls a way new nodes are added to the tree.
  - Currently supported only if `tree_method` is set to 'hist'.
  - Choices: {'depthwise', 'lossguide'}
    - 'depthwise': split at nodes closest to the root.
    - 'lossguide': split at nodes with highest loss change.
* max_leaves, [default=0]
  - Maximum number of nodes to be added. Only relevant for the 'lossguide' grow policy.
* max_bin, [default=256]
  - This is only used if 'hist' is specified as `tree_method`.
  - Maximum number of discrete bins to bucket continuous features.
  - Increasing this number improves the optimality of splits at the cost of higher computation time.
* predictor, [default='cpu_predictor']
  - The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
    - 'cpu_predictor': Multicore CPU prediction algorithm.
    - 'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.

Additional parameters for Dart Booster
--------------------------------------
* sample_type [default="uniform"]
  - type of sampling algorithm.
    - "uniform": dropped trees are selected uniformly.
    - "weighted": dropped trees are selected in proportion to weight.
* normalize_type [default="tree"]
  - type of normalization algorithm.
    - "tree": new trees have the same weight of each of dropped trees.
      - weight of new trees are 1 / (k + learning_rate)
      - dropped trees are scaled by a factor of k / (k + learning_rate)
    - "forest": new trees have the same weight of sum of dropped trees (forest).
      - weight of new trees are 1 / (1 + learning_rate)
      - dropped trees are scaled by a factor of 1 / (1 + learning_rate)
* rate_drop [default=0.0]
  - dropout rate (a fraction of previous trees to drop during the dropout).
  - range: [0.0, 1.0]
* one_drop [default=0]
  - when this flag is enabled, at least one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout from the original DART paper).
* skip_drop [default=0.0]
  - Probability of skipping the dropout procedure during a boosting iteration.
    - If a dropout is skipped, new trees are added in the same manner as gbtree.
    - Note that non-zero skip_drop has higher priority than rate_drop or one_drop.
  - range: [0.0, 1.0]

Parameters for Linear Booster
-----------------------------
* lambda [default=0, alias: reg_lambda]
  - L2 regularization term on weights, increase this value will make model more conservative.
* alpha [default=0, alias: reg_alpha]
  - L1 regularization term on weights, increase this value will make model more conservative.
* lambda_bias [default=0, alias: reg_lambda_bias]
  - L2 regularization term on bias (no L1 reg on bias because it is not important)

Parameters for Tweedie Regression
---------------------------------
* tweedie_variance_power [default=1.5]
  - parameter that controls the variance of the Tweedie distribution
    - var(y) ~ E(y)^tweedie_variance_power
  - range: (1,2)
  - set closer to 2 to shift towards a gamma distribution
  - set closer to 1 to shift towards a Poisson distribution.

Learning Task Parameters
------------------------
Specify the learning task and the corresponding learning objective. The objective options are below:
* objective [default=reg:linear]
  - "reg:linear" --linear regression
  - "reg:logistic" --logistic regression
  - "binary:logistic" --logistic regression for binary classification, output probability
  - "binary:logitraw" --logistic regression for binary classification, output score before logistic transformation
  - "count:poisson" --poisson regression for count data, output mean of poisson distribution
    - max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
  - "multi:softmax" --set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
  - "multi:softprob" --same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
  - "rank:pairwise" --set XGBoost to do ranking task by minimizing the pairwise loss
  - "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Applications)
  - "reg:tweedie" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be [Tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Applications).
* base_score [default=0.5]
  - the initial prediction score of all instances, global bias
  - for sufficient number of iterations, changing this value will not have too much effect.
* eval_metric [default according to objective]
  - evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )
  - User can add multiple evaluation metrics, for python user, remember to pass the metrics in as list of parameters pairs instead of map, so that latter 'eval_metric' won't override previous one
  - The choices are listed below:
    - "rmse": [root mean square error](http://en.wikipedia.org/wiki/Root_mean_square_error)
    - "mae": [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
    - "logloss": negative [log-likelihood](http://en.wikipedia.org/wiki/Log-likelihood)
    - "error": Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
    - "error@t": a different than 0.5 binary classification threshold value could be specified by providing a numerical value through 't'.
    - "merror": Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
    - "mlogloss": [Multiclass logloss](https://www.kaggle.com/wiki/LogLoss)
    - "auc": [Area under the curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve) for ranking evaluation.
    - "ndcg":[Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/NDCG)
    - "map":[Mean average precision](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)
    - "ndcg@n","map@n": n can be assigned as an integer to cut off the top positions in the lists for evaluation.
    - "ndcg-","map-","ndcg@n-","map@n-": In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding "-" in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions.
training repeatedly
  - "poisson-nloglik": negative log-likelihood for Poisson regression
  - "gamma-nloglik": negative log-likelihood for gamma regression
  - "gamma-deviance": residual deviance for gamma regression
  - "tweedie-nloglik": negative log-likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter)
* seed [default=0]
  - random number seed.

Command Line Parameters
-----------------------
The following parameters are only used in the console version of xgboost
* use_buffer [default=1]
  - Whether to create a binary buffer from text input. Doing so normally will speed up loading times
* num_round
  - The number of rounds for boosting
* data
  - The path of training data
* test:data
  - The path of test data to do prediction
* save_period [default=0]
  - the period to save the model, setting save_period=10 means that for every 10 rounds XGBoost will save the model, setting it to 0 means not saving any model during the training.
* task [default=train] options: train, pred, eval, dump
  - train: training using data
  - pred: making prediction for test:data
  - eval: for evaluating statistics specified by eval[name]=filename
  - dump: for dump the learned model into text format (preliminary)
* model_in [default=NULL]
  - path to input model, needed for test, eval, dump, if it is specified in training, xgboost will continue training from the input model
* model_out [default=NULL]
  - path to output model after training finishes, if not specified, will output like 0003.model where 0003 is number of rounds to do boosting.
* model_dir [default=models]
  - The output directory of the saved models during training
* fmap
  - feature map, used for dump model
* name_dump [default=dump.txt]
  - name of model dump file
* name_pred [default=pred.txt]
  - name of prediction file, used in pred mode
* pred_margin [default=0]
  - predict margin instead of transformed probability
