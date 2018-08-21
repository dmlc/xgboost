Binary Classification
=====================
This is the quick start tutorial for xgboost CLI version.
Here we demonstrate how to use XGBoost for a binary classification task. Before getting started, make sure you compile xgboost in the root directory of the project by typing ```make```.
The script 'runexp.sh' can be used to run the demo. Here we use [mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom) from UCI machine learning repository.

### Tutorial
#### Generate Input Data
XGBoost takes LibSVM format. An example of faked input data is below:
```
1 101:1.2 102:0.03
0 1:2.1 10001:300 10002:400
...
```
Each line represent a single instance, and in the first line '1' is the instance label,'101' and '102' are feature indices, '1.2' and '0.03' are feature values. In the binary classification case, '1' is used to indicate positive samples, and '0' is used to indicate negative samples. We also support probability values in [0,1] as label, to indicate the probability of the instance being positive.


First we will transform the dataset into classic LibSVM format and split the data into training set and test set by running:
```
python mapfeat.py
python mknfold.py agaricus.txt 1
```
The two files, 'agaricus.txt.train' and 'agaricus.txt.test' will be used as training set and test set.

#### Training
Then we can run the training process:
```
../../xgboost mushroom.conf
```

mushroom.conf is the configuration for both training and testing. Each line containing the [attribute]=[value] configuration:

```conf
# General Parameters, see comment for each definition
# can be gbtree or gblinear
booster = gbtree
# choose logistic regression loss function for binary classification
objective = binary:logistic

# Tree Booster Parameters
# step size shrinkage
eta = 1.0
# minimum loss reduction required to make a further partition
gamma = 1.0
# minimum sum of instance weight(hessian) needed in a child
min_child_weight = 1
# maximum depth of a tree
max_depth = 3

# Task Parameters
# the number of round to do boosting
num_round = 2
# 0 means do not save any model except the final round model
save_period = 0
# The path of training data
data = "agaricus.txt.train"
# The path of validation data, used to monitor training process, here [test] sets name of the validation set
eval[test] = "agaricus.txt.test"
# The path of test data
test:data = "agaricus.txt.test"
```
We use the tree booster and logistic regression objective in our setting. This indicates that we accomplish our task using classic gradient boosting regression tree(GBRT), which is a promising method for binary classification.

The parameters shown in the example gives the most common ones that are needed to use xgboost.
If you are interested in more parameter settings, the complete parameter settings and detailed descriptions are [here](../../doc/parameter.md). Besides putting the parameters in the configuration file, we can set them by passing them as arguments as below:

```
../../xgboost mushroom.conf max_depth=6
```
This means that the parameter max_depth will be set as 6 rather than 3 in the conf file. When you use command line, make sure max_depth=6 is passed in as single argument, i.e. do not contain space in the argument. When a parameter setting is provided in both command line input and  the config file, the command line setting will override the setting in config file.

In this example, we use tree booster for gradient boosting. If you would like to use linear booster for regression, you can keep all the parameters except booster and the tree booster parameters as below:
```conf
# General Parameters
# choose the linear booster
booster = gblinear
...

# Change Tree Booster Parameters into Linear Booster Parameters
# L2 regularization term on weights, default 0
lambda = 0.01
# L1 regularization term on weights, default 0
alpha = 0.01
# L2 regularization term on bias, default 0
lambda_bias = 0.01

# Regression Parameters
...
```

#### Get Predictions
After training, we can use the output model to get the prediction of the test data:
```
../../xgboost mushroom.conf task=pred model_in=0002.model
```
For binary classification, the output predictions are probability confidence scores in [0,1], corresponds to the probability of the label to be positive.

#### Dump Model
This is a preliminary feature, so only tree models support text dump. XGBoost can display the tree models in text or JSON files, and we can scan the model in an easy way:
```
../../xgboost mushroom.conf task=dump model_in=0002.model name_dump=dump.raw.txt
../../xgboost mushroom.conf task=dump model_in=0002.model fmap=featmap.txt name_dump=dump.nice.txt
```

In this demo, the tree boosters obtained will be printed in dump.raw.txt and dump.nice.txt, and the latter one is easier to understand because of usage of feature mapping featmap.txt

Format of ```featmap.txt: <featureid> <featurename> <q or i or int>\n ```:
  - Feature id must be from 0 to number of features, in sorted order.
  - i means this feature is binary indicator feature
  - q means this feature is a quantitative value, such as age, time, can be missing
  - int means this feature is integer value (when int is hinted, the decision boundary will be integer)

#### Monitoring Progress
When you run training we can find there are messages displayed on screen
```
tree train end, 1 roots, 12 extra nodes, 0 pruned nodes ,max_depth=3
[0]  test-error:0.016139
boosting round 1, 0 sec elapsed

tree train end, 1 roots, 10 extra nodes, 0 pruned nodes ,max_depth=3
[1]  test-error:0.000000
```
The messages for evaluation are printed into stderr, so if you want only to log the evaluation progress, simply type
```
../../xgboost mushroom.conf 2>log.txt
```
Then you can find the following content in log.txt
```
[0]     test-error:0.016139
[1]     test-error:0.000000
```
We can also monitor both training and test statistics, by adding following lines to configure
```conf
eval[test] = "agaricus.txt.test"
eval[trainname] = "agaricus.txt.train"
```
Run the command again, we can find the log file becomes
```
[0]     test-error:0.016139     trainname-error:0.014433
[1]     test-error:0.000000     trainname-error:0.001228
```
The rule is eval[name-printed-in-log] = filename, then the file will be added to monitoring process, and evaluated each round.

xgboost also supports monitoring multiple metrics, suppose we also want to monitor average log-likelihood of each prediction during training, simply add ```eval_metric=logloss``` to configure. Run again, we can find the log file becomes
```
[0]     test-error:0.016139     test-negllik:0.029795   trainname-error:0.014433        trainname-negllik:0.027023
[1]     test-error:0.000000     test-negllik:0.000000   trainname-error:0.001228        trainname-negllik:0.002457
```
### Saving Progress Models
If you want to save model every two round, simply set save_period=2. You will find 0002.model in the current folder. If you want to change the output folder of models, add model_dir=foldername. By default xgboost saves the model of last round.

#### Continue from Existing Model
If you want to continue boosting from existing model, say 0002.model, use
```
../../xgboost mushroom.conf model_in=0002.model num_round=2 model_out=continue.model
```
xgboost will load from 0002.model continue boosting for 2 rounds, and save output to continue.model. However, beware that the training and evaluation data specified in mushroom.conf should not change when you use this function.
#### Use Multi-Threading
When you are working with a large dataset, you may want to take advantage of parallelism. If your compiler supports OpenMP, xgboost is naturally multi-threaded, to set number of parallel running add ```nthread``` parameter to you configuration.
Eg. ```nthread=10```

Set nthread to be the number of your real cpu (On Unix, this can be found using ```lscpu```)
Some systems will have ```Thread(s) per core = 2```, for example, a 4 core cpu with 8 threads, in such case set ```nthread=4``` and not 8.

