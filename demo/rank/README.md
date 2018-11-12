Learning to rank
====
XGBoost supports accomplishing ranking tasks. In ranking scenario, data are often grouped and we need the [group information file](../../doc/tutorials/input_format.rst#group-input-format) to specify ranking tasks. The model used in XGBoost for ranking is the LambdaRank, this function is not yet completed. Currently, we provide pairwise rank.

### Parameters
The configuration setting is similar to the regression and binary classification setting, except user need to specify the objectives:

```
...
objective="rank:pairwise"
...
```
For more usage details please refer to the [binary classification demo](../binary_classification),

Instructions
====
The dataset for ranking demo is from LETOR04 MQ2008 fold1.
Before running the examples, you need to get the data by running:

```
./wgetdata.sh
```

### Command Line
Run the example:
```
./runexp.sh
```

### Python
There are two ways of doing ranking in python.  

Run the example using `xgboost.train`:
```
python rank.py
```

Run the example using `XGBRanker`:
```
python rank_sklearn.py
```
