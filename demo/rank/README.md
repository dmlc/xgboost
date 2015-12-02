Learning to rank
====
XGBoost supports accomplishing ranking tasks. In ranking scenario, data are often grouped and we need the [group information file](../../doc/input_format.md#group-input-format) to specify ranking tasks. The model used in XGBoost for ranking is the LambdaRank, this function is not yet completed. Currently, we provide pairwise rank. 

### Parameters
The configuration setting is similar to the regression and binary classification setting,except user need to specify the objectives:

```
...
objective="rank:pairwise"
...
```
For more usage details please refer to the [binary classification demo](../binary_classification), 

Instructions
====
The dataset for ranking demo is from LETOR04 MQ2008 fold1, 
You can use the following command to run the example

Get the data: ./wgetdata.sh
Run the example: ./runexp.sh

