Regression
====
Using XGBoost for regression is very similar to using it for binary classification. We suggest that you can refer to the [binary classification demo](../binary_classification) first. In XGBoost if we use negative log likelihood as the loss function for regression, the training procedure is same as training binary classifier of XGBoost. 

### Tutorial
The dataset we used is the [computer hardware dataset from UCI repository](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware). The demo for regression is almost the same as the [binary classification demo](../binary_classification), except a little difference in general parameter:
```
# General parameter
# this is the only difference with classification, use reg:linear to do linear regression
# when labels are in [0,1] we can also use reg:logistic
objective = reg:linear
...

```

The input format is same as binary classification, except that the label is now the target regression values. We use linear regression here, if we want use objective = reg:logistic logistic regression, the label needed to be pre-scaled into [0,1].

