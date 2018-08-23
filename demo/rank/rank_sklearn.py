#!/usr/bin/python
import xgboost as xgb
from sklearn.datasets import load_svmlight_file


#  This script demonstrate how to do ranking with XGBRanker
x_train, y_train = load_svmlight_file("mq2008.train")
x_valid, y_valid = load_svmlight_file("mq2008.vali")
x_test, y_test = load_svmlight_file("mq2008.test")

group_train = []
with open("mq2008.train.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_train.append(int(line.split("\n")[0]))

group_valid = []
with open("mq2008.vali.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_valid.append(int(line.split("\n")[0]))

group_test = []
with open("mq2008.test.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_test.append(int(line.split("\n")[0]))

params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
          'gamma': 1.0, 'min_child_weight': 0.1,
          'max_depth': 6, 'n_estimators': 4}
model = xgb.sklearn.XGBRanker(**params)
model.fit(x_train, y_train, group_train,
          eval_set=[(x_valid, y_valid)], eval_group=[group_valid])
pred = model.predict(x_test)
