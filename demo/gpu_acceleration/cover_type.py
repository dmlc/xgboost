"""
Using xgboost on GPU devices
============================

Shows how to train a model on the `forest cover type
<https://archive.ics.uci.edu/ml/datasets/covertype>`_ dataset using GPU
acceleration. The forest cover type dataset has 581,012 rows and 54 features, making it
time consuming to process. We compare the run-time and accuracy of the GPU and CPU
histogram algorithms.

In addition, The demo showcases using GPU with other GPU-related libraries including
cupy and cuml. These libraries are not strictly required.

"""
import time

import cupy as cp
from cuml.model_selection import train_test_split
from sklearn.datasets import fetch_covtype

import xgboost as xgb

# Fetch dataset using sklearn
X, y = fetch_covtype(return_X_y=True)
X = cp.array(X)
y = cp.array(y)
y -= y.min()

# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, train_size=0.75, random_state=42
)

# Specify sufficient boosting iterations to reach a minimum
num_round = 3000

# Leave most parameters as default
clf = xgb.XGBClassifier(device="cuda", n_estimators=num_round)
# Train model
start = time.time()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
gpu_res = clf.evals_result()
print("GPU Training Time: %s seconds" % (str(time.time() - start)))

# Repeat for CPU algorithm
clf = xgb.XGBClassifier(device="cpu", n_estimators=num_round)
start = time.time()
cpu_res = clf.evals_result()
print("CPU Training Time: %s seconds" % (str(time.time() - start)))
