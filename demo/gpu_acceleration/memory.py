import xgboost as xgb
import numpy as np
import time
import pickle
import GPUtil

n = 10000
m = 1000
X = np.random.random((n, m))
y = np.random.random(n)

param = {'objective': 'binary:logistic',
         'tree_method': 'gpu_hist'
         }
iterations = 5
dtrain = xgb.DMatrix(X, label=y)

# High memory usage
# active bst objects with device memory persist across iterations
boosters = []
for i in range(iterations):
    bst = xgb.train(param, dtrain)
    boosters.append(bst)

print("Example 1")
GPUtil.showUtilization()
del boosters

# Better memory usage
# The bst object can be destroyed by the python gc, freeing device memory
# The gc may not immediately free the object, so more than one booster can be allocated at a time
boosters = []
for i in range(iterations):
    bst = xgb.train(param, dtrain)
    boosters.append(pickle.dumps(bst))

print("Example 2")
GPUtil.showUtilization()
del boosters

# Best memory usage
# The gc explicitly frees the booster before starting the next iteration
boosters = []
for i in range(iterations):
    bst = xgb.train(param, dtrain)
    boosters.append(pickle.dumps(bst))
    del bst

print("Example 3")
GPUtil.showUtilization()
del boosters
