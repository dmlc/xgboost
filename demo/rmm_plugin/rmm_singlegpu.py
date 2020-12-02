import xgboost as xgb
import rmm
from sklearn.datasets import make_classification

# Initialize RMM pool allocator
rmm.reinitialize(pool_allocator=True)

X, y = make_classification(n_samples=10000, n_informative=5, n_classes=3)
dtrain = xgb.DMatrix(X, label=y)

params = {'max_depth': 8, 'eta': 0.01, 'objective': 'multi:softprob', 'num_class': 3,
          'tree_method': 'gpu_hist'}
# XGBoost will automatically use the RMM pool allocator
bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train')])
