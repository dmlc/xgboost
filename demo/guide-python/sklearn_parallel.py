"""
Demo for using xgboost with sklearn
===================================
"""

import multiprocessing
from urllib.error import HTTPError

from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

if __name__ == "__main__":
    print("Parallel Parameter optimization")
    try:
        X, y = fetch_california_housing(return_X_y=True)
    except HTTPError:
        # Use a synthetic dataset instead if we couldn't
        X, y = make_regression(n_samples=20640, n_features=8, random_state=1234)
    # Make sure the number of threads is balanced.
    xgb_model = xgb.XGBRegressor(
        n_jobs=multiprocessing.cpu_count() // 2, tree_method="hist"
    )
    clf = GridSearchCV(
        xgb_model,
        {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200]},
        verbose=1,
        n_jobs=2,
    )
    clf.fit(X, y)
    print(clf.best_score_)
    print(clf.best_params_)
