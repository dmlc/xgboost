"""
Demo for using xgboost with sklearn
===================================
"""
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_california_housing
import xgboost as xgb
import multiprocessing

if __name__ == "__main__":
    print("Parallel Parameter optimization")
    X, y = fetch_california_housing(return_X_y=True)
    xgb_model = xgb.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2)
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]}, verbose=1,
                       n_jobs=2)
    clf.fit(X, y)
    print(clf.best_score_)
    print(clf.best_params_)
