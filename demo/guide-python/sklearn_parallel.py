from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
import xgboost as xgb

if __name__ == "__main__":
    print("Parallel Parameter optimization")
    boston = load_boston()

    y = boston['target']
    X = boston['data']
    xgb_model = xgb.XGBRegressor()
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]}, verbose=1,
                       n_jobs=2)
    clf.fit(X, y)
    print(clf.best_score_)
    print(clf.best_params_)
