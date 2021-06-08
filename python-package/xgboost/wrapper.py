from . import sklearn as xgb_sklearn
from sklearn import model_selection, base

valid_estimators = ['XGBClassifier',
                    'XGBRFClassifier',
                    'XGBRFRegressor',
                    'XGBRanker',
                    'XGBRegressor']


def get_wrapper(name, **kwargs):
    if name not in valid_estimators:
        raise ValueError(name, 'is not a valid estimator. \n Refer to valid_estimators.')
    subclass = getattr(xgb_sklearn, name)

    class XGBWrapper(subclass):
        def __init__(self, **_kwargs):
            super().__init__()
            super().set_params(**_kwargs)

        def fit(self, X, y, fit_params={'early_stopping_rounds': 5, 'verbose': 0}, validation_fraction=.5):
            classification = base.is_classifier(subclass)

            if classification:
                X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X,
                                                                                      y,
                                                                                      random_state=1,
                                                                                      test_size=validation_fraction,
                                                                                      stratify=y)
            else:
                X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X,
                                                                                      y,
                                                                                      random_state=1,
                                                                                      test_size=validation_fraction)

            valids = [X_valid,
                      y_valid]
            super().fit(X_train,
                        y_train,
                        eval_set=[valids],
                        **fit_params)
            return self

    return XGBWrapper(**kwargs)
