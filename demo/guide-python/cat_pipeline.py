"""
Feature engineering pipeline for categorical data
=================================================

The script showcases how to keep the categorical data encoding consistent across
training and inference. There are many ways to attain the same goal, this script can be
used as a starting point.

See Also
--------
- :doc:`Tutorial </tutorials/categorical>`
- :ref:`sphx_glr_python_examples_categorical.py`
- :ref:`sphx_glr_python_examples_cat_in_the_dat.py`

"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

import xgboost as xgb


def make_example_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Generate data for demo."""
    n_samples = 2048
    rng = np.random.default_rng(1994)

    # We have three categorical features, while the rest are numerical.
    categorical_features = ["brand_id", "retailer_id", "category_id"]

    df = pd.DataFrame(
        np.random.randint(32, 96, size=(n_samples, 3)),
        columns=categorical_features,
    )

    df["price"] = rng.integers(100, 200, size=(n_samples,))
    df["stock_status"] = rng.choice([True, False], n_samples)
    df["on_sale"] = rng.choice([True, False], n_samples)
    df["label"] = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    X = df.drop(["label"], axis=1)
    y = df["label"]

    return X, y, categorical_features


def native() -> None:
    """Using the native XGBoost interface."""
    X, y, cat_feats = make_example_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1994, test_size=0.2
    )

    # Create an encoder based on training data.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    enc.set_output(transform="pandas")
    enc = enc.fit(X_train[cat_feats])

    def enc_transform(X: pd.DataFrame) -> pd.DataFrame:
        # don't make change inplace so that we can have demonstrations for encoding
        X = X.copy()
        cat_cols = enc.transform(X[cat_feats])
        for i, name in enumerate(cat_feats):
            # create pd.Series based on the encoder
            cat_cols[name] = pd.Categorical.from_codes(
                codes=cat_cols[name].astype(np.int32), categories=enc.categories_[i]
            )
        X[cat_feats] = cat_cols
        return X

    # Encode the data based on fitted encoder.
    X_train_enc = enc_transform(X_train)
    X_test_enc = enc_transform(X_test)
    # Train XGBoost model using the native interface.
    Xy_train = xgb.QuantileDMatrix(X_train_enc, y_train, enable_categorical=True)
    Xy_test = xgb.QuantileDMatrix(
        X_test_enc, y_test, enable_categorical=True, ref=Xy_train
    )
    booster = xgb.train({}, Xy_train)
    booster.predict(Xy_test)

    # Following shows that data are encoded consistently.

    # We first obtain result from newly encoded data
    predt0 = booster.inplace_predict(enc_transform(X_train.head(16)))
    # then we obtain result from already encoded data from training.
    predt1 = booster.inplace_predict(X_train_enc.head(16))

    np.testing.assert_allclose(predt0, predt1)


def pipeline() -> None:
    """Using the sklearn pipeline."""
    X, y, cat_feats = make_example_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=3, test_size=0.2
    )

    enc = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            # all categorical feature names end with "_id"
            make_column_selector(pattern=".*_id"),
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # No need to set pandas output, we use `feature_types` to indicate the type of
    # features.

    # enc.set_output(transform="pandas")

    feature_types = ["c" if fn in cat_feats else "q" for fn in X_train.columns]
    reg = xgb.XGBRegressor(
        feature_types=feature_types, enable_categorical=True, n_estimators=10
    )
    p = make_pipeline(enc, reg)
    p.fit(X_train, y_train)
    # check XGBoost is using the feature type correctly.
    model_types = reg.get_booster().feature_types
    assert model_types is not None
    for a, b in zip(model_types, feature_types):
        assert a == b

    # Following shows that data are encoded consistently.

    # We first create a slice of data that doesn't contain all the categories
    predt0 = p.predict(X_train.iloc[:16, :])
    # Then we use the dataframe that contains all the categories
    predt1 = p.predict(X_train)[:16]

    # The resulting encoding is the same
    np.testing.assert_allclose(predt0, predt1)


if __name__ == "__main__":
    pipeline()
    native()
