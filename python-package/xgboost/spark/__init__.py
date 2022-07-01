"""PySpark XGBoost integration interface
"""

try:
    import pyspark
except ImportError:
    raise RuntimeError("xgboost spark python API requires pyspark package installed.")

from .estimator import (
    SparkXGBClassifier,
    SparkXGBClassifierModel,
    SparkXGBRegressor,
    SparkXGBRegressorModel,
)

__all__ = [
    "SparkXGBClassifier",
    "SparkXGBClassifierModel",
    "SparkXGBRegressor",
    "SparkXGBRegressorModel",
]
