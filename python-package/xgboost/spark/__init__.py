# type: ignore
"""PySpark XGBoost integration interface"""

try:
    import pyspark
except ImportError as e:
    raise ImportError("pyspark package needs to be installed to use this module") from e

from .estimator import (
    SparkXGBClassifier,
    SparkXGBClassifierModel,
    SparkXGBRanker,
    SparkXGBRankerModel,
    SparkXGBRegressor,
    SparkXGBRegressorModel,
)

__all__ = [
    "SparkXGBClassifier",
    "SparkXGBClassifierModel",
    "SparkXGBRegressor",
    "SparkXGBRegressorModel",
    "SparkXGBRanker",
    "SparkXGBRankerModel",
]
