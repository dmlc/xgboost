import sys

import xgboost4j

sys.modules["ml.dmlc.xgboost4j.scala.spark"] = xgboost4j
