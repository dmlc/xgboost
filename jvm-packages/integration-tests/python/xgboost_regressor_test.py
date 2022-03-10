#
# Copyright (c) 2022 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark.ml.linalg import Vectors
from xgboost.spark import XGBoostRegressor, XGBoostRegressionModel

from python.spark_init_internal import get_spark_i_know_what_i_am_doing


def test_save_xgboost_regressor():
    params = {
        'objective': 'reg:squarederror',
        'numRound': 5,
        'numWorkers': 2,
        'treeMethod': 'hist'
    }
    classifier = XGBoostRegressor(**params)
    classifier.write().overwrite().save("/tmp/xgboost-integration-tests/xgboost-regressor")
    classifier1 = XGBoostRegressor.load("/tmp/xgboost-integration-tests/xgboost-regressor")
    assert classifier1.getObjective() == 'reg:squarederror'
    assert classifier1.getNumRound() == 5
    assert classifier1.getNumWorkers() == 2
    assert classifier1.getTreeMethod() == 'hist'


def test_xgboost_regressor_training_without_error():
    spark = get_spark_i_know_what_i_am_doing()
    df = spark.createDataFrame([
        (1.0, Vectors.dense(1.0)),
        (0.0, Vectors.dense(2.0))], ["label", "features"])
    params = {
        'objective': 'reg:squarederror',
        'numRound': 5,
        'numWorkers': 1,
        'treeMethod': 'hist'
    }
    regressor = XGBoostRegressor(**params) \
        .setLabelCol('label') \
        .setFeaturesCol('features')
    regressor.write().overwrite().save("/tmp/xgboost-integration-tests/xgboost-regressor")
    regressor1 = XGBoostRegressor.load("/tmp/xgboost-integration-tests/xgboost-regressor")
    model = regressor1.fit(df)
    model.write().overwrite().save("/tmp/xgboost-integration-tests/xgboost-regressor-model")
    model1 = XGBoostRegressionModel.load("/tmp/xgboost-integration-tests/xgboost-regressor-model")
    model1.transform(df).show()
