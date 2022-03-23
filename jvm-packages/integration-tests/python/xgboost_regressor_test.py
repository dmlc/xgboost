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

from python.spark_init_internal import get_spark


def test_save_xgboost_regressor(xgboost_tmp_path):
    params = {
        'objective': 'reg:squarederror',
        'numRound': 5,
        'numWorkers': 2,
        'treeMethod': 'hist'
    }

    regressor_path = xgboost_tmp_path + "xgboost-regressor"

    classifier = XGBoostRegressor(**params)
    classifier.write().overwrite().save(regressor_path)
    classifier1 = XGBoostRegressor.load(regressor_path)
    assert classifier1.getObjective() == 'reg:squarederror'
    assert classifier1.getNumRound() == 5
    assert classifier1.getNumWorkers() == 2
    assert classifier1.getTreeMethod() == 'hist'


def test_xgboost_regressor_training_without_error(xgboost_tmp_path):
    spark = get_spark()
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
    regressor_path = xgboost_tmp_path + "xgboost-regressor"
    regressor.write().overwrite().save(regressor_path)
    regressor1 = XGBoostRegressor.load(regressor_path)
    model = regressor1.fit(df)

    model_path = xgboost_tmp_path + "xgboost-regressor-model"
    model.write().overwrite().save(model_path)
    model1 = XGBoostRegressionModel.load(model_path)
    model1.transform(df).show()
