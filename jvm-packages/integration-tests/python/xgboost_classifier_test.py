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

from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from xgboost.spark import XGBoostClassifier, XGBoostClassificationModel

from spark_init_internal import get_spark


def test_save_xgboost_classifier(xgboost_tmp_path):
    params = {
        'objective': 'binary:logistic',
        'numRound': 5,
        'numWorkers': 2,
        'treeMethod': 'hist'
    }
    classifier = XGBoostClassifier(**params)

    classifier_path = xgboost_tmp_path + "xgboost-classifier"

    classifier.write().overwrite().save(classifier_path)
    classifier1 = XGBoostClassifier.load(classifier_path)
    assert classifier1.getObjective() == 'binary:logistic'
    assert classifier1.getNumRound() == 5
    assert classifier1.getNumWorkers() == 2
    assert classifier1.getTreeMethod() == 'hist'


def test_xgboost_regressor_training_without_error(xgboost_tmp_path):
    spark = get_spark()
    df = spark.createDataFrame([
        ("a", Vectors.dense([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("b", Vectors.dense([5.0, 6.0, 7.0, 8.0, 9.0]))],
        ["label", "features"])
    label_name = 'label_indexed'
    string_indexer = StringIndexer(inputCol="label", outputCol=label_name).fit(df)
    indexed_df = string_indexer.transform(df).select(label_name, 'features')
    params = {
        'objective': 'binary:logistic',
        'numRound': 5,
        'numWorkers': 1,
        'treeMethod': 'hist'
    }
    classifier = XGBoostClassifier(**params) \
        .setLabelCol(label_name) \
        .setFeaturesCol('features')

    classifier_path = xgboost_tmp_path + "xgboost-classifier"
    classifier.write().overwrite().save(classifier_path)
    classifier1 = XGBoostClassifier.load(classifier_path)

    model_path = xgboost_tmp_path + "xgboost-classifier-model"
    model = classifier1.fit(indexed_df)
    model.write().overwrite().save(model_path)
    model1 = XGBoostClassificationModel.load(model_path)
    model1.transform(df).show()
