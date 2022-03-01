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

from xgboost.spark import XGBoostClassifier


def test_xgboost_parameters_from_dictionary():
    xgb_params = {'objective': 'multi:softprob',
                  'treeMethod': 'hist',
                  'numWorkers': 1,
                  'labelCol': 'classIndex',
                  'featuresCol': 'features',
                  'numRound': 100,
                  'numClass': 3}
    xgb = XGBoostClassifier(**xgb_params)
    assert xgb.getObjective() == 'multi:softprob'
    assert xgb.getTreeMethod() == 'hist'
    assert xgb.getNumWorkers() == 1
    assert xgb.getLabelCol() == 'classIndex'
    assert xgb.getFeaturesCol() == 'features'
    assert xgb.getNumRound() == 100
    assert xgb.getNumClass() == 3


def test_xgboost_set_parameter():
    xgb = XGBoostClassifier()
    xgb.setObjective('multi:softprob')
    xgb.setTreeMethod('hist')
    xgb.setNumWorkers(1)
    xgb.setLabelCol('classIndex')
    xgb.setFeaturesCol('features')
    xgb.setNumRound(100)
    xgb.setNumClass(3)
    assert xgb.getObjective() == 'multi:softprob'
    assert xgb.getTreeMethod() == 'hist'
    assert xgb.getNumWorkers() == 1
    assert xgb.getLabelCol() == 'classIndex'
    assert xgb.getFeaturesCol() == 'features'
    assert xgb.getNumRound() == 100
    assert xgb.getNumClass() == 3
