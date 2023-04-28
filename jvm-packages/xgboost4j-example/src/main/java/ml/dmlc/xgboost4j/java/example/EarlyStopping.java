/*
 Copyright (c) 2021 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j.java.example;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.java.example.util.DataLoader;

public class EarlyStopping {
  public static void main(String[] args) throws IOException, XGBoostError {
    DataLoader.CSRSparseData trainCSR =
        DataLoader.loadSVMFile("../../demo/data/agaricus.txt.train?format=libsvm");
    DataLoader.CSRSparseData testCSR =
        DataLoader.loadSVMFile("../../demo/data/agaricus.txt.test?format=libsvm");

    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "false");
      }
    };

    DMatrix trainXy = new DMatrix(trainCSR.rowHeaders, trainCSR.colIndex, trainCSR.data,
                                  DMatrix.SparseType.CSR, 127);
    trainXy.setLabel(trainCSR.labels);
    DMatrix testXy = new DMatrix(testCSR.rowHeaders, testCSR.colIndex, testCSR.data,
                                 DMatrix.SparseType.CSR, 127);
    testXy.setLabel(testCSR.labels);

    int nRounds = 128;
    int nEarlyStoppingRounds = 4;

    Map<String, DMatrix> watches = new LinkedHashMap<>();
    watches.put("training", trainXy);
    watches.put("test", testXy);

    float[][] metrics = new float[watches.size()][nRounds];
    Booster booster = XGBoost.train(trainXy, paramMap, nRounds,
                                    watches, metrics, null, null, nEarlyStoppingRounds);

    int bestIter = Integer.valueOf(booster.getAttr("best_iteration"));
    float bestScore = Float.valueOf(booster.getAttr("best_score"));

    System.out.printf("Best iter: %d, Best score: %f\n", bestIter, bestScore);
  }
}
