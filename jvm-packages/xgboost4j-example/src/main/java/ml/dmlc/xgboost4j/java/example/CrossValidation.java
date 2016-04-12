/*
 Copyright (c) 2014 by Contributors

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

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

/**
 * an example of cross validation
 *
 * @author hzx
 */
public class CrossValidation {
  public static void main(String[] args) throws IOException, XGBoostError {
    //load train mat
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");

    //set params
    HashMap<String, Object> params = new HashMap<String, Object>();

    params.put("eta", 1.0);
    params.put("max_depth", 3);
    params.put("silent", 1);
    params.put("nthread", 6);
    params.put("objective", "binary:logistic");
    params.put("gamma", 1.0);
    params.put("eval_metric", "error");

    //do 5-fold cross validation
    int round = 2;
    int nfold = 5;
    //set additional eval_metrics
    String[] metrics = null;

    String[] evalHist = XGBoost.crossValidation(trainMat, params, round, nfold, metrics, null,
            null);
  }
}
