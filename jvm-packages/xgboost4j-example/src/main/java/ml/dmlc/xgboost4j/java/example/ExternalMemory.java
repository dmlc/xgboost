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

import java.util.HashMap;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

/**
 * simple example for using external memory version
 *
 * @author hzx
 */
public class ExternalMemory {
  public static void main(String[] args) throws XGBoostError {
    //this is the only difference, add a # followed by a cache prefix name
    //several cache file with the prefix will be generated
    //currently only support convert from libsvm file
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train#dtrain.cache");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test#dtest.cache");

    //specify parameters
    HashMap<String, Object> params = new HashMap<String, Object>();
    params.put("eta", 1.0);
    params.put("max_depth", 2);
    params.put("silent", 1);
    params.put("objective", "binary:logistic");

    //performance notice: set nthread to be the number of your real cpu
    //some cpu offer two threads per core, for example, a 4 core cpu with 8 threads, in such case
    // set nthread=4
    //param.put("nthread", num_real_cpu);

    //specify watchList
    HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
    watches.put("train", trainMat);
    watches.put("test", testMat);

    //set round
    int round = 2;

    //train a boost model
    Booster booster = XGBoost.train(trainMat, params, round, watches, null, null);
  }
}
