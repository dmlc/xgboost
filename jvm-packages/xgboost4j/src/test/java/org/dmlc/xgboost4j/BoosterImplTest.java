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
package org.dmlc.xgboost4j;

import junit.framework.TestCase;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.junit.Test;

import java.util.*;
import java.util.Map.Entry;

/**
 * test cases for Booster
 *
 * @author hzx
 */
public class BoosterImplTest {
  public static class EvalError implements IEvaluation {
    private static final Log logger = LogFactory.getLog(EvalError.class);

    String evalMetric = "custom_error";

    public EvalError() {
    }

    @Override
    public String getMetric() {
      return evalMetric;
    }

    @Override
    public float eval(float[][] predicts, org.dmlc.xgboost4j.DMatrix dmat) {
      float error = 0f;
      float[] labels;
      try {
        labels = dmat.getLabel();
      } catch (XGBoostError ex) {
        logger.error(ex);
        return -1f;
      }
      int nrow = predicts.length;
      for (int i = 0; i < nrow; i++) {
        if (labels[i] == 0f && predicts[i][0] > 0) {
          error++;
        } else if (labels[i] == 1f && predicts[i][0] <= 0) {
          error++;
        }
      }

      return error / labels.length;
    }
  }

  @Test
  public void testBoosterBasic() throws XGBoostError {
    org.dmlc.xgboost4j.DMatrix trainMat = new org.dmlc.xgboost4j.DMatrix("../../demo/data/agaricus.txt.train");
    org.dmlc.xgboost4j.DMatrix testMat = new org.dmlc.xgboost4j.DMatrix("../../demo/data/agaricus.txt.test");

    //set params
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("eta", 1.0);
        put("max_depth", 2);
        put("silent", 1);
        put("objective", "binary:logistic");
      }
    };

    //set watchList
    HashMap<String, org.dmlc.xgboost4j.DMatrix> watches = new HashMap<>();

    watches.put("train", trainMat);
    watches.put("test", testMat);

    //set round
    int round = 2;

    //train a boost model
    Booster booster = XGBoost.train(paramMap, trainMat, round, watches, null, null);

    //predict raw output
    float[][] predicts = booster.predict(testMat, true);

    //eval
    IEvaluation eval = new EvalError();
    //error must be less than 0.1
    TestCase.assertTrue(eval.eval(predicts, testMat) < 0.1f);

    //test dump model

  }

  /**
   * test cross valiation
   *
   * @throws XGBoostError
   */
  @Test
  public void testCV() throws XGBoostError {
    //load train mat
    org.dmlc.xgboost4j.DMatrix trainMat = new org.dmlc.xgboost4j.DMatrix("../../demo/data/agaricus.txt.train");

    //set params
    Map<String, Object> param = new HashMap<String, Object>() {
      {
        put("eta", 1.0);
        put("max_depth", 3);
        put("silent", 1);
        put("nthread", 6);
        put("objective", "binary:logistic");
        put("gamma", 1.0);
        put("eval_metric", "error");
      }
    };

    //do 5-fold cross validation
    int round = 2;
    int nfold = 5;
    //set additional eval_metrics
    String[] metrics = null;

    String[] evalHist = XGBoost.crossValiation(param, trainMat, round, nfold, metrics,
            null, null);
  }
}
