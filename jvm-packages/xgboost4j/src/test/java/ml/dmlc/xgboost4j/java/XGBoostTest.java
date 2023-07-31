/*
 Copyright (c) 2023 by Contributors

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

package ml.dmlc.xgboost4j.java;

import junit.framework.TestCase;
import ml.dmlc.xgboost4j.LabeledPoint;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class XGBoostTest {

  private String composeEvalInfo(String metric, String evalName) {
    return "[0]\t" + evalName + "-" + metric + ":" + "\ttest";
  }

  @Test
  public void testIsMaximizeEvaluation() {
    String[] minimum_metrics = {"mape", "logloss", "error", "others"};
    String[] evalNames = {"set-abc"};

    HashMap<String, Object> params = new HashMap<>();

    // test1, infer the metric from faked log
    for (String x : XGBoost.MAXIMIZ_METRICES) {
      String evalInfo = composeEvalInfo(x, evalNames[0]);
      TestCase.assertTrue(XGBoost.isMaximizeEvaluation(evalInfo, evalNames, params));
    }

    // test2, the direction for mape should be minimum
    String evalInfo = composeEvalInfo("mape", evalNames[0]);
    TestCase.assertFalse(XGBoost.isMaximizeEvaluation(evalInfo, evalNames, params));

    // test3, force maximize_evaluation_metrics
    params.clear();
    params.put("maximize_evaluation_metrics", true);
    // auc should be max,
    evalInfo = composeEvalInfo("auc", evalNames[0]);
    TestCase.assertTrue(XGBoost.isMaximizeEvaluation(evalInfo, evalNames, params));

    params.clear();
    params.put("maximize_evaluation_metrics", false);
    // auc should be min,
    evalInfo = composeEvalInfo("auc", evalNames[0]);
    TestCase.assertFalse(XGBoost.isMaximizeEvaluation(evalInfo, evalNames, params));

    // test4, set the metric manually
    for (String x : XGBoost.MAXIMIZ_METRICES) {
      params.clear();
      params.put("eval_metric", x);
      evalInfo = composeEvalInfo(x, evalNames[0]);
      TestCase.assertTrue(XGBoost.isMaximizeEvaluation(evalInfo, evalNames, params));
    }

    // test5, set the metric manually
    for (String x : minimum_metrics) {
      params.clear();
      params.put("eval_metric", x);
      evalInfo = composeEvalInfo(x, evalNames[0]);
      TestCase.assertFalse(XGBoost.isMaximizeEvaluation(evalInfo, evalNames, params));
    }

  }

  @Test
  public void testEarlyStop() throws XGBoostError {
    Random random = new Random(1);

    java.util.ArrayList<Float> labelall = new java.util.ArrayList<Float>();
    int nrep = 3000;
    java.util.List<LabeledPoint> blist = new java.util.LinkedList<LabeledPoint>();
    for (int i = 0; i < nrep; ++i) {
      LabeledPoint p = new LabeledPoint(
        i % 2, 4,
        new int[]{0, 1, 2, 3},
        new float[]{random.nextFloat(), random.nextFloat(), random.nextFloat(), random.nextFloat()});
      blist.add(p);
      labelall.add(p.label());
    }

    DMatrix dmat = new DMatrix(blist.iterator(), null);

    int round = 50;
    int earlyStop = 2;

    HashMap<String, Object> mapParams = new HashMap<>();
    mapParams.put("eta", 0.1);
    mapParams.put("objective", "binary:logistic");
    mapParams.put("max_depth", 3);
    mapParams.put("eval_metric", "auc");
    mapParams.put("silent", 0);

    HashMap<String, DMatrix> mapWatches = new HashMap<>();
    mapWatches.put("selTrain-*", dmat);

    try {
      Booster booster = XGBoost.train(dmat, mapParams, round, mapWatches, null, null, null, earlyStop);
      Map<String, String> attrs = booster.getAttrs();
      TestCase.assertTrue(Integer.valueOf(attrs.get("best_iteration")) < round - 1);
    } catch (Exception e) {
      TestCase.assertFalse(false);
    }

  }
}
