/*
 Copyright (c) 2014-2022 by Contributors

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

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import junit.framework.TestCase;
import org.junit.Test;

/**
 * test cases for Booster
 *
 * @author hzx
 */
public class BoosterImplTest {
  private String train_uri = "../../demo/data/agaricus.txt.train?indexing_mode=1&format=libsvm";
  private String test_uri = "../../demo/data/agaricus.txt.test?indexing_mode=1&format=libsvm";

  public static class EvalError implements IEvaluation {
    @Override
    public String getMetric() {
      return "custom_error";
    }

    @Override
    public float eval(float[][] predicts, DMatrix dmat) {
      float error = 0f;
      float[] labels;
      try {
        labels = dmat.getLabel();
      } catch (XGBoostError ex) {
        throw new RuntimeException(ex);
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

  private Booster trainBooster(DMatrix trainMat, DMatrix testMat) throws XGBoostError {
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
    HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();

    watches.put("train", trainMat);
    watches.put("test", testMat);

    //set round
    int round = 5;

    //train a boost model
    return XGBoost.train(trainMat, paramMap, round, watches, null, null);
  }

  @Test
  public void testBoosterBasic() throws XGBoostError, IOException {

    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);

    //predict raw output
    float[][] predicts = booster.predict(testMat, true, 0);

    //eval
    IEvaluation eval = new EvalError();
    //error must be less than 0.1
    TestCase.assertTrue(eval.eval(predicts, testMat) < 0.1f);
  }

  @Test
  public void saveLoadModelWithPath() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    IEvaluation eval = new EvalError();

    Booster booster = trainBooster(trainMat, testMat);
    // save and load
    File temp = File.createTempFile("temp", "model");
    temp.deleteOnExit();
    booster.saveModel(temp.getAbsolutePath());

    Booster bst2 = XGBoost.loadModel(temp.getAbsolutePath());
    assert (Arrays.equals(bst2.toByteArray("ubj"), booster.toByteArray("ubj")));
    assert (Arrays.equals(bst2.toByteArray("json"), booster.toByteArray("json")));
    assert (Arrays.equals(bst2.toByteArray("deprecated"), booster.toByteArray("deprecated")));
    float[][] predicts2 = bst2.predict(testMat, true, 0);
    TestCase.assertTrue(eval.eval(predicts2, testMat) < 0.1f);
  }

  @Test
  public void saveLoadModelWithStream() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);

    ByteArrayOutputStream output = new ByteArrayOutputStream();
    booster.saveModel(output);
    IEvaluation eval = new EvalError();
    Booster loadedBooster = XGBoost.loadModel(new ByteArrayInputStream(output.toByteArray()));
    float originalPredictError = eval.eval(booster.predict(testMat, true), testMat);
    TestCase.assertTrue("originalPredictErr:" + originalPredictError,
            originalPredictError < 0.1f);
    float loadedPredictError = eval.eval(loadedBooster.predict(testMat, true), testMat);
    TestCase.assertTrue("loadedPredictErr:" + loadedPredictError, loadedPredictError < 0.1f);
  }

  private static class IncreasingEval implements IEvaluation {
    private int value = 1;

    @Override
    public String getMetric() {
      return "inc";
    }

    @Override
    public float eval(float[][] predicts, DMatrix dmat) {
      return value++;
    }
  }

  @Test
  public void testDescendMetricsWithBoundaryCondition() {
    // maximize_evaluation_metrics = false
    int totalIterations = 11;
    int earlyStoppingRound = 10;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    int bestIteration = 0;

    for (int itr = 0; itr < totalIterations; itr++) {
      boolean es = XGBoost.shouldEarlyStop(earlyStoppingRound, itr, bestIteration);
      if (itr == totalIterations - 1) {
        TestCase.assertTrue(es);
      } else {
        TestCase.assertFalse(es);
      }
    }
  }

  @Test
  public void testEarlyStoppingForMultipleMetrics() {
    // maximize_evaluation_metrics = true
    int earlyStoppingRound = 3;
    int totalIterations = 5;
    int numOfMetrics = 3;
    float[][] metrics = new float[numOfMetrics][totalIterations];
    // Only assign metric values to the first dataset, zeros for other datasets
    for (int i = 0; i < numOfMetrics; i++) {
      for (int j = 0; j < totalIterations; j++) {
        metrics[0][j] = j;
      }
    }
    int bestIteration;

    for (int i = 0; i < totalIterations; i++) {
      bestIteration = i;
      boolean es = XGBoost.shouldEarlyStop(earlyStoppingRound, i, bestIteration);
      TestCase.assertFalse(es);
    }

    // when we have multiple datasets, only the last one was used to determinate early stop
    // Here we changed the metric of the first dataset, it doesn't have any effect to the final result
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    for (int i = 0; i < totalIterations; i++) {
      bestIteration = i;
      boolean es = XGBoost.shouldEarlyStop(earlyStoppingRound, i, bestIteration);
      TestCase.assertFalse(es);
    }

    // Now assign metric values to the last dataset.
    for (int i = 0; i < totalIterations; i++) {
      metrics[2][i] = totalIterations - i;
    }
    bestIteration = 0;

    for (int i = 0; i < totalIterations; i++) {
      // if any metrics off, we need to stop
      boolean es = XGBoost.shouldEarlyStop(earlyStoppingRound, i, bestIteration);
      if (i >= earlyStoppingRound) {
        TestCase.assertTrue(es);
      } else {
        TestCase.assertFalse(es);
      }
    }
  }

  @Test
  public void testDescendMetrics() {
    // maximize_evaluation_metrics = false
    int totalIterations = 10;
    int earlyStoppingRounds = 5;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    int bestIteration = 0;

    boolean es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    TestCase.assertTrue(es);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    bestIteration = totalIterations - 1;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    TestCase.assertFalse(es);

    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    metrics[0][4] = 1;
    metrics[0][9] = 5;

    bestIteration = 4;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    TestCase.assertTrue(es);
  }

  @Test
  public void testAscendMetricsWithBoundaryCondition() {
    // maximize_evaluation_metrics = true
    int totalIterations = 11;
    int earlyStoppingRounds = 10;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    int bestIteration = 0;

    for (int itr = 0; itr < totalIterations; itr++) {
      boolean es = XGBoost.shouldEarlyStop(earlyStoppingRounds, itr, bestIteration);
      if (itr == totalIterations - 1) {
        TestCase.assertTrue(es);
      } else {
        TestCase.assertFalse(es);
      }
    }
  }

  @Test
  public void testAscendMetrics() {
    // maximize_evaluation_metrics = true
    int totalIterations = 10;
    int earlyStoppingRounds = 5;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    int bestIteration = 0;

    boolean es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    TestCase.assertTrue(es);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    bestIteration = totalIterations - 1;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    TestCase.assertFalse(es);

    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    metrics[0][4] = 9;
    metrics[0][9] = 4;

    bestIteration = 4;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    TestCase.assertTrue(es);
  }

  @Test
  public void testBoosterEarlyStop() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "false");
      }
    };
    Map<String, DMatrix> watches = new LinkedHashMap<>();
    watches.put("training", trainMat);
    watches.put("test", testMat);

    final int round = 10;
    int earlyStoppingRound = 2;
    float[][] metrics = new float[watches.size()][round];
    XGBoost.train(trainMat, paramMap, round, watches, metrics, null, new IncreasingEval(),
            earlyStoppingRound);

    // Make sure we've stopped early.
    for (int w = 0; w < watches.size(); w++) {
      for (int r = 0; r <= earlyStoppingRound; r++) {
        TestCase.assertFalse(0.0f == metrics[w][r]);
      }
    }

    for (int w = 0; w < watches.size(); w++) {
      for (int r = earlyStoppingRound + 1; r < round; r++) {
        TestCase.assertEquals(0.0f, metrics[w][r]);
      }
    }
  }

  @Test
  public void testEarlyStoppingAttributes() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "false");
      }
    };
    Map<String, DMatrix> watches = new LinkedHashMap<>();
    watches.put("training", trainMat);
    watches.put("test", testMat);

    int round = 30;
    int earlyStoppingRound = 4;
    float[][] metrics = new float[watches.size()][round];

    Booster booster = XGBoost.train(trainMat, paramMap, round,
				    watches, metrics, null, null, earlyStoppingRound);

    int bestIter = Integer.valueOf(booster.getAttr("best_iteration"));
    float bestScore = Float.valueOf(booster.getAttr("best_score"));
    TestCase.assertEquals(bestIter, round - 1);
    TestCase.assertEquals(bestScore, metrics[watches.size() - 1][round - 1]);
  }

  private void testWithQuantileHisto(DMatrix trainingSet, Map<String, DMatrix> watches, int round,
                                      Map<String, Object> paramMap, float threshold) throws XGBoostError {
    float[][] metrics = new float[watches.size()][round];
    Booster booster = XGBoost.train(trainingSet, paramMap, round, watches,
            metrics, null, null, 0);
    for (int i = 0; i < metrics.length; i++)
      for (int j = 1; j < metrics[i].length; j++) {
        TestCase.assertTrue(metrics[i][j] >= metrics[i][j - 1] ||
                Math.abs(metrics[i][j] - metrics[i][j - 1]) < 0.1);
      }
    for (int i = 0; i < metrics.length; i++)
      for (int j = 0; j < metrics[i].length; j++) {
        TestCase.assertTrue(metrics[i][j] >= threshold);
      }
    booster.dispose();
  }

  @Test
  public void testQuantileHistoDepthWise() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("tree_method", "hist");
        put("grow_policy", "depthwise");
        put("eval_metric", "auc");
      }
    };
    Map<String, DMatrix> watches = new HashMap<>();
    watches.put("training", trainMat);
    watches.put("test", testMat);
    testWithQuantileHisto(trainMat, watches, 10, paramMap, 0.95f);
  }

  @Test
  public void testQuantileHistoLossGuide() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("tree_method", "hist");
        put("grow_policy", "lossguide");
        put("max_leaves", 8);
        put("eval_metric", "auc");
      }
    };
    Map<String, DMatrix> watches = new HashMap<>();
    watches.put("training", trainMat);
    watches.put("test", testMat);
    testWithQuantileHisto(trainMat, watches, 10, paramMap, 0.95f);
  }

  @Test
  public void testQuantileHistoLossGuideMaxBin() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("tree_method", "hist");
        put("grow_policy", "lossguide");
        put("max_leaves", 8);
        put("max_bin", 16);
        put("eval_metric", "auc");
      }
    };
    Map<String, DMatrix> watches = new HashMap<>();
    watches.put("training", trainMat);
    testWithQuantileHisto(trainMat, watches, 10, paramMap, 0.95f);
  }

  @Test
  public void testDumpModelJson() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    String[] dump = booster.getModelDump("", false, "json");
    TestCase.assertEquals("  { \"nodeid\":", dump[0].substring(0, 13));

    // test with specified feature names
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    dump = booster.getModelDump(featureNames, false, "json");
    TestCase.assertTrue(dump[0].contains("test_feature_name_"));
  }

  @Test
  public void testGetFeatureScore() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Integer> scoreMap = booster.getFeatureScore(featureNames);
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceGain() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "gain");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceTotalGain() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "total_gain");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceCover() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "cover");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceTotalCover() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "total_cover");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testQuantileHistoDepthwiseMaxDepth() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("tree_method", "hist");
        put("grow_policy", "depthwise");
        put("eval_metric", "auc");
      }
    };
    Map<String, DMatrix> watches = new HashMap<>();
    watches.put("training", trainMat);
    testWithQuantileHisto(trainMat, watches, 10, paramMap, 0.95f);
  }

  @Test
  public void testQuantileHistoDepthwiseMaxDepthMaxBin() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("tree_method", "hist");
        put("max_bin", 2);
        put("grow_policy", "depthwise");
        put("eval_metric", "auc");
      }
    };
    Map<String, DMatrix> watches = new HashMap<>();
    watches.put("training", trainMat);
    testWithQuantileHisto(trainMat, watches, 10, paramMap, 0.95f);
  }

  /**
   * test cross valiation
   *
   * @throws XGBoostError
   */
  @Test
  public void testCV() throws XGBoostError {
    //load train mat
    DMatrix trainMat = new DMatrix(this.train_uri);

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
    String[] evalHist = XGBoost.crossValidation(trainMat, param, round, nfold, null, null, null);
  }

  /**
   * test train from existing model
   *
   * @throws XGBoostError
   */
  @Test
  public void testTrainFromExistingModel() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    IEvaluation eval = new EvalError();

    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("eta", 1.0);
        put("max_depth", 2);
        put("silent", 1);
        put("objective", "binary:logistic");
      }
    };

    //set watchList
    HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();

    watches.put("train", trainMat);
    watches.put("test", testMat);

    // Train without saving temp booster
    int round = 4;
    Booster booster1 = XGBoost.train(trainMat, paramMap, round, watches, null, null, null, 0);
    float booster1error = eval.eval(booster1.predict(testMat, true, 0), testMat);

    // Train with temp Booster
    round = 2;
    Booster tempBooster = XGBoost.train(trainMat, paramMap, round, watches, null, null, null, 0);
    float tempBoosterError = eval.eval(tempBooster.predict(testMat, true, 0), testMat);

    // Save tempBooster to bytestream and load back
    int prevVersion = tempBooster.getVersion();
    ByteArrayInputStream in = new ByteArrayInputStream(tempBooster.toByteArray());
    tempBooster = XGBoost.loadModel(in);
    in.close();
    tempBooster.setVersion(prevVersion);

    // Continue training using tempBooster
    round = 4;
    Booster booster2 = XGBoost.train(trainMat, paramMap, round, watches, null, null, null, 0, tempBooster);
    float booster2error = eval.eval(booster2.predict(testMat, true, 0), testMat);
    TestCase.assertTrue(booster1error == booster2error);
    TestCase.assertTrue(tempBoosterError > booster2error);
  }

  /**
   * test set/get attributes to/from a booster
   *
   * @throws XGBoostError
   */
  @Test
  public void testSetAndGetAttrs() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    booster.setAttr("testKey1", "testValue1");
    TestCase.assertEquals(booster.getAttr("testKey1"), "testValue1");
    booster.setAttr("testKey1", "testValue2");
    TestCase.assertEquals(booster.getAttr("testKey1"), "testValue2");

    booster.setAttrs(new HashMap<String, String>(){{
      put("aa", "AA");
      put("bb", "BB");
      put("cc", "CC");
    }});

    Map<String, String> attr = booster.getAttrs();
    TestCase.assertEquals(attr.size(), 6);
    TestCase.assertEquals(attr.get("testKey1"), "testValue2");
    TestCase.assertEquals(attr.get("aa"), "AA");
    TestCase.assertEquals(attr.get("bb"), "BB");
    TestCase.assertEquals(attr.get("cc"), "CC");
  }

  /**
   * test get number of features from a booster
   *
   * @throws XGBoostError
   */
  @Test
  public void testGetNumFeature() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Booster booster = trainBooster(trainMat, testMat);
    TestCase.assertEquals(booster.getNumFeature(), 126);
  }
}
