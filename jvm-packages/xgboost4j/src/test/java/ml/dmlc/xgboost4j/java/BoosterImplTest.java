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
import java.util.*;
import java.util.stream.Collectors;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * test cases for Booster
 *
 * @author hzx
 */
public class BoosterImplTest {
  private final String train_uri = "../../demo/data/agaricus.txt.train?indexing_mode=1&format=libsvm";
  private final String test_uri = "../../demo/data/agaricus.txt.test?indexing_mode=1&format=libsvm";

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
  public void testBoosterBasic() throws XGBoostError {

    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    try (Booster booster = trainBooster(trainMat, testMat)) {

      //predict raw output
      float[][] predicts = booster.predict(testMat, true, 0);

      //eval
      IEvaluation eval = new EvalError();
      //error must be less than 0.1
      assertTrue(eval.eval(predicts, testMat) < 0.1f);
    }
  }

  @Test
  public void saveLoadModelWithPath() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    IEvaluation eval = new EvalError();

    try (Booster booster = trainBooster(trainMat, testMat)) {
      // save and load
      File temp = File.createTempFile("temp", "model");
      temp.deleteOnExit();
      booster.saveModel(temp.getAbsolutePath());

      try (Booster bst2 = XGBoost.loadModel(temp.getAbsolutePath())) {
        for (String format: Arrays.asList("ubj", "json", "deprecated")) {
          assertArrayEquals(bst2.toByteArray(format), booster.toByteArray(format));
        }
        float[][] predicts2 = bst2.predict(testMat, true, 0);
        assertTrue(eval.eval(predicts2, testMat) < 0.1f);
      }
    }
  }

  @Test
  public void saveLoadModelWithFeaturesWithPath() throws Exception {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);
    IEvaluation eval = new EvalError();

    String[] featureNames = new String[126];
    String[] featureTypes = new String[126];
    for(int i = 0; i < 126; i++) {
      featureNames[i] = "test_feature_name_" + i;
      featureTypes[i] = "q";
    }
    trainMat.setFeatureNames(featureNames);
    testMat.setFeatureNames(featureNames);
    trainMat.setFeatureTypes(featureTypes);
    testMat.setFeatureTypes(featureTypes);

    try(Booster booster = trainBooster(trainMat, testMat)) {
      // save and load, only json format save and load feature_name and feature_type
      File temp = File.createTempFile("temp", ".json");
      temp.deleteOnExit();
      booster.saveModel(temp.getAbsolutePath());

      try (Booster bst2 = XGBoost.loadModel(temp.getAbsolutePath())) {
        for (String format: Arrays.asList("ubj", "json", "deprecated")) {
          assertArrayEquals(bst2.toByteArray(format), booster.toByteArray(format));
        }
        float[][] predicts2 = bst2.predict(testMat, true, 0);
        assertTrue(eval.eval(predicts2, testMat) < 0.1f);
      }
    }
  }

  @Test
  public void saveLoadModelWithStream() throws Exception {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    try (Booster booster = trainBooster(trainMat, testMat)) {

      ByteArrayOutputStream output = new ByteArrayOutputStream();
      booster.saveModel(output);
      IEvaluation eval = new EvalError();
      try (Booster loadedBooster = XGBoost.loadModel(new ByteArrayInputStream(output.toByteArray()))) {
        float originalPredictError = eval.eval(booster.predict(testMat, true), testMat);
        assertTrue("originalPredictErr:" + originalPredictError,
          originalPredictError < 0.1f);
        float loadedPredictError = eval.eval(loadedBooster.predict(testMat, true), testMat);
        assertTrue("loadedPredictErr:" + loadedPredictError, loadedPredictError < 0.1f);
      }
    }
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
        assertTrue(es);
      } else {
        assertFalse(es);
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
      assertFalse(es);
    }

    // when we have multiple datasets, only the last one was used to determinate early stop
    // Here we changed the metric of the first dataset, it doesn't have any effect to the final result
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    for (int i = 0; i < totalIterations; i++) {
      bestIteration = i;
      boolean es = XGBoost.shouldEarlyStop(earlyStoppingRound, i, bestIteration);
      assertFalse(es);
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
        assertTrue(es);
      } else {
        assertFalse(es);
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
    assertTrue(es);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    bestIteration = totalIterations - 1;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    assertFalse(es);

    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    metrics[0][4] = 1;
    metrics[0][9] = 5;

    bestIteration = 4;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    assertTrue(es);
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
        assertTrue(es);
      } else {
        assertFalse(es);
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
    assertTrue(es);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    bestIteration = totalIterations - 1;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    assertFalse(es);

    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    metrics[0][4] = 9;
    metrics[0][9] = 4;

    bestIteration = 4;

    es = XGBoost.shouldEarlyStop(earlyStoppingRounds, totalIterations - 1, bestIteration);
    assertTrue(es);
  }

  @Test
  public void testBoosterEarlyStop() throws XGBoostError {
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
    try (Booster b = XGBoost.train(trainMat, paramMap, round, watches, metrics, null, new IncreasingEval(),
      earlyStoppingRound)) {
      // Not needed.
    }

    // Make sure we've stopped early.
    for (int w = 0; w < watches.size(); w++) {
      for (int r = 0; r <= earlyStoppingRound; r++) {
        assertNotEquals(0.0f, metrics[w][r], 0.0);
      }
    }

    for (int w = 0; w < watches.size(); w++) {
      for (int r = earlyStoppingRound + 1; r < round; r++) {
        assertEquals(0.0f, metrics[w][r], 0.0);
      }
    }
  }

  @Test
  public void testEarlyStoppingAttributes() throws XGBoostError {
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

    try (Booster booster = XGBoost.train(trainMat, paramMap, round,
				    watches, metrics, null, null, earlyStoppingRound)) {
      int bestIter = Integer.parseInt(booster.getAttr("best_iteration"));
      float bestScore = Float.parseFloat(booster.getAttr("best_score"));
      assertEquals(bestIter, round - 1);
      assertEquals(bestScore, metrics[watches.size() - 1][round - 1], 0.0);
    }
  }

  private void testWithQuantileHisto(DMatrix trainingSet, Map<String, DMatrix> watches, int round,
                                      Map<String, Object> paramMap, float threshold) throws XGBoostError {
    float[][] metrics = new float[watches.size()][round];
    try(Booster booster = XGBoost.train(trainingSet, paramMap, round, watches,
            metrics, null, null, 0)) {
      for (float[] metric : metrics)
        for (int j = 1; j < metric.length; j++) {
          assertTrue(metric[j] >= metric[j - 1] ||
            Math.abs(metric[j] - metric[j - 1]) < 0.1);
        }
      for (float[] metric : metrics) {
        for (float v : metric) {
          assertTrue(v >= threshold);
        }
      }
    }
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

    String[] dump;
    try (Booster booster = trainBooster(trainMat, testMat)) {
      dump = booster.getModelDump("", false, "json");
      assertEquals("  { \"nodeid\":", dump[0].substring(0, 13));

      // test with specified feature names
      String[] featureNames = new String[126];
      for (int i = 0; i < 126; i++) {
        featureNames[i] = "test_feature_name_" + i;
      }
      dump = booster.getModelDump(featureNames, false, "json");
    }
    assertTrue(dump[0].contains("test_feature_name_"));
  }

  @Test
  public void testGetFeatureScore() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    Map<String, Integer> scoreMap;
    String[] featureNames = new String[126];
    try (Booster booster = trainBooster(trainMat, testMat)) {
      for (int i = 0; i < 126; i++) {
        featureNames[i] = "test_feature_name_" + i;
      }
      scoreMap = booster.getFeatureScore(featureNames);
    }

    for (String fName: scoreMap.keySet()) {
      assertTrue(fName.startsWith("test_feature_name_"));
      assertTrue(Arrays.stream(featureNames).anyMatch(v -> v.equalsIgnoreCase(fName)));
    }
  }

  @Test
  public void testGetFeatureImportance() throws XGBoostError {
    DMatrix trainMat = new DMatrix(this.train_uri);
    DMatrix testMat = new DMatrix(this.test_uri);

    try (Booster booster = trainBooster(trainMat, testMat)) {
      String[] featureNames = new String[126];
      for (int i = 0; i < 126; i++) {
        featureNames[i] = "test_feature_name_" + i;
      }
      for (String importanceType: Arrays.asList("gain", "total_gain", "cover", "total_cover")) {
        Map<String, Double> scoreMap = booster.getScore(featureNames, importanceType);
        for (String fName: scoreMap.keySet()) {
          assertTrue(fName.startsWith("test_feature_name_"));
          assertTrue(Arrays.stream(featureNames).anyMatch(v -> v.equalsIgnoreCase(fName)));
        }
      }
    }
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
    assertEquals(booster1error, booster2error, 0.0);
    assertTrue(tempBoosterError > booster2error);
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

    Map<String, String> attr;
    try (Booster booster = trainBooster(trainMat, testMat)) {
      booster.setAttr("testKey1", "testValue1");
      assertEquals(booster.getAttr("testKey1"), "testValue1");
      booster.setAttr("testKey1", "testValue2");
      assertEquals(booster.getAttr("testKey1"), "testValue2");

      booster.setAttrs(new HashMap<String, String>() {{
        put("aa", "AA");
        put("bb", "BB");
        put("cc", "CC");
      }});

      attr = booster.getAttrs();
    }
    assertEquals(attr.size(), 6);
    assertEquals(attr.get("testKey1"), "testValue2");
    assertEquals(attr.get("aa"), "AA");
    assertEquals(attr.get("bb"), "BB");
    assertEquals(attr.get("cc"), "CC");
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

    try (Booster booster = trainBooster(trainMat, testMat)) {
      assertEquals(booster.getNumFeature(), 126);
    }
  }
}
