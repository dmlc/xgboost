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
package ml.dmlc.xgboost4j.java;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
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

  /*
  @Test
  public void testBoosterBasic() throws XGBoostError, IOException {

    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

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
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
    IEvaluation eval = new EvalError();

    Booster booster = trainBooster(trainMat, testMat);
    // save and load
    File temp = File.createTempFile("temp", "model");
    temp.deleteOnExit();
    booster.saveModel(temp.getAbsolutePath());

    Booster bst2 = XGBoost.loadModel(temp.getAbsolutePath());
    assert (Arrays.equals(bst2.toByteArray(), booster.toByteArray()));
    float[][] predicts2 = bst2.predict(testMat, true, 0);
    TestCase.assertTrue(eval.eval(predicts2, testMat) < 0.1f);
  }

  @Test
  public void saveLoadModelWithStream() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

    Booster booster = trainBooster(trainMat, testMat);

    Path tempDir = Files.createTempDirectory("boosterTest-");
    File tempFile = Files.createTempFile("", "").toFile();
    booster.saveModel(new FileOutputStream(tempFile));
    IEvaluation eval = new EvalError();
    Booster loadedBooster = XGBoost.loadModel(new FileInputStream(tempFile));
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
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "false");
      }
    };
    int totalIterations = 10;
    int earlyStoppingRounds = 10;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    for (int itr = 0; itr < totalIterations; itr++) {
      boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics,
              itr);
      if (itr == totalIterations - 1) {
        TestCase.assertFalse(onTrack);
        for (int i = 0; i < totalIterations; i++) {
          metrics[0][i] = totalIterations - i;
        }
        onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics,
                totalIterations - 1);
        TestCase.assertTrue(onTrack);
      } else {
        TestCase.assertTrue(onTrack);
      }
    }
  }

  @Test
  public void testEarlyStoppingForMultipleMetrics() {
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "true");
      }
    };
    int earlyStoppingRound = 3;
    int totalIterations = 5;
    int numOfMetrics = 3;
    float[][] metrics = new float[numOfMetrics][totalIterations];
    for (int i = 0; i < numOfMetrics; i++) {
      for (int j = 0; j < totalIterations; j++) {
        metrics[0][j] = j;
      }
    }
    for (int i = 0; i < totalIterations; i++) {
      boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRound, metrics, i);
      TestCase.assertTrue(onTrack);
    }
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    // when we have multiple datasets, the training metrics is not considered
    for (int i = 0; i < totalIterations; i++) {
      boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRound, metrics, i);
      TestCase.assertTrue(onTrack);
    }
    for (int i = 0; i < totalIterations; i++) {
      metrics[1][i] = totalIterations - i;
    }
    for (int i = 0; i < totalIterations; i++) {
      // if any metrics off, we need to stop
      boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRound, metrics, i);
      if (i >= earlyStoppingRound - 1) {
        TestCase.assertFalse(onTrack);
      } else {
        TestCase.assertTrue(onTrack);
      }
    }
  }

  @Test
  public void testDescendMetrics() {
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "false");
      }
    };
    int totalIterations = 10;
    int earlyStoppingRounds = 5;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics,
            totalIterations - 1);
    TestCase.assertFalse(onTrack);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics,
            totalIterations - 1);
    TestCase.assertTrue(onTrack);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    metrics[0][5] = 1;
    metrics[0][6] = 2;
    metrics[0][7] = 3;
    metrics[0][8] = 4;
    metrics[0][9] = 1;
    onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics,
            totalIterations - 1);
    TestCase.assertTrue(onTrack);
  }

  @Test
  public void testAscendMetricsWithBoundaryCondition() {
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "true");
      }
    };
    int totalIterations = 10;
    int earlyStoppingRounds = 10;
    float[][] metrics = new float[1][totalIterations];
    for (int iter = 0; iter < totalIterations; iter++) {
      if (iter == totalIterations - 1) {
        for (int i = 0; i < totalIterations; i++) {
          metrics[0][i] = i;
        }
        boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics, iter);
        TestCase.assertTrue(onTrack);
        for (int i = 0; i < totalIterations; i++) {
          metrics[0][i] = totalIterations - i;
        }
        onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics, iter);
        TestCase.assertFalse(onTrack);
      } else {
        for (int i = 0; i < totalIterations; i++) {
          metrics[0][i] = totalIterations - i;
        }
        boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics, iter);
        TestCase.assertTrue(onTrack);
      }
    }
  }

  @Test
  public void testAscendMetrics() {
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 3);
        put("silent", 1);
        put("objective", "binary:logistic");
        put("maximize_evaluation_metrics", "true");
      }
    };
    int totalIterations = 10;
    int earlyStoppingRounds = 5;
    float[][] metrics = new float[1][totalIterations];
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    boolean onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics, totalIterations - 1);
    TestCase.assertTrue(onTrack);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = totalIterations - i;
    }
    onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics, totalIterations - 1);
    TestCase.assertFalse(onTrack);
    for (int i = 0; i < totalIterations; i++) {
      metrics[0][i] = i;
    }
    metrics[0][5] = 9;
    metrics[0][6] = 8;
    metrics[0][7] = 7;
    metrics[0][8] = 6;
    metrics[0][9] = 9;
    onTrack = XGBoost.judgeIfTrainingOnTrack(paramMap, earlyStoppingRounds, metrics, totalIterations - 1);
    TestCase.assertTrue(onTrack);
  }

  @Test
  public void testBoosterEarlyStop() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
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
      for (int r = 0; r < earlyStoppingRound; r++) {
        TestCase.assertFalse(0.0f == metrics[w][r]);
      }
    }

    for (int w = 0; w < watches.size(); w++) {
      for (int r = earlyStoppingRound; r < round; r++) {
        TestCase.assertEquals(0.0f, metrics[w][r]);
      }
    }
  }*/

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
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
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

/*
  @Test
  public void testQuantileHistoLossGuide() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 0);
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
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 0);
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
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

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
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Integer> scoreMap = booster.getFeatureScore(featureNames);
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceGain() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "gain");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceTotalGain() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "total_gain");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceCover() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "cover");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }

  @Test
  public void testGetFeatureImportanceTotalCover() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

    Booster booster = trainBooster(trainMat, testMat);
    String[] featureNames = new String[126];
    for(int i = 0; i < 126; i++) featureNames[i] = "test_feature_name_" + i;
    Map<String, Double> scoreMap = booster.getScore(featureNames, "total_cover");
    for (String fName: scoreMap.keySet()) TestCase.assertTrue(fName.startsWith("test_feature_name_"));
  }*/

/*
  @Test
  public void testQuantileHistoDepthwiseMaxDepth() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
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

  /*
  @Test
  public void testQuantileHistoDepthwiseMaxDepthMaxBin() throws XGBoostError {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
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
  /*
  @Test
  public void testCV() throws XGBoostError {
    //load train mat
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");

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
  }*/

  /**
   * test train from existing model
   *
   * @throws XGBoostError
   */
  /*
  @Test
  public void testTrainFromExistingModel() throws XGBoostError, IOException {
    DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
    DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");
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
  }*/
}
