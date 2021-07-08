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
import java.util.*;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * trainer for xgboost
 *
 * @author hzx
 */
public class XGBoost {
  private static final Log logger = LogFactory.getLog(XGBoost.class);

  /**
   * load model from modelPath
   *
   * @param modelPath booster modelPath (model generated by booster.saveModel)
   * @throws XGBoostError native error
   */
  public static Booster loadModel(String modelPath)
          throws XGBoostError {
    return Booster.loadModel(modelPath);
  }

  /**
   * Load a new Booster model from a file opened as input stream.
   * The assumption is the input stream only contains one XGBoost Model.
   * This can be used to load existing booster models saved by other xgboost bindings.
   *
   * @param in The input stream of the file,
   *           will be closed after this function call.
   * @return The create boosted
   * @throws XGBoostError
   * @throws IOException
   */
  public static Booster loadModel(InputStream in) throws XGBoostError, IOException {
    int size;
    byte[] buf = new byte[1<<20];
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    while ((size = in.read(buf)) != -1) {
      os.write(buf, 0, size);
    }
    in.close();
    return Booster.loadModel(buf);
  }

  /**
   * Load a new Booster model from a byte array buffer.
   * The assumption is the array only contains one XGBoost Model.
   * This can be used to load existing booster models saved by other xgboost bindings.
   *
   * @param buffer The byte contents of the booster.
   * @return The create boosted
   * @throws XGBoostError
   */
  public static Booster loadModel(byte[] buffer) throws XGBoostError, IOException {
    return Booster.loadModel(buffer);
  }

  /**
   * Train a booster given parameters.
   *
   * @param dtrain  Data to be trained.
   * @param params  Parameters.
   * @param round   Number of boosting iterations.
   * @param watches a group of items to be evaluated during training, this allows user to watch
   *                performance on the validation set.
   * @param obj     customized objective
   * @param eval    customized evaluation
   * @return The trained booster.
   */
  public static Booster train(
          DMatrix dtrain,
          Map<String, Object> params,
          int round,
          Map<String, DMatrix> watches,
          IObjective obj,
          IEvaluation eval) throws XGBoostError {
    return train(dtrain, params, round, watches, null, obj, eval, 0);
  }

  /**
   * Train a booster given parameters.
   *
   * @param dtrain  Data to be trained.
   * @param params  Parameters.
   * @param round   Number of boosting iterations.
   * @param watches a group of items to be evaluated during training, this allows user to watch
   *                performance on the validation set.
   * @param metrics array containing the evaluation metrics for each matrix in watches for each
   *                iteration
   * @param earlyStoppingRound if non-zero, training would be stopped
   *                           after a specified number of consecutive
   *                           increases in any evaluation metric.
   * @param obj     customized objective
   * @param eval    customized evaluation
   * @return The trained booster.
   */
  public static Booster train(
          DMatrix dtrain,
          Map<String, Object> params,
          int round,
          Map<String, DMatrix> watches,
          float[][] metrics,
          IObjective obj,
          IEvaluation eval,
          int earlyStoppingRound) throws XGBoostError {
    return train(dtrain, params, round, watches, metrics, obj, eval, earlyStoppingRound, null);
  }

  private static void saveCheckpoint(
          Booster booster,
          int iter,
          Set<Integer> checkpointIterations,
          ExternalCheckpointManager ecm) throws XGBoostError {
    try {
      if (checkpointIterations.contains(iter)) {
        ecm.updateCheckpoint(booster);
      }
    } catch (Exception e) {
      logger.error("failed to save checkpoint in XGBoost4J at iteration " + iter, e);
      throw new XGBoostError("failed to save checkpoint in XGBoost4J at iteration" + iter, e);
    }
  }

  public static Booster trainAndSaveCheckpoint(
      DMatrix dtrain,
      Map<String, Object> params,
      int numRounds,
      Map<String, DMatrix> watches,
      float[][] metrics,
      IObjective obj,
      IEvaluation eval,
      int earlyStoppingRounds,
      Booster booster,
      int checkpointInterval,
      String checkpointPath,
      FileSystem fs) throws XGBoostError, IOException {
    //collect eval matrixs
    String[] evalNames;
    DMatrix[] evalMats;
    float bestScore;
    int bestIteration;
    List<String> names = new ArrayList<String>();
    List<DMatrix> mats = new ArrayList<DMatrix>();
    Set<Integer> checkpointIterations = new HashSet<>();
    ExternalCheckpointManager ecm = null;
    if (checkpointPath != null) {
      ecm = new ExternalCheckpointManager(checkpointPath, fs);
    }

    for (Map.Entry<String, DMatrix> evalEntry : watches.entrySet()) {
      names.add(evalEntry.getKey());
      mats.add(evalEntry.getValue());
    }

    evalNames = names.toArray(new String[names.size()]);
    evalMats = mats.toArray(new DMatrix[mats.size()]);
    if (isMaximizeEvaluation(params)) {
      bestScore = -Float.MAX_VALUE;
    } else {
      bestScore = Float.MAX_VALUE;
    }
    bestIteration = 0;
    metrics = metrics == null ? new float[evalNames.length][numRounds] : metrics;

    //collect all data matrixs
    DMatrix[] allMats;
    if (evalMats.length > 0) {
      allMats = new DMatrix[evalMats.length + 1];
      allMats[0] = dtrain;
      System.arraycopy(evalMats, 0, allMats, 1, evalMats.length);
    } else {
      allMats = new DMatrix[1];
      allMats[0] = dtrain;
    }

    //initialize booster
    if (booster == null) {
      // Start training on a new booster
      booster = new Booster(params, allMats);
      booster.loadRabitCheckpoint();
    } else {
      // Start training on an existing booster
      booster.setParams(params);
    }

    if (ecm != null) {
      checkpointIterations = new HashSet<>(ecm.getCheckpointRounds(checkpointInterval, numRounds));
    }

    // begin to train
    for (int iter = booster.getVersion() / 2; iter < numRounds; iter++) {
      if (booster.getVersion() % 2 == 0) {
        if (obj != null) {
          booster.update(dtrain, obj);
        } else {
          booster.update(dtrain, iter);
        }
        saveCheckpoint(booster, iter, checkpointIterations, ecm);
        booster.saveRabitCheckpoint();
      }

      //evaluation
      if (evalMats.length > 0) {
        float[] metricsOut = new float[evalMats.length];
        String evalInfo;
        if (eval != null) {
          evalInfo = booster.evalSet(evalMats, evalNames, eval, metricsOut);
        } else {
          evalInfo = booster.evalSet(evalMats, evalNames, iter, metricsOut);
        }
        for (int i = 0; i < metricsOut.length; i++) {
          metrics[i][iter] = metricsOut[i];
        }

        // If there is more than one evaluation datasets, the last one would be used
        // to determinate early stop.
        float score = metricsOut[metricsOut.length - 1];
        if (isMaximizeEvaluation(params)) {
          // Update best score if the current score is better (no update when equal)
          if (score > bestScore) {
            bestScore = score;
            bestIteration = iter;
            booster.setAttr("bestIteration", String.valueOf(bestIteration));
            booster.setAttr("bestScore", String.valueOf(bestScore));
          }
        } else {
          if (score < bestScore) {
            bestScore = score;
            bestIteration = iter;
            booster.setAttr("bestIteration", String.valueOf(bestIteration));
            booster.setAttr("bestScore", String.valueOf(bestScore));
          }
        }
        if (earlyStoppingRounds > 0) {
          if (shouldEarlyStop(earlyStoppingRounds, iter, bestIteration)) {
            Rabit.trackerPrint(String.format(
                    "early stopping after %d rounds away from the best iteration",
                    earlyStoppingRounds));
            break;
          }
        }
        if (Rabit.getRank() == 0 && shouldPrint(params, iter)) {
          if (shouldPrint(params, iter)){
            Rabit.trackerPrint(evalInfo + '\n');
          }
        }
      }
      booster.saveRabitCheckpoint();
    }
    return booster;
  }

  /**
   * Train a booster given parameters.
   *
   * @param dtrain  Data to be trained.
   * @param params  Parameters.
   * @param round   Number of boosting iterations.
   * @param watches a group of items to be evaluated during training, this allows user to watch
   *                performance on the validation set.
   * @param metrics array containing the evaluation metrics for each matrix in watches for each
   *                iteration
   * @param earlyStoppingRounds if non-zero, training would be stopped
   *                           after a specified number of consecutive
   *                           goes to the unexpected direction in any evaluation metric.
   * @param obj     customized objective
   * @param eval    customized evaluation
   * @param booster train from scratch if set to null; train from an existing booster if not null.
   * @return The trained booster.
   */
  public static Booster train(
          DMatrix dtrain,
          Map<String, Object> params,
          int round,
          Map<String, DMatrix> watches,
          float[][] metrics,
          IObjective obj,
          IEvaluation eval,
          int earlyStoppingRounds,
          Booster booster) throws XGBoostError {
    try {
      return trainAndSaveCheckpoint(dtrain, params, round, watches, metrics, obj, eval,
              earlyStoppingRounds, booster,
              -1, null, null);
    } catch (IOException e) {
      logger.error("training failed in xgboost4j", e);
      throw new XGBoostError("training failed in xgboost4j ", e);
    }
  }

  private static Integer tryGetIntFromObject(Object o) {
    if (o instanceof Integer) {
      return (int)o;
    } else if (o instanceof String) {
      try {
        return Integer.parseInt((String)o);
      } catch (NumberFormatException e) {
        return null;
      }
    } else {
      return null;
    }
  }

  private static boolean shouldPrint(Map<String, Object> params, int iter) {
    Object silent = params.get("silent");
    Integer silentInt = tryGetIntFromObject(silent);
    if (silent != null) {
      if (silent.equals("true") || silent.equals("True")
              || (silentInt != null && silentInt != 0)) {
        return false;  // "silent" will stop printing, otherwise go look at "verbose_eval"
      }
    }

    Object verboseEval = params.get("verbose_eval");
    Integer verboseEvalInt = tryGetIntFromObject(verboseEval);
    if (verboseEval == null) {
      return true; // Default to printing evalInfo
    } else if (verboseEval.equals("false") || verboseEval.equals("False")) {
      return false;
    } else if (verboseEvalInt != null) {
      if (verboseEvalInt == 0) {
        return false;
      } else {
        return iter % verboseEvalInt == 0;
      }
    } else {
      return true; // Don't understand the option, default to printing
    }
  }

  static boolean shouldEarlyStop(int earlyStoppingRounds, int iter, int bestIteration) {
    return iter - bestIteration >= earlyStoppingRounds;
  }

  private static boolean isMaximizeEvaluation(Map<String, Object> params) {
    try {
      String maximize = String.valueOf(params.get("maximize_evaluation_metrics"));
      assert(maximize != null);
      return Boolean.valueOf(maximize);
    } catch (Exception ex) {
      logger.error("maximize_evaluation_metrics has to be specified for enabling early stop," +
              " allowed value: true/false", ex);
      throw ex;
    }
  }

  /**
   * Cross-validation with given parameters.
   *
   * @param data    Data to be trained.
   * @param params  Booster params.
   * @param round   Number of boosting iterations.
   * @param nfold   Number of folds in CV.
   * @param metrics Evaluation metrics to be watched in CV.
   * @param obj     customized objective (set to null if not used)
   * @param eval    customized evaluation (set to null if not used)
   * @return evaluation history
   * @throws XGBoostError native error
   */
  public static String[] crossValidation(
      DMatrix data,
      Map<String, Object> params,
      int round,
      int nfold,
      String[] metrics,
      IObjective obj,
      IEvaluation eval) throws XGBoostError {
    CVPack[] cvPacks = makeNFold(data, nfold, params, metrics);
    String[] evalHist = new String[round];
    String[] results = new String[cvPacks.length];
    for (int i = 0; i < round; i++) {
      for (CVPack cvPack : cvPacks) {
        if (obj != null) {
          cvPack.update(obj);
        } else {
          cvPack.update(i);
        }
      }

      for (int j = 0; j < cvPacks.length; j++) {
        if (eval != null) {
          results[j] = cvPacks[j].eval(eval);
        } else {
          results[j] = cvPacks[j].eval(i);
        }
      }

      evalHist[i] = aggCVResults(results);
      logger.info(evalHist[i]);
    }
    return evalHist;
  }

  /**
   * make an n-fold array of CVPack from random indices
   *
   * @param data        original data
   * @param nfold       num of folds
   * @param params      booster parameters
   * @param evalMetrics Evaluation metrics
   * @return CV package array
   * @throws XGBoostError native error
   */
  private static CVPack[] makeNFold(DMatrix data, int nfold, Map<String, Object> params,
                                    String[] evalMetrics) throws XGBoostError {
    List<Integer> samples = genRandPermutationNums(0, (int) data.rowNum());
    int step = samples.size() / nfold;
    int[] testSlice = new int[step];
    int[] trainSlice = new int[samples.size() - step];
    int testid, trainid;
    CVPack[] cvPacks = new CVPack[nfold];
    for (int i = 0; i < nfold; i++) {
      testid = 0;
      trainid = 0;
      for (int j = 0; j < samples.size(); j++) {
        if (j > (i * step) && j < (i * step + step) && testid < step) {
          testSlice[testid] = samples.get(j);
          testid++;
        } else {
          if (trainid < samples.size() - step) {
            trainSlice[trainid] = samples.get(j);
            trainid++;
          } else {
            testSlice[testid] = samples.get(j);
            testid++;
          }
        }
      }

      DMatrix dtrain = data.slice(trainSlice);
      DMatrix dtest = data.slice(testSlice);
      CVPack cvPack = new CVPack(dtrain, dtest, params);
      //set eval types
      if (evalMetrics != null) {
        for (String type : evalMetrics) {
          cvPack.booster.setParam("eval_metric", type);
        }
      }
      cvPacks[i] = cvPack;
    }

    return cvPacks;
  }

  private static List<Integer> genRandPermutationNums(int start, int end) {
    List<Integer> samples = new ArrayList<Integer>();
    for (int i = start; i < end; i++) {
      samples.add(i);
    }
    Collections.shuffle(samples);
    return samples;
  }

  /**
   * Aggregate cross-validation results.
   *
   * @param results eval info from each data sample
   * @return cross-validation eval info
   */
  private static String aggCVResults(String[] results) {
    Map<String, List<Float>> cvMap = new HashMap<String, List<Float>>();
    String aggResult = results[0].split("\t")[0];
    for (String result : results) {
      String[] items = result.split("\t");
      for (int i = 1; i < items.length; i++) {
        String[] tup = items[i].split(":");
        String key = tup[0];
        Float value = Float.valueOf(tup[1]);
        if (!cvMap.containsKey(key)) {
          cvMap.put(key, new ArrayList<Float>());
        }
        cvMap.get(key).add(value);
      }
    }

    for (String key : cvMap.keySet()) {
      float value = 0f;
      for (Float tvalue : cvMap.get(key)) {
        value += tvalue;
      }
      value /= cvMap.get(key).size();
      aggResult += String.format("\tcv-%s:%f", key, value);
    }

    return aggResult;
  }

  /**
   * cross validation package for xgb
   *
   * @author hzx
   */
  private static class CVPack {
    DMatrix dtrain;
    DMatrix dtest;
    DMatrix[] dmats;
    String[] names;
    Booster booster;

    /**
     * create an cross validation package
     *
     * @param dtrain train data
     * @param dtest  test data
     * @param params parameters
     * @throws XGBoostError native error
     */
    public CVPack(DMatrix dtrain, DMatrix dtest, Map<String, Object> params)
            throws XGBoostError {
      dmats = new DMatrix[]{dtrain, dtest};
      booster = new Booster(params, dmats);
      names = new String[]{"train", "test"};
      this.dtrain = dtrain;
      this.dtest = dtest;
    }

    /**
     * update one iteration
     *
     * @param iter iteration num
     * @throws XGBoostError native error
     */
    public void update(int iter) throws XGBoostError {
      booster.update(dtrain, iter);
    }

    /**
     * update one iteration
     *
     * @param obj  customized objective
     * @throws XGBoostError native error
     */
    public void update(IObjective obj) throws XGBoostError {
      booster.update(dtrain, obj);
    }

    /**
     * evaluation
     *
     * @param iter iteration num
     * @return evaluation
     * @throws XGBoostError native error
     */
    public String eval(int iter) throws XGBoostError {
      return booster.evalSet(dmats, names, iter);
    }

    /**
     * evaluation
     *
     * @param eval customized eval
     * @return evaluation
     * @throws XGBoostError native error
     */
    public String eval(IEvaluation eval) throws XGBoostError {
      return booster.evalSet(dmats, names, eval);
    }
  }
}
