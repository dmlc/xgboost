package ml.dmlc.xgboost4j;

import java.io.IOException;
import java.util.Map;

public interface Booster {

  /**
   * set parameter
   *
   * @param key   param name
   * @param value param value
   */
  void setParam(String key, String value) throws XGBoostError;

  /**
   * set parameters
   *
   * @param params parameters key-value map
   */
  void setParams(Map<String, Object> params) throws XGBoostError;

  /**
   * Update (one iteration)
   *
   * @param dtrain training data
   * @param iter   current iteration number
   */
  void update(DMatrix dtrain, int iter) throws XGBoostError;

  /**
   * update with customize obj func
   *
   * @param dtrain training data
   * @param obj    customized objective class
   */
  void update(DMatrix dtrain, IObjective obj) throws XGBoostError;

  /**
   * update with give grad and hess
   *
   * @param dtrain training data
   * @param grad   first order of gradient
   * @param hess   seconde order of gradient
   */
  void boost(DMatrix dtrain, float[] grad, float[] hess) throws XGBoostError;

  /**
   * evaluate with given dmatrixs.
   *
   * @param evalMatrixs dmatrixs for evaluation
   * @param evalNames   name for eval dmatrixs, used for check results
   * @param iter        current eval iteration
   * @return eval information
   */
  String evalSet(DMatrix[] evalMatrixs, String[] evalNames, int iter) throws XGBoostError;

  /**
   * evaluate with given customized Evaluation class
   *
   * @param evalMatrixs evaluation matrix
   * @param evalNames   evaluation names
   * @param eval        custom evaluator
   * @return eval information
   */
  String evalSet(DMatrix[] evalMatrixs, String[] evalNames, IEvaluation eval) throws XGBoostError;

  /**
   * Predict with data
   *
   * @param data dmatrix storing the input
   * @return predict result
   */
  float[][] predict(DMatrix data) throws XGBoostError;


  /**
   * Predict with data
   *
   * @param data         dmatrix storing the input
   * @param outPutMargin Whether to output the raw untransformed margin value.
   * @return predict result
   */
  float[][] predict(DMatrix data, boolean outPutMargin) throws XGBoostError;


  /**
   * Predict with data
   *
   * @param data         dmatrix storing the input
   * @param outPutMargin Whether to output the raw untransformed margin value.
   * @param treeLimit    Limit number of trees in the prediction; defaults to 0 (use all trees).
   * @return predict result
   */
  float[][] predict(DMatrix data, boolean outPutMargin, int treeLimit) throws XGBoostError;


  /**
   * Predict with data
   * @param data dmatrix storing the input
   * @param treeLimit Limit number of trees in the prediction; defaults to 0 (use all trees).
   * @param predLeaf When this option is on, the output will be a matrix of (nsample, ntrees),
   *                 nsample = data.numRow with each record indicating the predicted leaf index of
   *                 each sample in each tree. Note that the leaf index of a tree is unique per
   *                 tree, so you may find leaf 1 in both tree 1 and tree 0.
   * @return predict result
   * @throws XGBoostError native error
   */
  float[][] predict(DMatrix data, int treeLimit, boolean predLeaf) throws XGBoostError;

  /**
   * save model to modelPath
   *
   * @param modelPath model path
   */
  void saveModel(String modelPath) throws XGBoostError;

  /**
   * Dump model into a text file.
   *
   * @param modelPath file to save dumped model info
   * @param withStats bool Controls whether the split statistics are output.
   */
  void dumpModel(String modelPath, boolean withStats) throws IOException, XGBoostError;

  /**
   * Dump model into a text file.
   *
   * @param modelPath  file to save dumped model info
   * @param featureMap featureMap file
   * @param withStats  bool
   *                   Controls whether the split statistics are output.
   */
  void dumpModel(String modelPath, String featureMap, boolean withStats)
          throws IOException, XGBoostError;

  /**
   * get importance of each feature
   *
   * @return featureMap  key: feature index, value: feature importance score
   */
  Map<String, Integer> getFeatureScore() throws XGBoostError ;

  /**
   * get importance of each feature
   *
   * @param featureMap file to save dumped model info
   * @return featureMap  key: feature index, value: feature importance score
   */
  Map<String, Integer> getFeatureScore(String featureMap) throws XGBoostError;

  void dispose();
}
