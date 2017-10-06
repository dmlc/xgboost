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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoSerializable;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Booster for xgboost, this is a model API that support interactive build of a XGBoost Model
 */
public class Booster implements Serializable, KryoSerializable {
  private static final Log logger = LogFactory.getLog(Booster.class);
  // handle to the booster.
  private long handle = 0;

  /**
   * Create a new Booster with empty stage.
   *
   * @param params  Model parameters
   * @param cacheMats Cached DMatrix entries,
   *                  the prediction of these DMatrices will become faster than not-cached data.
   * @throws XGBoostError native error
   */
  Booster(Map<String, Object> params, DMatrix[] cacheMats) throws XGBoostError {
    init(cacheMats);
    setParam("seed", "0");
    setParams(params);
  }

  /**
   * Load a new Booster model from modelPath
   * @param modelPath The path to the model.
   * @return The created Booster.
   * @throws XGBoostError
   */
  static Booster loadModel(String modelPath) throws XGBoostError {
    if (modelPath == null) {
      throw new NullPointerException("modelPath : null");
    }
    Booster ret = new Booster(new HashMap<String, Object>(), new DMatrix[0]);
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterLoadModel(ret.handle, modelPath));
    return ret;
  }

  /**
   * Load a new Booster model from a file opened as input stream.
   * The assumption is the input stream only contains one XGBoost Model.
   * This can be used to load existing booster models saved by other xgboost bindings.
   *
   * @param in The input stream of the file.
   * @return The create boosted
   * @throws XGBoostError
   * @throws IOException
   */
  static Booster loadModel(InputStream in) throws XGBoostError, IOException {
    int size;
    byte[] buf = new byte[1<<20];
    ByteArrayOutputStream os = new ByteArrayOutputStream();
    while ((size = in.read(buf)) != -1) {
      os.write(buf, 0, size);
    }
    in.close();
    Booster ret = new Booster(new HashMap<String, Object>(), new DMatrix[0]);
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterLoadModelFromBuffer(ret.handle,os.toByteArray()));
    return ret;
  }

  /**
   * Set parameter to the Booster.
   *
   * @param key   param name
   * @param value param value
   * @throws XGBoostError native error
   */
  public final void setParam(String key, Object value) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterSetParam(handle, key, value.toString()));
  }

  /**
   * Set parameters to the Booster.
   *
   * @param params parameters key-value map
   * @throws XGBoostError native error
   */
  public void setParams(Map<String, Object> params) throws XGBoostError {
    if (params != null) {
      for (Map.Entry<String, Object> entry : params.entrySet()) {
        setParam(entry.getKey(), entry.getValue().toString());
      }
    }
  }

  /**
   * Update the booster for one iteration.
   *
   * @param dtrain training data
   * @param iter   current iteration number
   * @throws XGBoostError native error
   */
  public void update(DMatrix dtrain, int iter) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterUpdateOneIter(handle, iter, dtrain.getHandle()));
  }

  /**
   * Update with customize obj func
   *
   * @param dtrain training data
   * @param obj    customized objective class
   * @throws XGBoostError native error
   */
  public void update(DMatrix dtrain, IObjective obj) throws XGBoostError {
    float[][] predicts = this.predict(dtrain, true, 0, false, false);
    List<float[]> gradients = obj.getGradient(predicts, dtrain);
    boost(dtrain, gradients.get(0), gradients.get(1));
  }

  /**
   * update with give grad and hess
   *
   * @param dtrain training data
   * @param grad   first order of gradient
   * @param hess   seconde order of gradient
   * @throws XGBoostError native error
   */
  public void boost(DMatrix dtrain, float[] grad, float[] hess) throws XGBoostError {
    if (grad.length != hess.length) {
      throw new AssertionError(String.format("grad/hess length mismatch %s / %s", grad.length,
              hess.length));
    }
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterBoostOneIter(handle,
            dtrain.getHandle(), grad, hess));
  }

  /**
   * evaluate with given dmatrixs.
   *
   * @param evalMatrixs dmatrixs for evaluation
   * @param evalNames   name for eval dmatrixs, used for check results
   * @param iter        current eval iteration
   * @return eval information
   * @throws XGBoostError native error
   */
  public String evalSet(DMatrix[] evalMatrixs, String[] evalNames, int iter) throws XGBoostError {
    long[] handles = dmatrixsToHandles(evalMatrixs);
    String[] evalInfo = new String[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterEvalOneIter(handle, iter, handles, evalNames,
            evalInfo));
    return evalInfo[0];
  }

  /**
   * evaluate with given dmatrixs.
   *
   * @param evalMatrixs dmatrixs for evaluation
   * @param evalNames   name for eval dmatrixs, used for check results
   * @param iter        current eval iteration
   * @param metricsOut  output array containing the evaluation metrics for each evalMatrix
   * @return eval information
   * @throws XGBoostError native error
   */
  public String evalSet(DMatrix[] evalMatrixs, String[] evalNames, int iter, float[] metricsOut)
          throws XGBoostError {
    String stringFormat = evalSet(evalMatrixs, evalNames, iter);
    String[] metricPairs = stringFormat.split("\t");
    for (int i = 1; i < metricPairs.length; i++) {
      metricsOut[i - 1] = Float.valueOf(metricPairs[i].split(":")[1]);
    }
    return stringFormat;
  }

  /**
   * evaluate with given customized Evaluation class
   *
   * @param evalMatrixs evaluation matrix
   * @param evalNames   evaluation names
   * @param eval        custom evaluator
   * @return eval information
   * @throws XGBoostError native error
   */
  public String evalSet(DMatrix[] evalMatrixs, String[] evalNames, IEvaluation eval)
          throws XGBoostError {
    // Hopefully, a tiny redundant allocation wouldn't hurt.
    return evalSet(evalMatrixs, evalNames, eval, new float[evalNames.length]);
  }

  public String evalSet(DMatrix[] evalMatrixs, String[] evalNames, IEvaluation eval,
                        float[] metricsOut) throws XGBoostError {
    String evalInfo = "";
    for (int i = 0; i < evalNames.length; i++) {
      String evalName = evalNames[i];
      DMatrix evalMat = evalMatrixs[i];
      float evalResult = eval.eval(predict(evalMat), evalMat);
      String evalMetric = eval.getMetric();
      evalInfo += String.format("\t%s-%s:%f", evalName, evalMetric, evalResult);
      metricsOut[i] = evalResult;
    }
    return evalInfo;
  }

  /**
   * Advanced predict function with all the options.
   *
   * @param data         data
   * @param outputMargin output margin
   * @param treeLimit    limit number of trees, 0 means all trees.
   * @param predLeaf     prediction minimum to keep leafs
   * @param predContribs prediction feature contributions
   * @return predict results
   */
  private synchronized float[][] predict(DMatrix data,
                                         boolean outputMargin,
                                         int treeLimit,
                                         boolean predLeaf,
                                         boolean predContribs) throws XGBoostError {
    int optionMask = 0;
    if (outputMargin) {
      optionMask = 1;
    }
    if (predLeaf) {
      optionMask = 2;
    }
    if (predContribs) {
      optionMask = 4;
    }
    float[][] rawPredicts = new float[1][];
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterPredict(handle, data.getHandle(), optionMask,
            treeLimit, rawPredicts));
    int row = (int) data.rowNum();
    int col = rawPredicts[0].length / row;
    float[][] predicts = new float[row][col];
    int r, c;
    for (int i = 0; i < rawPredicts[0].length; i++) {
      r = i / col;
      c = i % col;
      predicts[r][c] = rawPredicts[0][i];
    }
    return predicts;
  }

  /**
   * Predict leaf indices given the data
   *
   * @param data The input data.
   * @param treeLimit Number of trees to include, 0 means all trees.
   * @return The leaf indices of the instance.
   * @throws XGBoostError
   */
  public float[][] predictLeaf(DMatrix data, int treeLimit) throws XGBoostError {
    return this.predict(data, false, treeLimit, true, false);
  }

  /**
   * Output feature contributions toward predictions of given data
   *
   * @param data The input data.
   * @param treeLimit Number of trees to include, 0 means all trees.
   * @return The feature contributions and bias.
   * @throws XGBoostError
   */
  public float[][] predictContrib(DMatrix data, int treeLimit) throws XGBoostError {
    return this.predict(data, false, treeLimit, true, true);
  }

  /**
   * Predict with data
   *
   * @param data dmatrix storing the input
   * @return predict result
   * @throws XGBoostError native error
   */
  public float[][] predict(DMatrix data) throws XGBoostError {
    return this.predict(data, false, 0, false, false);
  }

  /**
   * Predict with data
   *
   * @param data  data
   * @param outputMargin output margin
   * @return predict results
   */
  public float[][] predict(DMatrix data, boolean outputMargin) throws XGBoostError {
    return this.predict(data, outputMargin, 0, false, false);
  }

  /**
   * Advanced predict function with all the options.
   *
   * @param data         data
   * @param outputMargin output margin
   * @param treeLimit    limit number of trees, 0 means all trees.
   * @return predict results
   */
  public float[][] predict(DMatrix data, boolean outputMargin, int treeLimit) throws XGBoostError {
    return this.predict(data, outputMargin, treeLimit, false, false);
  }

  /**
   * Save model to modelPath
   *
   * @param modelPath model path
   */
  public void saveModel(String modelPath) throws XGBoostError{
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterSaveModel(handle, modelPath));
  }

  /**
   * Save the model to file opened as output stream.
   * The model format is compatible with other xgboost bindings.
   * The output stream can only save one xgboost model.
   * This function will close the OutputStream after the save.
   *
   * @param out The output stream
   */
  public void saveModel(OutputStream out) throws XGBoostError, IOException {
    out.write(this.toByteArray());
    out.close();
  }

  /**
   * Get the dump of the model as a string array
   *
   * @param withStats Controls whether the split statistics are output.
   * @return dumped model information
   * @throws XGBoostError native error
   */
  public String[] getModelDump(String featureMap, boolean withStats) throws XGBoostError {
    return getModelDump(featureMap, withStats, "text");
  }

  public String[] getModelDump(String featureMap, boolean withStats, String format)
         throws XGBoostError {
    int statsFlag = 0;
    if (featureMap == null) {
      featureMap = "";
    }
    if (withStats) {
      statsFlag = 1;
    }
    if (format == null) {
      format = "text";
    }
    String[][] modelInfos = new String[1][];
    XGBoostJNI.checkCall(
            XGBoostJNI.XGBoosterDumpModelEx(handle, featureMap, statsFlag, format, modelInfos));
    return modelInfos[0];
  }

  /**
   * Get importance of each feature
   *
   * @return featureMap  key: feature index, value: feature importance score, can be nill
   * @throws XGBoostError native error
   */
  public Map<String, Integer> getFeatureScore(String featureMap) throws XGBoostError {
    String[] modelInfos = getModelDump(featureMap, false);
    Map<String, Integer> featureScore = new HashMap<String, Integer>();
    for (String tree : modelInfos) {
      for (String node : tree.split("\n")) {
        String[] array = node.split("\\[");
        if (array.length == 1) {
          continue;
        }
        String fid = array[1].split("\\]")[0];
        fid = fid.split("<")[0];
        if (featureScore.containsKey(fid)) {
          featureScore.put(fid, 1 + featureScore.get(fid));
        } else {
          featureScore.put(fid, 1);
        }
      }
    }
    return featureScore;
  }

  /**
   * Save the model as byte array representation.
   * Write these bytes to a file will give compatible format with other xgboost bindings.
   *
   * If java natively support HDFS file API, use toByteArray and write the ByteArray
   *
   * @param withStats Controls whether the split statistics are output.
   * @return dumped model information
   * @throws XGBoostError native error
   */
  private String[] getDumpInfo(boolean withStats) throws XGBoostError {
    int statsFlag = 0;
    if (withStats) {
      statsFlag = 1;
    }
    String[][] modelInfos = new String[1][];
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterDumpModelEx(handle, "", statsFlag, "text",
            modelInfos));
    return modelInfos[0];
  }

  /**
   *
   * @return the saved byte array.
   * @throws XGBoostError native error
   */
  public byte[] toByteArray() throws XGBoostError {
    byte[][] bytes = new byte[1][];
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterGetModelRaw(this.handle, bytes));
    return bytes[0];
  }

  /**
   * Load the booster model from thread-local rabit checkpoint.
   * This is only used in distributed training.
   * @return the stored version number of the checkpoint.
   * @throws XGBoostError
   */
  int loadRabitCheckpoint() throws XGBoostError {
    int[] out = new int[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterLoadRabitCheckpoint(this.handle, out));
    return out[0];
  }

  /**
   * Save the booster model into thread-local rabit checkpoint.
   * This is only used in distributed training.
   * @throws XGBoostError
   */
  void saveRabitCheckpoint() throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterSaveRabitCheckpoint(this.handle));
  }

  /**
   * Internal initialization function.
   * @param cacheMats The cached DMatrix.
   * @throws XGBoostError
   */
  private void init(DMatrix[] cacheMats) throws XGBoostError {
    long[] handles = null;
    if (cacheMats != null) {
      handles = dmatrixsToHandles(cacheMats);
    }
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGBoosterCreate(handles, out));

    handle = out[0];
  }

  /**
   * transfer DMatrix array to handle array (used for native functions)
   *
   * @param dmatrixs
   * @return handle array for input dmatrixs
   */
  private static long[] dmatrixsToHandles(DMatrix[] dmatrixs) {
    long[] handles = new long[dmatrixs.length];
    for (int i = 0; i < dmatrixs.length; i++) {
      handles[i] = dmatrixs[i].getHandle();
    }
    return handles;
  }

  // making Booster serializable
  private void writeObject(java.io.ObjectOutputStream out) throws IOException {
    try {
      out.writeObject(this.toByteArray());
    } catch (XGBoostError ex) {
      ex.printStackTrace();
      logger.error(ex.getMessage());
    }
  }

  private void readObject(java.io.ObjectInputStream in)
          throws IOException, ClassNotFoundException {
    try {
      this.init(null);
      byte[] bytes = (byte[])in.readObject();
      XGBoostJNI.checkCall(XGBoostJNI.XGBoosterLoadModelFromBuffer(this.handle, bytes));
    } catch (XGBoostError ex) {
      ex.printStackTrace();
      logger.error(ex.getMessage());
    }
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();
    dispose();
  }

  public synchronized void dispose() {
    if (handle != 0L) {
      XGBoostJNI.XGBoosterFree(handle);
      handle = 0;
    }
  }

  @Override
  public void write(Kryo kryo, Output output) {
    try {
      byte[] serObj = this.toByteArray();
      int serObjSize = serObj.length;
      System.out.println("==== serialized obj size " + serObjSize);
      output.writeInt(serObjSize);
      output.write(serObj);
    } catch (XGBoostError ex) {
      ex.printStackTrace();
      logger.error(ex.getMessage());
    }
  }

  @Override
  public void read(Kryo kryo, Input input) {
    try {
      this.init(null);
      int serObjSize = input.readInt();
      System.out.println("==== the size of the object: " + serObjSize);
      byte[] bytes = new byte[serObjSize];
      input.readBytes(bytes);
      XGBoostJNI.checkCall(XGBoostJNI.XGBoosterLoadModelFromBuffer(this.handle, bytes));
    } catch (XGBoostError ex) {
      ex.printStackTrace();
      logger.error(ex.getMessage());
    }
  }
}
