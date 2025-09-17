/*
 Copyright (c) 2014-2023 by Contributors

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

package ml.dmlc.xgboost4j.java.flink;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.StreamSupport;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.util.Collector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;


public class XGBoostModel implements Serializable {
  private static final org.slf4j.Logger logger =
      org.slf4j.LoggerFactory.getLogger(XGBoostModel.class);

  private final Booster booster;
  private final PredictorFunction predictorFunction;


  public XGBoostModel(Booster booster) {
    this.booster = booster;
    this.predictorFunction = new PredictorFunction(booster);
  }

  /**
   * Save the model as a Hadoop filesystem file.
   *
   * @param modelPath The model path as in Hadoop path.
   */
  public void saveModelAsHadoopFile(String modelPath) throws IOException, XGBoostError {
    booster.saveModel(FileSystem.get(new Configuration()).create(new Path(modelPath)));
  }

  public byte[] toByteArray(String format) throws XGBoostError {
    return booster.toByteArray(format);
  }

  /**
   * Save the model as a Hadoop filesystem file.
   *
   * @param modelPath The model path as in Hadoop path.
   * @param format The model format (ubj, json, deprecated)
   * @throws XGBoostError internal error
   * @throws IOException save error
   */
  public void saveModelAsHadoopFile(String modelPath, String format)
      throws IOException, XGBoostError {
    booster.saveModel(FileSystem.get(new Configuration()).create(new Path(modelPath)), format);
  }

  /**
   * predict with the given DMatrix
   *
   * @param testSet the local test set represented as DMatrix
   * @return prediction result
   */
  public float[][] predict(DMatrix testSet) throws XGBoostError {
    return booster.predict(testSet, true, 0);
  }

  /**
   * Predict given vector dataset.
   *
   * @param data The dataset to be predicted.
   * @return The prediction result.
   */
  public DataSet<Float[]> predict(DataSet<Vector> data) {
    return data.mapPartition(predictorFunction);
  }


  private static class PredictorFunction implements MapPartitionFunction<Vector, Float[]> {

    private final Booster booster;

    public PredictorFunction(Booster booster) {
      this.booster = booster;
    }

    @Override
    public void mapPartition(Iterable<Vector> it, Collector<Float[]> out) throws Exception {
      final Iterator<LabeledPoint> dataIter =
          StreamSupport.stream(it.spliterator(), false)
            .map(Vector::toSparse)
            .map(PredictorFunction::fromVector)
            .iterator();

      if (dataIter.hasNext()) {
        final DMatrix data = new DMatrix(dataIter, null);
        float[][] predictions = booster.predict(data, true, 2);
        Arrays.stream(predictions).map(ArrayUtils::toObject).forEach(out::collect);
      } else {
        logger.debug("Empty partition");
      }
    }

    private static LabeledPoint fromVector(SparseVector vector) {
      final int[] index = vector.indices;
      final double[] value = vector.values;
      int size = value.length;
      final float[] values = new float[size];
      for (int i = 0; i < size; i++) {
        values[i] = (float) value[i];
      }
      return new LabeledPoint(0.0f, vector.size(), index, values);
    }
  }
}
