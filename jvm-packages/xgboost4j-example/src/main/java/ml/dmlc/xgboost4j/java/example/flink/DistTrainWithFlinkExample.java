/*
 Copyright (c) 2014-2021 by Contributors

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
package ml.dmlc.xgboost4j.java.example.flink;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.MapOperator;
import org.apache.flink.api.java.tuple.Tuple13;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.utils.DataSetUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;

import ml.dmlc.xgboost4j.java.flink.XGBoost;
import ml.dmlc.xgboost4j.java.flink.XGBoostModel;


public class DistTrainWithFlinkExample {

  static Tuple2<XGBoostModel, DataSet<Float[]>> runPrediction(
      ExecutionEnvironment env,
      java.nio.file.Path trainPath,
      int percentage) throws Exception {
    // reading data
    final DataSet<Tuple2<Long, Tuple2<Vector, Double>>> data =
        DataSetUtils.zipWithIndex(parseCsv(env, trainPath));
    final long size = data.count();
    final long trainCount = Math.round(size * 0.01 * percentage);
    final DataSet<Tuple2<Vector, Double>> trainData =
        data
          .filter(item -> item.f0 < trainCount)
          .map(t -> t.f1)
          .returns(TypeInformation.of(new TypeHint<Tuple2<Vector, Double>>(){}));
    final DataSet<Vector> testData =
        data
          .filter(tuple -> tuple.f0 >= trainCount)
          .map(t -> t.f1.f0)
          .returns(TypeInformation.of(new TypeHint<Vector>(){}));

    // define parameters
    HashMap<String, Object> paramMap = new HashMap<String, Object>(3);
    paramMap.put("eta", 0.1);
    paramMap.put("max_depth", 2);
    paramMap.put("objective", "binary:logistic");

    // number of iterations
    final int round = 2;
    // train the model
    XGBoostModel model = XGBoost.train(trainData, paramMap, round);
    DataSet<Float[]> predTest = model.predict(testData);
    return new Tuple2<XGBoostModel, DataSet<Float[]>>(model, predTest);
  }

  private static MapOperator<Tuple13<Double, String, Double, Double, Double, Integer, Integer,
      Integer, Integer, Integer, Integer, Integer, Integer>,
      Tuple2<Vector, Double>> parseCsv(ExecutionEnvironment env, Path trainPath) {
    return env.readCsvFile(trainPath.toString())
      .ignoreFirstLine()
      .types(Double.class, String.class, Double.class, Double.class, Double.class,
        Integer.class, Integer.class, Integer.class, Integer.class, Integer.class,
        Integer.class, Integer.class, Integer.class)
      .map(DistTrainWithFlinkExample::mapFunction);
  }

  private static Tuple2<Vector, Double> mapFunction(Tuple13<Double, String, Double, Double, Double,
      Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer> tuple) {
    final DenseVector dense = Vectors.dense(tuple.f2, tuple.f3, tuple.f4, tuple.f5, tuple.f6,
        tuple.f7, tuple.f8, tuple.f9, tuple.f10, tuple.f11, tuple.f12);
    if (tuple.f1.contains("inf")) {
      return new Tuple2<Vector, Double>(dense, 1.0);
    } else {
      return new Tuple2<Vector, Double>(dense, 0.0);
    }
  }

  public static void main(String[] args) throws Exception {
    final java.nio.file.Path parentPath = java.nio.file.Paths.get(Arrays.stream(args)
        .findFirst().orElse("."));
    final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
    Tuple2<XGBoostModel, DataSet<Float[]>> tuple2 = runPrediction(
        env, parentPath.resolve("veterans_lung_cancer.csv"), 70
    );
    List<Float[]> list = tuple2.f1.collect();
    System.out.println(list.size());
  }
}
