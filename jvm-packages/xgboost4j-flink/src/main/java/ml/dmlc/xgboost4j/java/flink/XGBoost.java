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


import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.util.Collector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.Communicator;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.RabitTracker;
import ml.dmlc.xgboost4j.java.XGBoostError;


public class XGBoost {
  private static final Logger logger = LoggerFactory.getLogger(XGBoost.class);

  private static class MapFunction
      extends RichMapPartitionFunction<Tuple2<Vector, Double>, XGBoostModel> {

    private final Map<String, Object> params;
    private final int round;
    private final Map<String, String> workerEnvs;

    public MapFunction(Map<String, Object> params, int round, Map<String, String> workerEnvs) {
      this.params = params;
      this.round = round;
      this.workerEnvs = workerEnvs;
    }

    public void mapPartition(java.lang.Iterable<Tuple2<Vector, Double>> it,
                             Collector<XGBoostModel> collector) throws XGBoostError {
      workerEnvs.put(
          "DMLC_TASK_ID",
          String.valueOf(this.getRuntimeContext().getIndexOfThisSubtask())
      );

      if (logger.isInfoEnabled()) {
        logger.info("start with env: {}", workerEnvs.entrySet().stream()
            .map(e -> String.format("\"%s\": \"%s\"", e.getKey(), e.getValue()))
            .collect(Collectors.joining(", "))
        );
      }

      final Iterator<LabeledPoint> dataIter =
          StreamSupport
            .stream(it.spliterator(), false)
            .map(VectorToPointMapper.INSTANCE)
            .iterator();

      if (dataIter.hasNext()) {
        final DMatrix trainMat = new DMatrix(dataIter, null);
        int numEarlyStoppingRounds =
            Optional.ofNullable(params.get("numEarlyStoppingRounds"))
              .map(x -> Integer.parseInt(x.toString()))
              .orElse(0);

        final Booster booster = trainBooster(trainMat, numEarlyStoppingRounds);
        collector.collect(new XGBoostModel(booster));
      } else {
        logger.warn("Nothing to train with.");
      }
    }

    private Booster trainBooster(DMatrix trainMat,
                                 int numEarlyStoppingRounds) throws XGBoostError {
      Booster booster;
      final Map<String, DMatrix> watches =
          new HashMap<String, DMatrix>() {{ put("train", trainMat); }};
      try {
        Communicator.init(workerEnvs);
        booster = ml.dmlc.xgboost4j.java.XGBoost
          .train(
            trainMat,
            params,
            round,
            watches,
            null,
            null,
            null,
            numEarlyStoppingRounds);
      } catch (XGBoostError xgbException) {
        final String identifier = String.valueOf(this.getRuntimeContext().getIndexOfThisSubtask());
        logger.warn(
            String.format("XGBooster worker %s has failed due to", identifier),
            xgbException
        );
        throw xgbException;
      } finally {
        Communicator.shutdown();
      }
      return booster;
    }

    private static class VectorToPointMapper
        implements Function<Tuple2<Vector, Double>, LabeledPoint> {
      public static VectorToPointMapper INSTANCE = new VectorToPointMapper();
      @Override
      public LabeledPoint apply(Tuple2<Vector, Double> tuple) {
        final SparseVector vector = tuple.f0.toSparse();
        final double[] values = vector.values;
        final int size = values.length;
        final float[] array = new float[size];
        for (int i = 0; i < size; i++) {
          array[i] = (float) values[i];
        }
        return new LabeledPoint(
          tuple.f1.floatValue(),
          vector.size(),
          vector.indices,
          array);
      }
    }
  }

  /**
   * Load XGBoost model from path, using Hadoop Filesystem API.
   *
   * @param modelPath The path that is accessible by hadoop filesystem API.
   * @return The loaded model
   */
  public static XGBoostModel loadModelFromHadoopFile(final String modelPath) throws Exception {
    final FileSystem fileSystem = FileSystem.get(new Configuration());
    final Path f = new Path(modelPath);

    try (FSDataInputStream opened = fileSystem.open(f)) {
      return new XGBoostModel(ml.dmlc.xgboost4j.java.XGBoost.loadModel(opened));
    }
  }

  /**
   * Train a xgboost model with link.
   *
   * @param dtrain The training data.
   * @param params XGBoost parameters.
   * @param numBoostRound  Number of rounds to train.
   */
  public static XGBoostModel train(DataSet<Tuple2<Vector, Double>> dtrain,
                                   Map<String, Object> params,
                                   int numBoostRound) throws Exception {
    final RabitTracker tracker =
        new RabitTracker(dtrain.getExecutionEnvironment().getParallelism());
    if (tracker.start(0L)) {
      return dtrain
        .mapPartition(new MapFunction(params, numBoostRound, tracker.getWorkerEnvs()))
        .reduce((x, y) -> x)
        .collect()
        .get(0);
    } else {
      throw new Error("Tracker cannot be started");
    }
  }
}
