/*
 Copyright (c) 2021 by Contributors

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

package ml.dmlc.xgboost4j.gpu.java;

import java.io.File;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import junit.framework.TestCase;

import org.junit.Test;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CSVOptions;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.ColumnBatch;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.QuantileDMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

/**
 * Tests the BoosterTest trained by DMatrix
 * @throws XGBoostError
 */
public class BoosterTest {

  @Test
  public void testBooster() throws XGBoostError {
    String trainingDataPath = "../../demo/data/veterans_lung_cancer.csv";
    Schema schema = Schema.builder()
      .column(DType.FLOAT32, "A")
      .column(DType.FLOAT32, "B")
      .column(DType.FLOAT32, "C")
      .column(DType.FLOAT32, "D")

      .column(DType.FLOAT32, "E")
      .column(DType.FLOAT32, "F")
      .column(DType.FLOAT32, "G")
      .column(DType.FLOAT32, "H")

      .column(DType.FLOAT32, "I")
      .column(DType.FLOAT32, "J")
      .column(DType.FLOAT32, "K")
      .column(DType.FLOAT32, "L")

      .column(DType.FLOAT32, "label")
      .build();
    CSVOptions opts = CSVOptions.builder()
      .hasHeader().build();

    int maxBin = 16;
    int round = 10;
    //set params
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 2);
        put("objective", "binary:logistic");
        put("num_round", round);
        put("num_workers", 1);
        put("tree_method", "gpu_hist");
        put("predictor", "gpu_predictor");
        put("max_bin", maxBin);
      }
    };

    try (Table tmpTable = Table.readCSV(schema, opts, new File(trainingDataPath))) {
      ColumnVector[] df = new ColumnVector[10];
      // exclude the first two columns, they are label bounds and contain inf.
      for (int i = 2; i < 12; ++i) {
        df[i - 2] = tmpTable.getColumn(i);
      }
      try (Table X = new Table(df);) {
        ColumnVector[] labels = new ColumnVector[1];
        labels[0] = tmpTable.getColumn(12);

        try (Table y = new Table(labels);) {

          CudfColumnBatch batch = new CudfColumnBatch(X, y, null, null);
          CudfColumn labelColumn = CudfColumn.from(tmpTable.getColumn(12));

          //set watchList
          HashMap<String, DMatrix> watches = new HashMap<>();

          DMatrix dMatrix1 = new DMatrix(batch, Float.NaN, 1);
          dMatrix1.setLabel(labelColumn);
          watches.put("train", dMatrix1);
          Booster model1 = XGBoost.train(dMatrix1, paramMap, round, watches, null, null);

          List<ColumnBatch> tables = new LinkedList<>();
          tables.add(batch);
          DMatrix incrementalDMatrix = new QuantileDMatrix(tables.iterator(), Float.NaN, maxBin, 1);
          //set watchList
          HashMap<String, DMatrix> watches1 = new HashMap<>();
          watches1.put("train", incrementalDMatrix);
          Booster model2 = XGBoost.train(incrementalDMatrix, paramMap, round, watches1, null, null);

          float[][] predicat1 = model1.predict(dMatrix1);
          float[][] predicat2 = model2.predict(dMatrix1);

          for (int i = 0; i < tmpTable.getRowCount(); i++) {
            TestCase.assertTrue(predicat1[i][0] - predicat2[i][0] < 1e-6);
          }
        }
      }
    }
  }

}
