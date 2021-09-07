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
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

/**
 * Tests the BoosterTest trained by ColumnDMatrix
 * @throws XGBoostError
 */
public class BoosterTest {

  @Test
  public void testBooster() throws XGBoostError {
    Schema schema = Schema.builder()
      .column(DType.FLOAT32, "A")  // column 0
      .column(DType.FLOAT32, "B")  // column 1
      .column(DType.FLOAT32, "C")  // column 2
      .column(DType.FLOAT32, "D")  // column 3
      .column(DType.FLOAT32, "label") // column 4
      .build();

    int maxBin = 16;
    int round = 100;
    //set params
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 2);
        put("objective", "multi:softprob");
        put("num_class", 3);
        put("num_round", round);
        put("num_workers", 1);
        put("tree_method", "gpu_hist");
        put("predictor", "gpu_predictor");
        put("max_bin", maxBin);
      }
    };

    try (Table tmpTable = Table.readCSV(schema,
        new File("./src/test/resources/iris.data.csv"))) {

      CudfTable cudfTable = new CudfTable(tmpTable, new int[]{0, 1, 2, 3}, new int[]{4});

      //set watchList
      HashMap<String, DMatrix> watches = new HashMap<>();

      ColumnDMatrix dMatrix1 = new ColumnDMatrix(cudfTable, Float.NaN, 1);
      dMatrix1.setLabel(cudfTable);
      watches.put("train", dMatrix1);
      Booster model1 = XGBoost.train(dMatrix1, paramMap, round, watches, null, null);

      List<XGBoostTable> tables = new LinkedList<>();
      tables.add(cudfTable);
      DMatrix incrementalDMatrix = new ColumnDMatrix(tables.iterator(), Float.NaN, maxBin, 1);
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
