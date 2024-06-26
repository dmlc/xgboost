/*
 Copyright (c) 2021-2024 by Contributors

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

import java.io.File;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * Tests the BoosterTest trained by DMatrix
 *
 * @throws XGBoostError
 */
public class BoosterTest {

  @Test
  public void testBooster() throws XGBoostError {
    String resourcePath = getClass().getResource("/binary.train.parquet").getFile();

    int maxBin = 16;
    int round = 10;
    //set params
    Map<String, Object> paramMap = new HashMap<String, Object>() {
      {
        put("max_depth", 2);
        put("objective", "binary:logistic");
        put("num_round", round);
        put("num_workers", 1);
        put("tree_method", "hist");
        put("device", "cuda");
        put("max_bin", maxBin);
      }
    };

    try (Table table = Table.readParquet(new File(resourcePath))) {
      ColumnVector[] features = new ColumnVector[6];
      for (int i = 0; i < 6; i++) {
        features[i] = table.getColumn(i);
      }

      try (Table X = new Table(features)) {
        ColumnVector[] labels = new ColumnVector[1];
        labels[0] = table.getColumn(6);

        try (Table y = new Table(labels)) {

          CudfColumnBatch batch = new CudfColumnBatch(X, y, null, null);
          CudfColumn labelColumn = CudfColumn.from(y.getColumn(0));

          // train XGBoost Booster base on DMatrix
          HashMap<String, DMatrix> watches = new HashMap<>();
          DMatrix dMatrix1 = new DMatrix(batch, Float.NaN, 1);
          dMatrix1.setLabel(labelColumn);
          watches.put("train", dMatrix1);
          Booster model1 = XGBoost.train(dMatrix1, paramMap, round, watches, null, null);

          // train XGBoost Booster base on QuantileDMatrix
          List<ColumnBatch> tables = new LinkedList<>();
          tables.add(batch);
          DMatrix incrementalDMatrix = new QuantileDMatrix(tables.iterator(), Float.NaN, maxBin, 1);
          HashMap<String, DMatrix> watches1 = new HashMap<>();
          watches1.put("train", incrementalDMatrix);
          Booster model2 = XGBoost.train(incrementalDMatrix, paramMap, round, watches1, null, null);

          float[][] predicat1 = model1.predict(dMatrix1);
          float[][] predicat2 = model2.predict(dMatrix1);

          for (int i = 0; i < table.getRowCount(); i++) {
            TestCase.assertTrue(predicat1[i][0] - predicat2[i][0] < 1e-6);
          }
        }
      }
    }
  }
}
