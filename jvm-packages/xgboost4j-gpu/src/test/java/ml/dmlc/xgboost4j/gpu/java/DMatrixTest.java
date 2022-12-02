/*
 Copyright (c) 2021-2022 by Contributors

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

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import junit.framework.TestCase;

import com.google.common.primitives.Floats;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;

import ai.rapids.cudf.Table;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.QuantileDMatrix;
import ml.dmlc.xgboost4j.java.ColumnBatch;
import ml.dmlc.xgboost4j.java.XGBoostError;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test suite for DMatrix based on GPU
 */
public class DMatrixTest {

  @Test
  public void testCreateFromArrayInterfaceColumns() {
    Float[] labelFloats = new Float[]{2f, 4f, 6f, 8f, 10f};

    Throwable ex = null;
    try (
      Table X = new Table.TestBuilder().column(1.f, null, 5.f, 7.f, 9.f).build();
      Table y = new Table.TestBuilder().column(labelFloats).build();
      Table w = new Table.TestBuilder().column(labelFloats).build();
      Table margin = new Table.TestBuilder().column(labelFloats).build();) {

      CudfColumnBatch cudfDataFrame = new CudfColumnBatch(X, y, w, null);

      CudfColumn labelColumn = CudfColumn.from(y.getColumn(0));
      CudfColumn weightColumn = CudfColumn.from(w.getColumn(0));
      CudfColumn baseMarginColumn = CudfColumn.from(margin.getColumn(0));

      DMatrix dMatrix = new DMatrix(cudfDataFrame, 0, 1);
      dMatrix.setLabel(labelColumn);
      dMatrix.setWeight(weightColumn);
      dMatrix.setBaseMargin(baseMarginColumn);

      String[] featureNames = new String[]{"f1"};
      dMatrix.setFeatureNames(featureNames);
      String[] retFeatureNames = dMatrix.getFeatureNames();
      assertArrayEquals(featureNames, retFeatureNames);

      String[] featureTypes = new String[]{"i"};
      dMatrix.setFeatureTypes(featureTypes);
      String[] retFeatureTypes = dMatrix.getFeatureTypes();
      assertArrayEquals(featureTypes, retFeatureTypes);

      float[] anchor = convertFloatTofloat(labelFloats);
      float[] label = dMatrix.getLabel();
      float[] weight = dMatrix.getWeight();
      float[] baseMargin = dMatrix.getBaseMargin();

      TestCase.assertTrue(Arrays.equals(anchor, label));
      TestCase.assertTrue(Arrays.equals(anchor, weight));
      TestCase.assertTrue(Arrays.equals(anchor, baseMargin));
    } catch (Throwable e) {
      ex = e;
      e.printStackTrace();
    }
    TestCase.assertNull(ex);
  }

  @Test
  public void testCreateFromColumnDataIterator() throws XGBoostError {

    Float[] label1 = {25f, 21f, 22f, 20f, 24f};
    Float[] weight1 = {1.3f, 2.31f, 0.32f, 3.3f, 1.34f};
    Float[] baseMargin1 = {1.2f, 0.2f, 1.3f, 2.4f, 3.5f};

    Float[] label2 = {9f, 5f, 4f, 10f, 12f};
    Float[] weight2 = {3.0f, 1.3f, 3.2f, 0.3f, 1.34f};
    Float[] baseMargin2 = {0.2f, 2.5f, 3.1f, 4.4f, 2.2f};

    try (
      Table X_0 = new Table.TestBuilder()
        .column(1.2f, null, 5.2f, 7.2f, 9.2f)
        .column(0.2f, 0.4f, 0.6f, 2.6f, 0.10f)
        .build();
      Table y_0 = new Table.TestBuilder().column(label1).build();
      Table w_0 = new Table.TestBuilder().column(weight1).build();
      Table m_0 = new Table.TestBuilder().column(baseMargin1).build();
      Table X_1 = new Table.TestBuilder().column(11.2f, 11.2f, 15.2f, 17.2f, 19.2f)
        .column(1.2f, 1.4f, null, 12.6f, 10.10f).build();
      Table y_1 = new Table.TestBuilder().column(label2).build();
      Table w_1 = new Table.TestBuilder().column(weight2).build();
      Table m_1 = new Table.TestBuilder().column(baseMargin2).build();) {

      List<ColumnBatch> tables = new LinkedList<>();

      tables.add(new CudfColumnBatch(X_0, y_0, w_0, m_0));
      tables.add(new CudfColumnBatch(X_1, y_1, w_1, m_1));

      DMatrix dmat = new QuantileDMatrix(tables.iterator(), 0.0f, 8, 1);

      float[] anchorLabel = convertFloatTofloat((Float[]) ArrayUtils.addAll(label1, label2));
      float[] anchorWeight = convertFloatTofloat((Float[]) ArrayUtils.addAll(weight1, weight2));
      float[] anchorBaseMargin = convertFloatTofloat((Float[]) ArrayUtils.addAll(baseMargin1, baseMargin2));

      TestCase.assertTrue(Arrays.equals(anchorLabel, dmat.getLabel()));
      TestCase.assertTrue(Arrays.equals(anchorWeight, dmat.getWeight()));
      TestCase.assertTrue(Arrays.equals(anchorBaseMargin, dmat.getBaseMargin()));
    }
  }

  private float[] convertFloatTofloat(Float[] in) {
    return Floats.toArray(Arrays.asList(in));
  }
}
