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

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import junit.framework.TestCase;

import com.google.common.primitives.Floats;

import org.apache.commons.lang.ArrayUtils;
import org.junit.Test;

import ai.rapids.cudf.Table;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.DataFrame;
import ml.dmlc.xgboost4j.java.XGBoostError;

/**
 * Test suite for DMatrix based on GPU
 */
public class DMatrixTest {

  @Test
  public void testCreateFromArrayInterfaceColumns() {
    Float[] labelFloats = new Float[]{2f, 4f, 6f, 8f, 10f};

    Throwable ex = null;
    try (
      Table table = new Table.TestBuilder()
        .column(1.f, null, 5.f, 7.f, 9.f) // the feature columns
        .column(labelFloats)              // the label column
        .build()) {

      CudfDataFrame cudfDataFrame = new CudfDataFrame(table, new int[]{0}, new int[]{1}, new int[]{1},
        new int[]{1});

      DMatrix dMatrix = new DMatrix(cudfDataFrame, 0, 1);
      dMatrix.setLabel(cudfDataFrame);
      dMatrix.setWeight(cudfDataFrame);
      dMatrix.setBaseMargin(cudfDataFrame);

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
      Table table = new Table.TestBuilder()
        .column(1.2f, null, 5.2f, 7.2f, 9.2f)
        .column(0.2f, 0.4f, 0.6f, 2.6f, 0.10f)
        .column(label1)
        .column(weight1)
        .column(baseMargin1)
        .build();
      Table table1 = new Table.TestBuilder()
        .column(11.2f, 11.2f, 15.2f, 17.2f, 19.2f)
        .column(1.2f, 1.4f, null, 12.6f, 10.10f)
        .column(label2)
        .column(weight2)
        .column(baseMargin2)
        .build()) {

      List<DataFrame> tables = new LinkedList<>();

      tables.add(new CudfDataFrame(table, new int[]{0, 1}, new int[]{2}, new int[]{3}, new int[]{4}));
      tables.add(new CudfDataFrame(table1, new int[]{0, 1}, new int[]{2}, new int[]{3}, new int[]{4}));

      DMatrix dmat = new DMatrix(tables.iterator(), 0.0f, 8, 1);

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
