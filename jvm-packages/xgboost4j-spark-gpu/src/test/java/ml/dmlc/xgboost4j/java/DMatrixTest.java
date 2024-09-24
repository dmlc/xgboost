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

import java.util.*;

import ai.rapids.cudf.Table;
import junit.framework.TestCase;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test suite for DMatrix based on GPU
 */
public class DMatrixTest {

  @Test
  public void testCreateFromArrayInterfaceColumns() {
    Float[] labelFloats = new Float[]{2f, 4f, 6f, 8f, 10f};
    Integer[] groups = new Integer[]{1, 1, 7, 7, 19, 26};
    int[] expectedGroup = new int[]{0, 2, 4, 5, 6};

    Throwable ex = null;
    try (
      Table X = new Table.TestBuilder().column(1.f, null, 5.f, 7.f, 9.f).build();
      Table y = new Table.TestBuilder().column(labelFloats).build();
      Table w = new Table.TestBuilder().column(labelFloats).build();
      Table q = new Table.TestBuilder().column(groups).build();
      Table margin = new Table.TestBuilder().column(labelFloats).build();) {

      CudfColumnBatch cudfDataFrame = new CudfColumnBatch(X, y, w, null, null);

      CudfColumn labelColumn = CudfColumn.from(y.getColumn(0));
      CudfColumn weightColumn = CudfColumn.from(w.getColumn(0));
      CudfColumn baseMarginColumn = CudfColumn.from(margin.getColumn(0));
      CudfColumn qidColumn = CudfColumn.from(q.getColumn(0));

      DMatrix dMatrix = new DMatrix(cudfDataFrame, 0, 1);
      dMatrix.setLabel(labelColumn);
      dMatrix.setWeight(weightColumn);
      dMatrix.setBaseMargin(baseMarginColumn);
      dMatrix.setQueryId(qidColumn);

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
      int[] group = dMatrix.getGroup();

      TestCase.assertTrue(Arrays.equals(anchor, label));
      TestCase.assertTrue(Arrays.equals(anchor, weight));
      TestCase.assertTrue(Arrays.equals(anchor, baseMargin));
      TestCase.assertTrue(Arrays.equals(expectedGroup, group));
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
    Integer[] groups1 = new Integer[]{1, 1, 7, 7, 19, 26};

    Float[] label2 = {9f, 5f, 4f, 10f, 12f};
    Float[] weight2 = {3.0f, 1.3f, 3.2f, 0.3f, 1.34f};
    Float[] baseMargin2 = {0.2f, 2.5f, 3.1f, 4.4f, 2.2f};
    Integer[] groups2 = new Integer[]{30, 30, 30, 40, 40};

    int[] expectedGroup = new int[]{0, 2, 4, 5, 6, 9, 11};

    try (
      Table X_0 = new Table.TestBuilder()
        .column(1.2f, null, 5.2f, 7.2f, 9.2f)
        .column(0.2f, 0.4f, 0.6f, 2.6f, 0.10f)
        .build();
      Table y_0 = new Table.TestBuilder().column(label1).build();
      Table w_0 = new Table.TestBuilder().column(weight1).build();
      Table m_0 = new Table.TestBuilder().column(baseMargin1).build();
      Table q_0 = new Table.TestBuilder().column(groups1).build();

      Table X_1 = new Table.TestBuilder().column(11.2f, 11.2f, 15.2f, 17.2f, 19.2f)
        .column(1.2f, 1.4f, null, 12.6f, 10.10f).build();
      Table y_1 = new Table.TestBuilder().column(label2).build();
      Table w_1 = new Table.TestBuilder().column(weight2).build();
      Table m_1 = new Table.TestBuilder().column(baseMargin2).build();) {
      Table q_1 = new Table.TestBuilder().column(groups2).build();

      List<ColumnBatch> tables = new LinkedList<>();

      tables.add(new CudfColumnBatch(X_0, y_0, w_0, m_0, q_0));
      tables.add(new CudfColumnBatch(X_1, y_1, w_1, m_1, q_1));

      QuantileDMatrix dmat = new QuantileDMatrix(tables.iterator(), 0.0f, 256, 1);
      float[] anchorLabel = convertFloatTofloat(label1, label2);
      float[] anchorWeight = convertFloatTofloat(weight1, weight2);
      float[] anchorBaseMargin = convertFloatTofloat(baseMargin1, baseMargin2);

      TestCase.assertTrue(Arrays.equals(anchorLabel, dmat.getLabel()));
      TestCase.assertTrue(Arrays.equals(anchorWeight, dmat.getWeight()));
      TestCase.assertTrue(Arrays.equals(anchorBaseMargin, dmat.getBaseMargin()));
      TestCase.assertTrue(Arrays.equals(expectedGroup, dmat.getGroup()));
    }
  }

  private Float[] generateFloatArray(int size, long seed) {
    Float[] array = new Float[size];
    Random random = new Random(seed);
    for (int i = 0; i < size; i++) {
      array[i] = random.nextFloat();
    }
    return array;
  }

   @Test
  public void testGetQuantileCut() throws XGBoostError {

    int rows = 100;
    try (
      Table X_0 = new Table.TestBuilder()
        .column(generateFloatArray(rows, 1l))
        .column(generateFloatArray(rows, 2l))
        .column(generateFloatArray(rows, 3l))
        .column(generateFloatArray(rows, 4l))
        .column(generateFloatArray(rows, 5l))
        .build();
      Table y_0 = new Table.TestBuilder().column(generateFloatArray(rows, 6l)).build();

      Table X_1 = new Table.TestBuilder()
        .column(generateFloatArray(rows, 11l))
        .column(generateFloatArray(rows, 12l))
        .column(generateFloatArray(rows, 13l))
        .column(generateFloatArray(rows, 14l))
        .column(generateFloatArray(rows, 15l))
        .build();
      Table y_1 = new Table.TestBuilder().column(generateFloatArray(rows, 16l)).build();
    ) {
      List<ColumnBatch> tables = new LinkedList<>();
      tables.add(new CudfColumnBatch(X_0, y_0, null, null, null));
      QuantileDMatrix train = new QuantileDMatrix(tables.iterator(), 0.0f, 256, 1);

      tables.clear();
      tables.add(new CudfColumnBatch(X_1, y_1, null, null, null));
      QuantileDMatrix eval = new QuantileDMatrix(tables.iterator(),  train, 0.0f, 256, 1);

      DMatrix.QuantileCut trainCut = train.getQuantileCut();
      DMatrix.QuantileCut evalCut = eval.getQuantileCut();

      TestCase.assertTrue(trainCut.getIndptr().length == evalCut.getIndptr().length);
      TestCase.assertTrue(Arrays.equals(trainCut.getIndptr(), evalCut.getIndptr()));

      TestCase.assertTrue(trainCut.getValues().length == evalCut.getValues().length);
      TestCase.assertTrue(Arrays.equals(trainCut.getValues(), evalCut.getValues()));
    }
  }

  private float[] convertFloatTofloat(Float[]... datas) {
    int totalLength = 0;
    for (Float[] data : datas) {
      totalLength += data.length;
    }
    float[] floatArray = new float[totalLength];
    int index = 0;
    for (Float[] data : datas) {
      for (int i = 0; i < data.length; i++) {
        floatArray[i + index] = data[i];
      }
      index += data.length;
    }
    return floatArray;
  }

}
