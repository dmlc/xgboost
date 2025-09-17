/*
 Copyright (c) 2014-2024 by Contributors

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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import junit.framework.TestCase;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.util.BigDenseMatrix;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * test cases for DMatrix
 *
 * @author hzx
 */
public class DMatrixTest {


  @Test
  public void testCreateFromDataIteratorWithMissingValue() throws XGBoostError {
    //create DMatrix from DataIterator
    java.util.List<LabeledPoint> blist = new java.util.LinkedList<>();
    blist.add(new LabeledPoint(0.1f, 4, null, new float[]{1, 0, 0, 0}));
    blist.add(new LabeledPoint(0.1f, 4, null, new float[]{Float.NaN, 13, 14, 15}));
    blist.add(new LabeledPoint(0.1f, 4, null, new float[]{21, 23, 0, 25}));

    // Default missing value: Float.NaN
    DMatrix dmat = new DMatrix(blist.iterator(), null);
    assert dmat.nonMissingNum() == 11;

    // missing value 0
    dmat = new DMatrix(blist.iterator(), null, 0.0f);
    assert dmat.nonMissingNum() == 12 - 4 - 1;

    // missing value 21
    dmat = new DMatrix(blist.iterator(), null, 21.0f);
    assert dmat.nonMissingNum() == 12 - 1 - 1;

    // missing value 101010101010
    dmat = new DMatrix(blist.iterator(), null, 101010101010.0f);
    assert dmat.nonMissingNum() == 12 - 1;
  }

  @Test
  public void testCreateFromDataIterator() throws XGBoostError {
    //create DMatrix from DataIterator

    java.util.ArrayList<Float> labelall = new java.util.ArrayList<Float>();
    int nrep = 3000;
    java.util.List<LabeledPoint> blist = new java.util.LinkedList<LabeledPoint>();
    for (int i = 0; i < nrep; ++i) {
      LabeledPoint p = new LabeledPoint(
        0.1f + i, 4, new int[]{0, 2, 3}, new float[]{3, 4, 5});
      blist.add(p);
      labelall.add(p.label());
    }
    DMatrix dmat = new DMatrix(blist.iterator(), null);
    // get label
    float[] labels = dmat.getLabel();
    for (int i = 0; i < labels.length; ++i) {
      TestCase.assertTrue(labelall.get(i) == labels[i]);
    }
  }

  @Test
  public void testCreateFromDataIteratorWithDiffFeatureSize() throws XGBoostError {
    //create DMatrix from DataIterator

    java.util.ArrayList<Float> labelall = new java.util.ArrayList<Float>();
    int nrep = 3000;
    java.util.List<LabeledPoint> blist = new java.util.LinkedList<LabeledPoint>();
    int featureSize = 4;
    for (int i = 0; i < nrep; ++i) {
      // set some rows with wrong feature size
      if (i % 10 == 1) {
        featureSize = 5;
      }
      LabeledPoint p = new LabeledPoint(
        0.1f + i, featureSize, new int[]{0, 2, 3}, new float[]{3, 4, 5});
      blist.add(p);
      labelall.add(p.label());
    }
    boolean success = true;
    try {
      DMatrix dmat = new DMatrix(blist.iterator(), null);
    } catch (XGBoostError e) {
      success = false;
    }
    TestCase.assertTrue(success == false);
  }

  @Test
  public void testCreateFromFile() throws XGBoostError {
    //create DMatrix from file
    String filePath = writeResourceIntoTempFile("/agaricus.txt.test");
    DMatrix dmat = new DMatrix(filePath + "?format=libsvm");
    //get label
    float[] labels = dmat.getLabel();
    //check length
    TestCase.assertTrue(dmat.rowNum() == labels.length);
    //set weights
    float[] weights = Arrays.copyOf(labels, labels.length);
    dmat.setWeight(weights);
    float[] dweights = dmat.getWeight();
    TestCase.assertTrue(Arrays.equals(weights, dweights));
  }

  @Test
  public void testCreateFromCSR() throws XGBoostError {
    //create Matrix from csr format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2 3 0
     * 4 0 2 3 5
     * 3 1 2 5 0
     */
    float[] data = new float[]{1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5};
    int[] colIndex = new int[]{0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3};
    long[] rowHeaders = new long[]{0, 3, 7, 11};
    DMatrix dmat1 = new DMatrix(rowHeaders, colIndex, data, DMatrix.SparseType.CSR);
    //check row num
    TestCase.assertTrue(dmat1.rowNum() == 3);
    //test set label
    float[] label1 = new float[]{1, 0, 1};
    dmat1.setLabel(label1);
    float[] label2 = dmat1.getLabel();
    TestCase.assertTrue(Arrays.equals(label1, label2));
  }

  @Test
  public void testCreateFromCSREx() throws XGBoostError {
    //create Matrix from csr format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2 3 0
     * 4 0 2 3 5
     * 3 1 2 5 0
     */
    float[] data = new float[]{1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5};
    int[] colIndex = new int[]{0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3};
    long[] rowHeaders = new long[]{0, 3, 7, 11};
    DMatrix dmat1 = new DMatrix(rowHeaders, colIndex, data, DMatrix.SparseType.CSR, 5);
    //check row num
    TestCase.assertTrue(dmat1.rowNum() == 3);
    //test set label
    float[] label1 = new float[]{1, 0, 1};
    dmat1.setLabel(label1);
    float[] label2 = dmat1.getLabel();
    TestCase.assertTrue(Arrays.equals(label1, label2));
  }

  @Test
  public void testCreateFromCSC() throws XGBoostError {
    //create Matrix from csc format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2
     * 3 0 4
     * 0 2 3
     * 5 3 1
     * 2 5 0
     */
    float[] data = new float[]{1, 3, 5, 2, 2, 3, 5, 2, 4, 3, 1};
    int[] rowIndex = new int[]{0, 1, 3, 4, 2, 3, 4, 0, 1, 2, 3};
    long[] colHeaders = new long[]{0, 4, 7, 11};
    DMatrix dmat1 = new DMatrix(colHeaders, rowIndex, data, DMatrix.SparseType.CSC);
    //check row num
    System.out.println(dmat1.rowNum());
    TestCase.assertTrue(dmat1.rowNum() == 5);
    //test set label
    float[] label1 = new float[]{1, 0, 1, 1, 1};
    dmat1.setLabel(label1);
    float[] label2 = dmat1.getLabel();
    TestCase.assertTrue(Arrays.equals(label1, label2));
  }

  @Test
  public void testCreateFromCSCEx() throws XGBoostError {
    //create Matrix from csc format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2
     * 3 0 4
     * 0 2 3
     * 5 3 1
     * 2 5 0
     */
    float[] data = new float[]{1, 3, 5, 2, 2, 3, 5, 2, 4, 3, 1};
    int[] rowIndex = new int[]{0, 1, 3, 4, 2, 3, 4, 0, 1, 2, 3};
    long[] colHeaders = new long[]{0, 4, 7, 11};
    DMatrix dmat1 = new DMatrix(colHeaders, rowIndex, data, DMatrix.SparseType.CSC, 5);
    //check row num
    System.out.println(dmat1.rowNum());
    TestCase.assertTrue(dmat1.rowNum() == 5);
    //test set label
    float[] label1 = new float[]{1, 0, 1, 1, 1};
    dmat1.setLabel(label1);
    float[] label2 = dmat1.getLabel();
    TestCase.assertTrue(Arrays.equals(label1, label2));
  }

  @Test
  public void testCreateFromDenseMatrix() throws XGBoostError {
    //create DMatrix from 10*5 dense matrix
    int nrow = 10;
    int ncol = 5;
    float[] data0 = new float[nrow * ncol];
    //put random nums
    Random random = new Random();
    for (int i = 0; i < nrow * ncol; i++) {
      data0[i] = random.nextFloat();
    }

    //create label
    float[] label0 = new float[nrow];
    for (int i = 0; i < nrow; i++) {
      label0[i] = random.nextFloat();
    }

    DMatrix dmat0 = new DMatrix(data0, nrow, ncol, Float.NaN);
    dmat0.setLabel(label0);

    //check
    TestCase.assertTrue(dmat0.rowNum() == 10);
    TestCase.assertTrue(dmat0.getLabel().length == 10);

    //set weights for each instance
    float[] weights = new float[nrow];
    for (int i = 0; i < nrow; i++) {
      weights[i] = random.nextFloat();
    }
    dmat0.setWeight(weights);

    TestCase.assertTrue(Arrays.equals(weights, dmat0.getWeight()));
  }

  private DMatrix createFromDenseMatrix() throws XGBoostError {
    //create DMatrix from 10*5 dense matrix
    int nrow = 10;
    int ncol = 5;
    float[] data0 = new float[nrow * ncol];
    //put random nums
    Random random = new Random();
    for (int i = 0; i < nrow * ncol; i++) {
      if (i % 10 == 0) {
        data0[i] = -0.1f;
      } else {
        data0[i] = random.nextFloat();
      }
    }

    //create label
    float[] label0 = new float[nrow];
    for (int i = 0; i < nrow; i++) {
      label0[i] = random.nextFloat();
    }

    DMatrix dm = new DMatrix(data0, nrow, ncol, -0.1f);
    dm.setLabel(label0);
    return dm;
  }

  @Test
  public void testCreateFromDenseMatrixWithMissingValue() throws XGBoostError {
    DMatrix dm = createFromDenseMatrix();
    //check
    TestCase.assertTrue(dm.rowNum() == 10);
    TestCase.assertTrue(dm.getLabel().length == 10);
  }

  @Test
  public void testCreateFromDenseMatrixRef() throws XGBoostError {
    //create DMatrix from 10*5 dense matrix
    final int nrow = 10;
    final int ncol = 5;

    DMatrix dmat0 = null;
    BigDenseMatrix data0 = null;
    try {
      data0 = new BigDenseMatrix(nrow, ncol);
      //put random nums
      Random random = new Random();
      for (int i = 0; i < nrow * ncol; i++) {
        data0.set(i, random.nextFloat());
      }

      //create label
      float[] label0 = new float[nrow];
      for (int i = 0; i < nrow; i++) {
        label0[i] = random.nextFloat();
      }

      dmat0 = new DMatrix(data0, Float.NaN);
      dmat0.setLabel(label0);

      //check
      TestCase.assertTrue(dmat0.rowNum() == 10);
      TestCase.assertTrue(dmat0.getLabel().length == 10);
    } finally {
      if (dmat0 != null) {
        dmat0.dispose();
      } else if (data0 != null) {
        data0.dispose();
      }
    }
  }

  @Test
  public void testTrainWithDenseMatrixRef() throws XGBoostError {
    Map<String, Object> rabitEnv = new HashMap<>();
    rabitEnv.put("DMLC_TASK_ID", "0");
    Communicator.init(rabitEnv);
    DMatrix trainMat = null;
    BigDenseMatrix data0 = null;
    try {
      // trivial dataset with 3 rows and 2 columns
      // (4,5) -> 1
      // (3,1) -> 2
      // (2,3) -> 3
      float[][] data = new float[][]{
        new float[]{4f, 5f},
        new float[]{3f, 1f},
        new float[]{2f, 3f}
      };
      data0 = new BigDenseMatrix(3, 2);
      for (int i = 0; i < data0.nrow; i++)
        for (int j = 0; j < data0.ncol; j++)
          data0.set(i, j, data[i][j]);

      trainMat = new DMatrix(data0, Float.NaN);
      trainMat.setLabel(new float[]{1f, 2f, 3f});

      HashMap<String, Object> params = new HashMap<>();
      params.put("eta", 1);
      params.put("max_depth", 5);
      params.put("silent", 1);
      params.put("objective", "reg:linear");
      params.put("seed", 123);

      HashMap<String, DMatrix> watches = new HashMap<>();
      watches.put("train", trainMat);

      Booster booster = XGBoost.train(trainMat, params, 10, watches, null, null);

      // check overfitting
      // (4,5) -> 1
      // (3,1) -> 2
      // (2,3) -> 3
      for (int i = 0; i < 3; i++) {
        float[][] preds = booster.predict(new DMatrix(data[i], 1, 2, Float.NaN));
        assertEquals(1, preds.length);
        assertArrayEquals(new float[]{(float) (i + 1)}, preds[0], 1e-2f);
      }
    } finally {
      if (trainMat != null)
        trainMat.dispose();
      else if (data0 != null) {
        data0.dispose();
      }
      Communicator.shutdown();
    }
  }

  private String writeResourceIntoTempFile(String resource) {
    InputStream input = getClass().getResourceAsStream(resource);
    if (input == null) {
      throw new IllegalArgumentException("Resource " + resource + " does not exist.");
    }
    File tmp;
    try {
      tmp = File.createTempFile("junit", ".test");
    } catch (IOException e) {
      throw new RuntimeException("Unable to write to temp file.", e);
    }
    byte[] buff = new byte[1024];
    try (FileOutputStream output = new FileOutputStream(tmp)) {
      int n;
      while ((n = input.read(buff)) > 0) {
        output.write(buff, 0, n);
      }
    } catch (IOException e) {
      throw new RuntimeException("Unable to write to temp file.", e);
    }
    return tmp.getAbsolutePath();
  }

  @Test
  public void testSetAndGetGroup() throws XGBoostError {
    //create DMatrix from 10*5 dense matrix
    int nrow = 10;
    int ncol = 5;
    float[] data0 = new float[nrow * ncol];
    //put random nums
    Random random = new Random();
    for (int i = 0; i < nrow * ncol; i++) {
      data0[i] = random.nextFloat();
    }

    //create label
    float[] label0 = new float[nrow];
    for (int i = 0; i < nrow; i++) {
      label0[i] = random.nextFloat();
    }

    //create two groups
    int[] groups = new int[]{5, 5};

    DMatrix dmat0 = new DMatrix(data0, nrow, ncol, -0.1f);
    dmat0.setLabel(label0);
    dmat0.setGroup(groups);

    //check
    TestCase.assertTrue(Arrays.equals(new int[]{0, 5, 10}, dmat0.getGroup()));
  }

  @Test
  public void testSetAndGetFeatureInfo() throws XGBoostError {
    //create DMatrix from 10*5 dense matrix
    int nrow = 10;
    int ncol = 5;
    float[] data = new float[nrow * ncol];
    //put random nums
    Random random = new Random();
    for (int i = 0; i < nrow * ncol; i++) {
      data[i] = random.nextInt();
    }

    DMatrix dmat = new DMatrix(data, nrow, ncol, Float.NaN);

    String[] featureNames = new String[]{"f1", "f2", "f3", "f4", "f5"};
    dmat.setFeatureNames(featureNames);
    String[] retFeatureNames = dmat.getFeatureNames();
    assertArrayEquals(featureNames, retFeatureNames);

    String[] featureTypes = new String[]{"i", "q", "c", "i", "q"};
    dmat.setFeatureTypes(featureTypes);
    String[] retFeatureTypes = dmat.getFeatureTypes();
    assertArrayEquals(featureTypes, retFeatureTypes);
  }

  @Test
  public void testSetAndGetQueryId() throws XGBoostError {
    //create DMatrix from 10*5 dense matrix
    int nrow = 10;
    int ncol = 5;
    float[] data0 = new float[nrow * ncol];
    //put random nums
    Random random = new Random();
    for (int i = 0; i < nrow * ncol; i++) {
      data0[i] = random.nextFloat();
    }

    //create label
    float[] label0 = new float[nrow];
    for (int i = 0; i < nrow; i++) {
      label0[i] = random.nextFloat();
    }

    //create two groups
    int[] qid = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int[] qidExpected = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    DMatrix dmat0 = new DMatrix(data0, nrow, ncol, -0.1f);
    dmat0.setLabel(label0);
    dmat0.setQueryId(qid);
    //check
    TestCase.assertTrue(Arrays.equals(qidExpected, dmat0.getGroup()));

    //create two groups
    int[] qid1 = new int[]{10, 10, 10, 20, 60, 60, 80, 80, 90, 100};
    int[] qidExpected1 = new int[]{0, 3, 4, 6, 8, 9, 10};
    dmat0.setQueryId(qid1);
    TestCase.assertTrue(Arrays.equals(qidExpected1, dmat0.getGroup()));

  }

  @Test
  public void getGetQuantileCut() throws XGBoostError {
    DMatrix Xy = createFromDenseMatrix();
    Map<String, Object> params = new HashMap<String, Object>();
    HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
    watches.put("train", Xy);
    XGBoost.train(Xy, params, 1, watches, null, null); // Create the cuts
    DMatrix.QuantileCut cuts = Xy.getQuantileCut();
    TestCase.assertEquals(cuts.indptr.length, 6);
    for (int i = 1; i < cuts.indptr.length; ++i) {
      // Number of bins for each feature + min value.
      TestCase.assertTrue(cuts.indptr[i] - cuts.indptr[i - 1] >= 5);
      TestCase.assertTrue(cuts.indptr[i] - cuts.indptr[i - 1] <= Xy.rowNum() + 1);
    }
    TestCase.assertEquals(cuts.values.length, cuts.indptr[cuts.indptr.length - 1]);
    for (int i = 1; i < cuts.indptr.length; ++i) {
      long begin = cuts.indptr[i - 1];
      long end = cuts.indptr[i];
      for (long j = begin + 1; j < end; ++j) {
        TestCase.assertTrue(cuts.values[(int) j] > cuts.values[(int) j - 1]);
      }
    }
  }
}
