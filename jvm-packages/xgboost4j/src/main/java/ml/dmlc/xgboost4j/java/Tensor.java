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

package ml.dmlc.xgboost4j.java;

import java.util.ArrayList;
import java.util.List;

/**
 * A Tensor to hold the prediction result.
 */
public class Tensor {
  // Dimension of Tensor
  private final long dim;
  // Shape of Tensor
  private final long[] shape;
  // The raw result predicted by XGBoost
  private final float[] rawResult;
  // The List type result for Java
  private List resultList;
  // The Array type result for Java
  private Object resultArray;

  public Tensor(long dim, long[] shape, float[] result) {
    this.dim = dim;
    this.shape = shape;
    this.rawResult = result;
  }

  /**
   * Get the dimension of Tensor
   */
  public long getDim() {
    return dim;
  }

  /**
   * Get the shape of Tensor
   */
  public long[] getShape() {
    return shape;
  }

  /**
   * Get the raw prediction result
   */
  public float[] getRawResult() {
    return rawResult;
  }

  /**
   * Convert the raw result to the List type
   */
  public synchronized List getResultList() {
    if (resultList == null) {
      // the following implementation needs to be improved
      resultList = new ArrayList();
      for (int i = 0; i < shape[0]; i++) {
        if (dim == 1) {
          resultList.add(rawResult[i]);
          continue;
        }

        ArrayList list1 = new ArrayList();
        for (int j = 0; j < shape[1]; j++) {
          if (dim == 2) {
            int index = (int) (i * shape[1] + j);
            list1.add(rawResult[index]);
            continue;
          }

          ArrayList list2 = new ArrayList();
          for (int k = 0; k < shape[2]; k++) {
            if (dim == 3) {
              int index = (int) (i * shape[1] + j * shape[2] + k);
              list2.add(rawResult[index]);
              continue;
            }

            ArrayList list3 = new ArrayList();
            for (int m = 0; m < shape[3]; m++) {
              int index = (int) (i * shape[1] + j * shape[2] + k * shape[3] + m);
              list3.add(rawResult[index]);
            }
            list2.add(list3);
          }
          list1.add(list2);
        }
        resultList.add(list1);
      }
    }
    return resultList;
  }

  public synchronized Object getResultArray() {
    if (resultArray == null) {
      if (dim == 1 || dim == 2) {
        int y = 1;
        if (dim == 2) {
          y = (int) shape[1];
        }
        int x = (int) shape[0];
        float[][] ret = new float[x][y];
        for (int i = 0; i < x; i++) {
          for (int j = 0; j < y; j++) {
            ret[i][j] = rawResult[i * y + j];
          }
        }
        resultArray = ret;
      } else if (dim == 3) {
        int x = (int) shape[0];
        int y = (int) shape[1];
        int z = (int) shape[2];
        float[][][] ret = new float[x][y][z];
        for (int i = 0; i < x; i++) {
          for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
              ret[i][j][k] = rawResult[i * y + j * z + k];
            }
          }
        }
        resultArray = ret;
      } else if (dim == 4) {
        int x = (int) shape[0];
        int y = (int) shape[1];
        int z = (int) shape[2];
        int t = (int) shape[3];
        float[][][][] ret = new float[x][y][z][t];
        for (int i = 0; i < x; i++) {
          for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
              for (int m = 0; m < z; m++) {
                ret[i][j][k][m] = rawResult[i * y + j * z + k * t + m];
              }
            }
            resultArray = ret;
          }
        }
      }
    }
    return resultArray;
  }
}
