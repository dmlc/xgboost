/*
 Copyright (c) 2014 by Contributors

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
package ml.dmlc.xgboost4j.java.util;

/**
 * Off-heap implementation of a Dense Matrix, matrix size is only limited by the
 * amount of the available memory and the matrix dimension cannot exceed
 * Integer.MAX_VALUE (this is consistent with XGBoost API restrictions on maximum
 * length of a response).
 */
public final class BigDenseMatrix {

  private static final int FLOAT_BYTE_SIZE = 4;
  public static final long MAX_MATRIX_SIZE = Long.MAX_VALUE / FLOAT_BYTE_SIZE;

  public final int nrow;
  public final int ncol;
  public final long address;

  public static void setDirect(long valAddress, float val) {
    UtilUnsafe.UNSAFE.putFloat(valAddress, val);
  }

  public static float getDirect(long valAddress) {
    return UtilUnsafe.UNSAFE.getFloat(valAddress);
  }

  public BigDenseMatrix(int nrow, int ncol) {
    final long size = (long) nrow * ncol;
    if (size > MAX_MATRIX_SIZE) {
      throw new IllegalArgumentException("Matrix too large; matrix size cannot exceed " +
          MAX_MATRIX_SIZE);
    }
    this.nrow = nrow;
    this.ncol = ncol;
    this.address = UtilUnsafe.UNSAFE.allocateMemory(size * FLOAT_BYTE_SIZE);
  }

  public final void set(long idx, float val) {
    setDirect(address + idx * FLOAT_BYTE_SIZE, val);
  }

  public final void set(int i, int j, float val) {
    set(index(i, j), val);
  }

  public final float get(long idx) {
    return getDirect(address + idx * FLOAT_BYTE_SIZE);
  }

  public final float get(int i, int j) {
    return get(index(i, j));
  }

  public final void dispose() {
    UtilUnsafe.UNSAFE.freeMemory(address);
  }

  private long index(int i, int j) {
    return (long) i * ncol + j;
  }

}
