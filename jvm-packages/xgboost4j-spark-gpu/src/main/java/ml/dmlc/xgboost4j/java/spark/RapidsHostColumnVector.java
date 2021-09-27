
/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.xgboost4j.java.spark;

import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * A GPU accelerated version of the Spark ColumnVector.
 * Most of the standard Spark APIs should never be called, as they assume that the data
 * is on the host, and we want to keep as much of the data on the device as possible.
 * We also provide GPU accelerated versions of the transitions to and from rows.
 */
public final class RapidsHostColumnVector extends ColumnVector {

  /**
   * Get the underlying host cudf columns from the batch.  This does not increment any
   * reference counts so if you want to use these columns after the batch is closed
   * you will need to do that on your own.
   */
  public static ai.rapids.cudf.HostColumnVector[] extractBases(ColumnarBatch batch) {
    int numColumns = batch.numCols();
    ai.rapids.cudf.HostColumnVector[] vectors = new ai.rapids.cudf.HostColumnVector[numColumns];
    for (int i = 0; i < vectors.length; i++) {
      vectors[i] = ((RapidsHostColumnVector)batch.column(i)).getBase();
    }
    return vectors;
  }

  /**
   * Get the underlying spark compatible host columns from the batch.  This does not increment any
   * reference counts so if you want to use these columns after the batch is closed
   * you will need to do that on your own.
   */
  public static RapidsHostColumnVector[] extractColumns(ColumnarBatch batch) {
    int numColumns = batch.numCols();
    RapidsHostColumnVector[] vectors = new RapidsHostColumnVector[numColumns];

    for (int i = 0; i < vectors.length; i++) {
      vectors[i] = ((RapidsHostColumnVector)batch.column(i));
    }
    return vectors;
  }


  private final ai.rapids.cudf.HostColumnVector cudfCv;

  /**
   * Sets up the data type of this column vector.
   */
  RapidsHostColumnVector(DataType type, ai.rapids.cudf.HostColumnVector cudfCv) {
    super(type);
    // TODO need some checks to be sure everything matches
    this.cudfCv = cudfCv;
  }

  public RapidsHostColumnVector incRefCount() {
    // Just pass through the reference counting
    cudfCv.incRefCount();
    return this;
  }

  @Override
  public void close() {
    // Just pass through the reference counting
    cudfCv.close();
  }

  @Override
  public boolean hasNull() {
    return cudfCv.hasNulls();
  }

  @Override
  public int numNulls() {
    return (int) cudfCv.getNullCount();
  }

  @Override
  public boolean isNullAt(int rowId) {
    return cudfCv.isNull(rowId);
  }

  @Override
  public boolean getBoolean(int rowId) {
    return cudfCv.getBoolean(rowId);
  }

  @Override
  public byte getByte(int rowId) {
    return cudfCv.getByte(rowId);
  }

  @Override
  public short getShort(int rowId) {
    return cudfCv.getShort(rowId);
  }

  @Override
  public int getInt(int rowId) {
    return cudfCv.getInt(rowId);
  }

  @Override
  public long getLong(int rowId) {
    return cudfCv.getLong(rowId);
  }

  @Override
  public float getFloat(int rowId) {
    return cudfCv.getFloat(rowId);
  }

  @Override
  public double getDouble(int rowId) {
    return cudfCv.getDouble(rowId);
  }

  @Override
  public ColumnarArray getArray(int rowId) {
    throw new IllegalStateException("Arrays are currently not supported by rapids cudf");
  }

  @Override
  public ColumnarMap getMap(int ordinal) {
    throw new IllegalStateException("Maps are currently not supported by rapids cudf");
  }

  @Override
  public Decimal getDecimal(int rowId, int precision, int scale) {
    throw new IllegalStateException("The decimal type is currently not supported by rapids cudf");
  }

  @Override
  public UTF8String getUTF8String(int rowId) {
    // TODO need a cheaper way to go directly to the String
    return UTF8String.fromString(cudfCv.getJavaString(rowId));
  }

  @Override
  public byte[] getBinary(int rowId) {
    throw new IllegalStateException("Binary data access is currently not supported by rapids cudf");
  }

  @Override
  public ColumnVector getChild(int ordinal) {
    throw new IllegalStateException("Struct and struct like types are currently " +
      "not supported by rapids cudf");
  }

  public ai.rapids.cudf.HostColumnVector getBase() {
    return cudfCv;
  }

  public long getRowCount() { return cudfCv.getRowCount(); }

  public GpuColumnVector copyToDevice() {
    return new GpuColumnVector(type, cudfCv.copyToDevice());
  }
}
