/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import java.util.Arrays;
import java.util.List;

import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;

import org.apache.spark.sql.catalyst.expressions.Attribute;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;


/**
 * A GPU accelerated version of the Spark ColumnVector.
 * Most of the standard Spark APIs should never be called, as they assume that the data
 * is on the host, and we want to keep as much of the data on the device as possible.
 * We also provide GPU accelerated versions of the transitions to and from rows.
 */
public class GpuColumnVector extends GpuColumnVectorBase {

  private static HostColumnVector.DataType convertFrom(DataType spark, boolean nullable) {
    if (spark instanceof ArrayType) {
      ArrayType arrayType = (ArrayType) spark;
      return new HostColumnVector.ListType(nullable,
        convertFrom(arrayType.elementType(), arrayType.containsNull()));
    } else if (spark instanceof MapType) {
      MapType mapType = (MapType) spark;
      return new HostColumnVector.ListType(nullable,
        new HostColumnVector.StructType(false, Arrays.asList(
          convertFrom(mapType.keyType(), false),
          convertFrom(mapType.valueType(), mapType.valueContainsNull())
        )));
    } else if (spark instanceof StructType) {
      StructType stType = (StructType) spark;
      HostColumnVector.DataType[] children = new HostColumnVector.DataType[stType.size()];
      StructField[] fields = stType.fields();
      for (int i = 0; i < children.length; i++) {
        children[i] = convertFrom(fields[i].dataType(), fields[i].nullable());
      }
      return new HostColumnVector.StructType(nullable, children);
    } else {
      // Only works for basic types
      return new HostColumnVector.BasicType(nullable, getNonNestedRapidsType(spark));
    }
  }

  public static final class GpuColumnarBatchBuilder implements AutoCloseable {
    private final HostColumnVector.ColumnBuilder[] builders;
    private final StructField[] fields;

    /**
     * A collection of builders for building up columnar data.
     * @param schema the schema of the batch.
     * @param rows the maximum number of rows in this batch.
     * @param batch if this is going to copy a ColumnarBatch in a non GPU format that batch
     *              we are going to copy. If not this may be null. This is used to get an idea
     *              of how big to allocate buffers that do not necessarily correspond to the
     *              number of rows.
     */
    public GpuColumnarBatchBuilder(StructType schema, int rows, ColumnarBatch batch) {
      fields = schema.fields();
      int len = fields.length;
      builders = new HostColumnVector.ColumnBuilder[len];
      boolean success = false;
      try {
        for (int i = 0; i < len; i++) {
          StructField field = fields[i];
          builders[i] = new HostColumnVector.ColumnBuilder(
            convertFrom(field.dataType(), field.nullable()), rows);
        }
        success = true;
      } finally {
        if (!success) {
          for (HostColumnVector.ColumnBuilder b: builders) {
            if (b != null) {
              b.close();
            }
          }
        }
      }
    }

    public HostColumnVector.ColumnBuilder builder(int i) {
      return builders[i];
    }

    public ColumnarBatch build(int rows) {
      ColumnVector[] vectors = new ColumnVector[builders.length];
      boolean success = false;
      try {
        for (int i = 0; i < builders.length; i++) {
          ai.rapids.cudf.ColumnVector cv = builders[i].buildAndPutOnDevice();
          vectors[i] = new GpuColumnVector(fields[i].dataType(), cv);
          builders[i] = null;
        }
        ColumnarBatch ret = new ColumnarBatch(vectors, rows);
        success = true;
        return ret;
      } finally {
        if (!success) {
          for (ColumnVector vec: vectors) {
            if (vec != null) {
              vec.close();
            }
          }
        }
      }
    }

    public HostColumnVector[] buildHostColumns() {
      HostColumnVector[] vectors = new HostColumnVector[builders.length];
      try {
        for (int i = 0; i < builders.length; i++) {
          vectors[i] = builders[i].build();
          builders[i] = null;
        }
        HostColumnVector[] result = vectors;
        vectors = null;
        return result;
      } finally {
        if (vectors != null) {
          for (HostColumnVector v : vectors) {
            if (v != null) {
              v.close();
            }
          }
        }
      }
    }

    @Override
    public void close() {
      for (HostColumnVector.ColumnBuilder b: builders) {
        if (b != null) {
          b.close();
        }
      }
    }
  }

  private static DType toRapidsOrNull(DataType type) {
    if (type instanceof LongType) {
      return DType.INT64;
    } else if (type instanceof DoubleType) {
      return DType.FLOAT64;
    } else if (type instanceof ByteType) {
      return DType.INT8;
    } else if (type instanceof BooleanType) {
      return DType.BOOL8;
    } else if (type instanceof ShortType) {
      return DType.INT16;
    } else if (type instanceof IntegerType) {
      return DType.INT32;
    } else if (type instanceof FloatType) {
      return DType.FLOAT32;
    } else if (type instanceof DateType) {
      return DType.TIMESTAMP_DAYS;
    } else if (type instanceof TimestampType) {
      return DType.TIMESTAMP_MICROSECONDS;
    } else if (type instanceof StringType) {
      return DType.STRING;
    } else if (type instanceof NullType) {
      // INT8 is used for both in this case
      return DType.INT8;
    } else if (type instanceof DecimalType) {
      // Decimal supportable check has been conducted in the GPU plan overriding stage.
      // So, we don't have to handle decimal-supportable problem at here.
      DecimalType dt = (DecimalType) type;
      if (dt.precision() > DType.DECIMAL64_MAX_PRECISION) {
        return null;
      } else {
        // Map all DecimalType to DECIMAL64, in case of underlying DType transaction.
        return DType.create(DType.DTypeEnum.DECIMAL64, -dt.scale());
      }
    }
    return null;
  }

  public static boolean isNonNestedSupportedType(DataType type) {
    return toRapidsOrNull(type) != null;
  }

  public static DType getNonNestedRapidsType(DataType type) {
    DType result = toRapidsOrNull(type);
    if (result == null) {
      throw new IllegalArgumentException(type + " is not supported for GPU processing yet.");
    }
    return result;
  }

  /**
   * Create an empty batch from the given format.  This should be used very sparingly because
   * returning an empty batch from an operator is almost always the wrong thing to do.
   */
  public static ColumnarBatch emptyBatch(StructType schema) {
    try (GpuColumnarBatchBuilder builder = new GpuColumnarBatchBuilder(schema, 0, null)) {
      return builder.build(0);
    }
  }

  /**
   * Create an empty batch from the given format.  This should be used very sparingly because
   * returning an empty batch from an operator is almost always the wrong thing to do.
   */
  public static ColumnarBatch emptyBatch(List<Attribute> format) {
    return emptyBatch(structFromAttributes(format));
  }


  /**
   * Create empty host column vectors from the given format.  This should only be necessary
   * when serializing an empty broadcast table.
   */
  public static HostColumnVector[] emptyHostColumns(StructType schema) {
    try (GpuColumnarBatchBuilder builder = new GpuColumnarBatchBuilder(schema, 0, null)) {
      return builder.buildHostColumns();
    }
  }

  /**
   * Create empty host column vectors from the given format.  This should only be necessary
   * when serializing an empty broadcast table.
   */
  public static HostColumnVector[] emptyHostColumns(List<Attribute> format) {
    return emptyHostColumns(structFromAttributes(format));
  }

  private static StructType structFromAttributes(List<Attribute> format) {
    StructField[] fields = new StructField[format.size()];
    int i = 0;
    for (Attribute attribute: format) {
      fields[i++] = new StructField(
        attribute.name(),
        attribute.dataType(),
        attribute.nullable(),
        null);
    }
    return new StructType(fields);
  }

  /**
   * Convert a Spark schema into a cudf schema
   * @param input the Spark schema to convert
   * @return the cudf schema
   */
  public static Schema from(StructType input) {
    Schema.Builder builder = Schema.builder();
    input.foreach(f -> builder.column(
        GpuColumnVector.getNonNestedRapidsType(f.dataType()), f.name()));
    return builder.build();
  }

  /**
   * Convert a ColumnarBatch to a table. The table will increment the reference count for all of
   * the columns in the batch, so you will need to close both the batch passed in and the table
   * returned to avoid any memory leaks.
   */
  public static Table from(ColumnarBatch batch) {
    return new Table(extractBases(batch));
  }

  /**
   * Get the data types for a batch.
   */
  public static DataType[] extractTypes(ColumnarBatch batch) {
    DataType[] ret = new DataType[batch.numCols()];
    for (int i = 0; i < batch.numCols(); i++) {
      ret[i] = batch.column(i).dataType();
    }
    return ret;
  }

  /**
   * Get the data types for a struct.
   */
  public static DataType[] extractTypes(StructType st) {
    DataType[] ret = new DataType[st.size()];
    for (int i = 0; i < st.size(); i++) {
      ret[i] = st.apply(i).dataType();
    }
    return ret;
  }

  /**
   * Convert a Table to a ColumnarBatch.  The columns in the table will have their reference counts
   * incremented so you will need to close both the table passed in and the batch returned to
   * not have any leaks.
   * @param colTypes the types of the columns that should be returned.
   */
  public static ColumnarBatch from(Table table, DataType[] colTypes) {
    return from(table, colTypes, 0, table.getNumberOfColumns());
  }

  /**
   * This should only ever be called from an assertion.
   */
  private static boolean typeConversionAllowed(ColumnView cv, DataType colType) {
    DType dt = cv.getType();
    // Only supports DECIMAL64, in case of DType transaction due to precision change.
    if (dt.isDecimalType() && dt.isBackedByLong()) {
      if (!(colType instanceof DecimalType)) {
        return false;
      }
      // check for overflow
      return ((DecimalType) colType).precision() <= DType.DECIMAL64_MAX_PRECISION;
    }
    if (!dt.isNestedType()) {
      return getNonNestedRapidsType(colType).equals(dt);
    }
    if (colType instanceof MapType) {
      MapType mType = (MapType) colType;
      // list of struct of key/value
      if (!(dt.equals(DType.LIST))) {
        return false;
      }
      try (ColumnView structCv = cv.getChildColumnView(0)) {
        if (!(structCv.getType().equals(DType.STRUCT))) {
          return false;
        }
        if (structCv.getNumChildren() != 2) {
          return false;
        }
        try (ColumnView keyCv = structCv.getChildColumnView(0)) {
          if (!typeConversionAllowed(keyCv, mType.keyType())) {
            return false;
          }
        }
        try (ColumnView valCv = structCv.getChildColumnView(1)) {
          return typeConversionAllowed(valCv, mType.valueType());
        }
      }
    } else if (colType instanceof ArrayType) {
      if (!(dt.equals(DType.LIST))) {
        return false;
      }
      try (ColumnView tmp = cv.getChildColumnView(0)) {
        return typeConversionAllowed(tmp, ((ArrayType) colType).elementType());
      }
    } else if (colType instanceof StructType) {
      if (!(dt.equals(DType.STRUCT))) {
        return false;
      }
      StructType st = (StructType) colType;
      final int numChildren = cv.getNumChildren();
      if (numChildren != st.size()) {
        return false;
      }
      for (int childIndex = 0; childIndex < numChildren; childIndex++) {
        try (ColumnView tmp = cv.getChildColumnView(childIndex)) {
          StructField entry = ((StructType) colType).apply(childIndex);
          if (!typeConversionAllowed(tmp, entry.dataType())) {
            return false;
          }
        }
      }
      return true;
    } else if (colType instanceof BinaryType) {
      if (!(dt.equals(DType.LIST))) {
        return false;
      }
      try (ColumnView tmp = cv.getChildColumnView(0)) {
        DType tmpType = tmp.getType();
        return tmpType.equals(DType.INT8) || tmpType.equals(DType.UINT8);
      }
    } else {
      // Unexpected type
      return false;
    }
  }

  /**
   * This should only ever be called from an assertion. This is to avoid the performance overhead
   * of doing the complicated check in production.  Sadly this means that we don't get to give a
   * clear message about what part of the check failed, so the assertions that use this should
   * include in the message both types so a user can see what is different about them.
   */
  static boolean typeConversionAllowed(Table table, DataType[] colTypes) {
    final int numColumns = table.getNumberOfColumns();
    if (numColumns != colTypes.length) {
      return false;
    }
    boolean ret = true;
    for (int colIndex = 0; colIndex < numColumns; colIndex++) {
      ret = ret && typeConversionAllowed(table.getColumn(colIndex), colTypes[colIndex]);
    }
    return ret;
  }

  /**
   * Get a ColumnarBatch from a set of columns in the Table. This gets the columns
   * starting at startColIndex and going until but not including untilColIndex. This will
   * increment the reference count for all columns converted so you will need to close
   * both the table that is passed in and the batch returned to be sure that there are no leaks.
   *
   * @param table a table of vectors
   * @param colTypes List of the column data types in the table passed in
   * @param startColIndex index of the first vector you want in the final ColumnarBatch
   * @param untilColIndex until index of the columns. (ie doesn't include that column num)
   * @return a ColumnarBatch of the vectors from the table
   */
  public static ColumnarBatch from(Table table, DataType[] colTypes, int startColIndex,
                                   int untilColIndex) {
    assert table != null : "Table cannot be null";
    assert typeConversionAllowed(table, colTypes) : "Type conversion is not allowed from " + table +
      " to " + Arrays.toString(colTypes);
    int numColumns = untilColIndex - startColIndex;
    ColumnVector[] columns = new ColumnVector[numColumns];
    int finalLoc = 0;
    boolean success = false;
    try {
      for (int i = startColIndex; i < untilColIndex; i++) {
        columns[finalLoc] = from(table.getColumn(i).incRefCount(), colTypes[i]);
        finalLoc++;
      }
      long rows = table.getRowCount();
      if (rows != (int) rows) {
        throw new IllegalStateException("Cannot support a batch larger that MAX INT rows");
      }
      ColumnarBatch ret = new ColumnarBatch(columns, (int)rows);
      success = true;
      return ret;
    } finally {
      if (!success) {
        for (ColumnVector cv: columns) {
          if (cv != null) {
            cv.close();
          }
        }
      }
    }
  }

  /**
   * Converts a cudf internal vector to a Spark compatible vector. No reference counts
   * are incremented so you need to either close the returned value or the input value,
   * but not both.
   */
  public static GpuColumnVector from(ai.rapids.cudf.ColumnVector cudfCv, DataType type) {
    assert typeConversionAllowed(cudfCv, type) : "Type conversion is not allowed from " + cudfCv +
      " to " + type;
    return new GpuColumnVector(type, cudfCv);
  }

  public static GpuColumnVector from(Scalar scalar, int count, DataType sparkType) {
    return from(ai.rapids.cudf.ColumnVector.fromScalar(scalar, count), sparkType);
  }

  /**
   * Get the underlying cudf columns from the batch.  This does not increment any
   * reference counts so if you want to use these columns after the batch is closed
   * you will need to do that on your own.
   */
  public static ai.rapids.cudf.ColumnVector[] extractBases(ColumnarBatch batch) {
    int numColumns = batch.numCols();
    ai.rapids.cudf.ColumnVector[] vectors = new ai.rapids.cudf.ColumnVector[numColumns];
    for (int i = 0; i < vectors.length; i++) {
      vectors[i] = ((GpuColumnVector)batch.column(i)).getBase();
    }
    return vectors;
  }

  /**
   * Get the underlying Spark compatible columns from the batch.  This does not increment any
   * reference counts so if you want to use these columns after the batch is closed
   * you will need to do that on your own.
   */
  public static GpuColumnVector[] extractColumns(ColumnarBatch batch) {
    int numColumns = batch.numCols();
    GpuColumnVector[] vectors = new GpuColumnVector[numColumns];

    for (int i = 0; i < vectors.length; i++) {
      vectors[i] = ((GpuColumnVector)batch.column(i));
    }
    return vectors;
  }

  /**
   * Convert the table into columns and return them, outside of a ColumnarBatch.
   * @param colType the types of the columns.
   */
  public static GpuColumnVector[] extractColumns(Table table, DataType[] colType) {
    try (ColumnarBatch batch = from(table, colType)) {
      return extractColumns(batch);
    }
  }

  private final ai.rapids.cudf.ColumnVector cudfCv;

  /**
   * Take an INT32 column vector and return a host side int array.  Don't use this for anything
   * too large.  Note that this ignores validity totally.
   */
  public static int[] toIntArray(ai.rapids.cudf.ColumnVector vec) {
    assert vec.getType() == DType.INT32;
    int rowCount = (int)vec.getRowCount();
    int[] output = new int[rowCount];
    try (HostColumnVector h = vec.copyToHost()) {
      for (int i = 0; i < rowCount; i++) {
        output[i] = h.getInt(i);
      }
    }
    return output;
  }

  /**
   * Sets up the data type of this column vector.
   */
  GpuColumnVector(DataType type, ai.rapids.cudf.ColumnVector cudfCv) {
    super(type);
    // TODO need some checks to be sure everything matches
    this.cudfCv = cudfCv;
  }

  public final GpuColumnVector incRefCount() {
    // Just pass through the reference counting
    cudfCv.incRefCount();
    return this;
  }

  @Override
  public final void close() {
    // Just pass through the reference counting
    cudfCv.close();
  }

  @Override
  public final boolean hasNull() {
    return cudfCv.hasNulls();
  }

  @Override
  public final int numNulls() {
    return (int) cudfCv.getNullCount();
  }

  public static long getTotalDeviceMemoryUsed(GpuColumnVector[] cv) {
    long sum = 0;
    for (int i = 0; i < cv.length; i++){
      sum += cv[i].getBase().getDeviceMemorySize();
    }
    return sum;
  }

  public static long getTotalDeviceMemoryUsed(Table tb) {
    long sum = 0;
    int len = tb.getNumberOfColumns();
    for (int i = 0; i < len; i++) {
      sum += tb.getColumn(i).getDeviceMemorySize();
    }
    return sum;
  }

  public final ai.rapids.cudf.ColumnVector getBase() {
    return cudfCv;
  }

  public final long getRowCount() { return cudfCv.getRowCount(); }

  public final RapidsHostColumnVector copyToHost() {
    return new RapidsHostColumnVector(type, cudfCv.copyToHost());
  }

  @Override
  public final String toString() {
    return getBase().toString();
  }
}
