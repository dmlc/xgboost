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

package ml.dmlc.xgboost4j.scala.spark.rapids

import ai.rapids.cudf.DType
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.catalyst.util.DateTimeConstants._

class RowConverter(schema: StructType, timeUnits: Seq[DType]) {
  private val converters = schema.fields.map {
    f => RowConverter.getConverterForType(f.dataType)
  }

  final def toExternalRow(row: InternalRow): Row = {
      if (row == null) {
        null
      } else {
        val ar = new Array[Any](row.numFields)
        var idx = 0
        while (idx < row.numFields) {
          ar(idx) = converters(idx).convert(row, idx, timeUnits)
          idx += 1
        }
        new GenericRowWithSchema(ar, schema)
      }
  }
}

object RowConverter {
  private abstract class TypeConverter {
    final def convert(row: InternalRow, column: Int, timeUnits: Seq[DType]): Any = {
      if (row.isNullAt(column)) null else convertImpl(row, column, timeUnits)
    }

    protected def convertImpl(row: InternalRow, column: Int): Any

    protected def convertImpl(row: InternalRow, column: Int, timeUnits: Seq[DType]): Any = {
      convertImpl(row, column)
    }
  }

  def isSupportedType(dataType: DataType): Boolean = {
    dataType match {
      case _: BooleanType | ByteType | ShortType | IntegerType | FloatType |
              LongType | DoubleType | DateType | TimestampType => true
      case _ => false
    }
  }

  private def getConverterForType(dataType: DataType): TypeConverter = {
    dataType match {
      case BooleanType => BooleanConverter
      case ByteType => ByteConverter
      case ShortType => ShortConverter
      case IntegerType => IntConverter
      case FloatType => FloatConverter
      case LongType => LongConverter
      case DoubleType => DoubleConverter
      case DateType => DateConverter
      case TimestampType => TimestampConverter
      case StringType => StringConverter
      case unknown => throw new UnsupportedOperationException(
        s"Type $unknown not supported")
    }
  }

  private object BooleanConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getBoolean(column)
  }

  private object ByteConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getByte(column)
  }

  private object ShortConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getShort(column)
  }

  private object IntConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getInt(column)
  }

  private object FloatConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getFloat(column)
  }

  private object LongConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getLong(column)
  }

  private object DoubleConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getDouble(column)
  }

  private object DateConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      DateTimeUtils.toJavaDate(row.getInt(column))
  }

  private object StringConverter extends TypeConverter {
    override def convertImpl(row: InternalRow, column: Int): Any =
      row.getString(column)
  }

  private object TimestampConverter extends TypeConverter {
    private val NANOS_PER_MICROS: Long = 1000 // to work compatible with Spark 2.3.3
    private def toMicros(value: Long, unit: DType): Long = {
      unit match {
        case DType.TIMESTAMP_SECONDS => value * MICROS_PER_SECOND
        case DType.TIMESTAMP_MILLISECONDS => value * MICROS_PER_MILLIS
        case DType.TIMESTAMP_MICROSECONDS => value
        case DType.TIMESTAMP_NANOSECONDS => value / NANOS_PER_MICROS
        case DType.TIMESTAMP_DAYS => throw new IllegalArgumentException("TIMESTAMP_DAYS is not" +
          " supported yet. You may use 'DateType' for it!")
        case _ => throw new IllegalArgumentException("Unsupported timestamp type ${unit}.")
      }
    }

    override protected def convertImpl(
        row: InternalRow, column: Int, timeUnits: Seq[DType]): Any = {
      DateTimeUtils.toJavaTimestamp(toMicros(row.getLong(column), timeUnits(column)))
    }

    override protected def convertImpl(row: InternalRow, column: Int): Any = null
  }
}
