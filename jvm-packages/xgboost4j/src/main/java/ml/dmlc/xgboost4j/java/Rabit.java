package ml.dmlc.xgboost4j.java;

import java.io.IOException;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Rabit global class for synchronization.
 */
public class Rabit {
  private static final Log logger = LogFactory.getLog(DMatrix.class);
  //load native library
  static {
    try {
      NativeLibLoader.initXGBoost();
    } catch (IOException ex) {
      logger.error("load native library failed.");
      logger.error(ex);
    }
  }

  public enum OpType {
    MAX(0), MIN(1), SUM(2), BITWISE_OR(3);

    private int op;

    public int getOperand() {
      return this.op;
    }

    OpType(int op) {
      this.op = op;
    }
  }

  public enum DataType {
    CHAR(0, 1), UCHAR(1, 1), INT(2, 4), UNIT(3, 4),
    LONG(4, 8), ULONG(5, 8), FLOAT(6, 4), DOUBLE(7, 8),
    LONGLONG(8, 8), ULONGLONG(9, 8);

    private int enumOp;
    private int size;

    public int getEnumOp() {
      return this.enumOp;
    }

    public int getSize() {
      return this.size;
    }

    DataType(int enumOp, int size) {
      this.enumOp = enumOp;
      this.size = size;
    }
  }

  private static void checkCall(int ret) throws XGBoostError {
    if (ret != 0) {
      throw new XGBoostError(XGBoostJNI.XGBGetLastError());
    }
  }

  /**
   * Initialize the rabit library on current working thread.
   * @param envs The additional environment variables to pass to rabit.
   * @throws XGBoostError
   */
  public static void init(Map<String, String> envs) throws XGBoostError {
    String[] args = new String[envs.size()];
    int idx = 0;
    for (java.util.Map.Entry<String, String> e : envs.entrySet()) {
      args[idx++] = e.getKey() + '=' + e.getValue();
    }
    checkCall(XGBoostJNI.RabitInit(args));
  }

  /**
   * Shutdown the rabit engine in current working thread, equals to finalize.
   * @throws XGBoostError
   */
  public static void shutdown() throws XGBoostError {
    checkCall(XGBoostJNI.RabitFinalize());
  }

  /**
   * Print the message on rabit tracker.
   * @param msg
   * @throws XGBoostError
   */
  public static void trackerPrint(String msg) throws XGBoostError {
    checkCall(XGBoostJNI.RabitTrackerPrint(msg));
  }

  /**
   * Get version number of current stored model in the thread.
   * which means how many calls to CheckPoint we made so far.
   * @return version Number.
   * @throws XGBoostError
   */
  public static int versionNumber() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XGBoostJNI.RabitVersionNumber(out));
    return out[0];
  }

  /**
   * get rank of current thread.
   * @return the rank.
   * @throws XGBoostError
   */
  public static int getRank() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XGBoostJNI.RabitGetRank(out));
    return out[0];
  }

  /**
   * get world size of current job.
   * @return the worldsize
   * @throws XGBoostError
   */
  public static int getWorldSize() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XGBoostJNI.RabitGetWorldSize(out));
    return out[0];
  }

  public static float[] allReduce(float[] elements, OpType op) {
    DataType dataType = DataType.FLOAT;
    ByteBuffer buffer = ByteBuffer.allocateDirect(dataType.getSize() * elements.length)
            .order(ByteOrder.nativeOrder());

    for (float el : elements) {
      buffer.putFloat(el);
    }
    buffer.flip();

    XGBoostJNI.RabitAllreduce(buffer, elements.length, dataType.getEnumOp(), op.getOperand());
    float[] results = new float[elements.length];
    buffer.asFloatBuffer().get(results);

    return results;
  }

  public static double[] allReduce(double[] elements, OpType op) {
    DataType dataType = DataType.DOUBLE;
    ByteBuffer buffer = ByteBuffer.allocateDirect(dataType.getSize() * elements.length)
            .order(ByteOrder.nativeOrder());

    for (double el : elements) {
      buffer.putDouble(el);
    }
    buffer.flip();

    XGBoostJNI.RabitAllreduce(buffer, elements.length, dataType.getEnumOp(), op.getOperand());
    double[] results = new double[elements.length];
    buffer.asDoubleBuffer().get(results);

    return results;
  }

  public static int[] allReduce(int[] elements, OpType op) {
    DataType dataType = DataType.INT;
    ByteBuffer buffer = ByteBuffer.allocateDirect(dataType.getSize() * elements.length)
            .order(ByteOrder.nativeOrder());

    for (int el : elements) {
      buffer.putInt(el);
    }
    buffer.flip();

    XGBoostJNI.RabitAllreduce(buffer, elements.length, dataType.getEnumOp(), op.getOperand());
    int[] results = new int[elements.length];
    buffer.asIntBuffer().get(results);

    return results;
  }

  public static long[] allReduce(long[] elements, OpType op) {
    DataType dataType = DataType.LONG;
    ByteBuffer buffer = ByteBuffer.allocateDirect(dataType.getSize() * elements.length)
            .order(ByteOrder.nativeOrder());

    for (long el : elements) {
      buffer.putLong(el);
    }
    buffer.flip();

    XGBoostJNI.RabitAllreduce(buffer, elements.length, dataType.getEnumOp(), op.getOperand());
    long[] results = new long[elements.length];
    buffer.asLongBuffer().get(results);

    return results;
  }

}
