package ml.dmlc.xgboost4j.java;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Collective communicator global class for synchronization.
 *
 * Currently the communicator API is experimental, function signatures may change in the future
 * without notice.
 */
public class Communicator {

  public enum OpType implements Serializable {
    MAX(0), MIN(1), SUM(2);

    private int op;

    public int getOperand() {
      return this.op;
    }

    OpType(int op) {
      this.op = op;
    }
  }

  public enum DataType implements Serializable {
    FLOAT16(0, 2), FLOAT32(1, 4), FLOAT64(2, 8),
    INT8(4, 1), INT16(5, 2), INT32(6, 4), INT64(7, 8),
    UINT8(8, 1), UINT16(9, 2), UINT32(10, 4), UINT64(11, 8);

    private final int enumOp;
    private final int size;

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
   * Initialize the collective communicator on current working thread.
   *
   * @param envs The additional environment variables to pass to the communicator.
   * @throws XGBoostError
   */
  public static void init(Map<String, Object> envs) throws XGBoostError {
    ObjectMapper mapper = new ObjectMapper();
    try {
      String jconfig = mapper.writeValueAsString(envs);
      checkCall(XGBoostJNI.CommunicatorInit(jconfig));
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to read arguments for the communicator.", ex);
    }
  }

  /**
   * Shutdown the communicator in current working thread, equals to finalize.
   *
   * @throws XGBoostError
   */
  public static void shutdown() throws XGBoostError {
    checkCall(XGBoostJNI.CommunicatorFinalize());
  }

  /**
   * Print the message via the communicator.
   *
   * @param msg
   * @throws XGBoostError
   */
  public static void communicatorPrint(String msg) throws XGBoostError {
    checkCall(XGBoostJNI.CommunicatorPrint(msg));
  }

  /**
   * get rank of current thread.
   *
   * @return the rank.
   * @throws XGBoostError
   */
  public static int getRank() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XGBoostJNI.CommunicatorGetRank(out));
    return out[0];
  }

  /**
   * get world size of current job.
   *
   * @return the worldsize
   * @throws XGBoostError
   */
  public static int getWorldSize() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XGBoostJNI.CommunicatorGetWorldSize(out));
    return out[0];
  }

  /**
   * perform Allreduce on distributed float vectors using operator op.
   *
   * @param elements local elements on distributed workers.
   * @param op       operator used for Allreduce.
   * @return All-reduced float elements according to the given operator.
   */
  public static float[] allReduce(float[] elements, OpType op) {
    DataType dataType = DataType.FLOAT32;
    ByteBuffer buffer = ByteBuffer.allocateDirect(dataType.getSize() * elements.length)
            .order(ByteOrder.nativeOrder());

    for (float el : elements) {
      buffer.putFloat(el);
    }
    buffer.flip();

    XGBoostJNI.CommunicatorAllreduce(buffer, elements.length, dataType.getEnumOp(),
            op.getOperand());
    float[] results = new float[elements.length];
    buffer.asFloatBuffer().get(results);

    return results;
  }
}
