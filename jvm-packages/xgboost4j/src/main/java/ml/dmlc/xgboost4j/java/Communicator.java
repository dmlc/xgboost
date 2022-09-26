package ml.dmlc.xgboost4j.java;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

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
    INT8(0, 1), UINT8(1, 1), INT32(2, 4), UINT32(3, 4),
    INT64(4, 8), UINT64(5, 8), FLOAT32(6, 4), FLOAT64(7, 8);

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

  // used as way to test/debug passed communicator init parameters
  public static Map<String, String> communicatorEnvs;
  public static List<String> mockList = new LinkedList<>();

  /**
   * Initialize the collective communicator on current working thread.
   *
   * @param envs The additional environment variables to pass to the communicator.
   * @throws XGBoostError
   */
  public static void init(Map<String, String> envs) throws XGBoostError {
    communicatorEnvs = envs;
    String[] args = new String[envs.size() * 2 + mockList.size() * 2];
    int idx = 0;
    for (java.util.Map.Entry<String, String> e : envs.entrySet()) {
      args[idx++] = e.getKey();
      args[idx++] = e.getValue();
    }
    // pass list of rabit mock strings eg mock=0,1,0,0
    for (String mock : mockList) {
      args[idx++] = "mock";
      args[idx++] = mock;
    }
    checkCall(XGBoostJNI.CommunicatorInit(args));
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
