package ml.dmlc.xgboost4j.java;

import java.io.IOException;
import java.io.Serializable;
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
}
