package ml.dmlc.xgboost4j;

import java.util.Map;
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Rabit global class for synchronization.
 */
class Rabit {
  private static final Log logger = LogFactory.getLog(DMatrix.class);
  //load native library
  static {
    try {
      NativeLibLoader.initXgBoost();
    } catch (IOException ex) {
      logger.error("load native library failed.");
      logger.error(ex);
    }
  }

  private static void checkCall(int ret) throws XGBoostError {
    if (ret != 0) {
      throw new XGBoostError(XgboostJNI.XGBGetLastError());
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
    checkCall(XgboostJNI.RabitInit(args));
  }

  /**
   * Shutdown the rabit engine in current working thread, equals to finalize.
   * @throws XGBoostError
   */
  public static void shutdown() throws XGBoostError {
    checkCall(XgboostJNI.RabitFinalize());
  }

  /**
   * Print the message on rabit tracker.
   * @param msg
   * @throws XGBoostError
   */
  public static void trackerPrint(String msg) throws XGBoostError {
    checkCall(XgboostJNI.RabitTrackerPrint(msg));
  }

  /**
   * Get version number of current stored model in the thread.
   * which means how many calls to CheckPoint we made so far.
   * @return version Number.
   * @throws XGBoostError
   */
  public static int versionNumber() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XgboostJNI.RabitVersionNumber(out));
    return out[0];
  }

  /**
   * get rank of current thread.
   * @return the rank.
   * @throws XGBoostError
   */
  public static int getRank() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XgboostJNI.RabitGetRank(out));
    return out[0];
  }

  /**
   * get world size of current job.
   * @return the worldsize
   * @throws XGBoostError
   */
  public static int getWorldSize() throws XGBoostError {
    int[] out = new int[1];
    checkCall(XgboostJNI.RabitGetWorldSize(out));
    return out[0];
  }
}
