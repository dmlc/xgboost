package ml.dmlc.xgboost4j.java;

import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Java implementation of the Rabit tracker to coordinate distributed workers.
 *
 * The tracker must be started on driver node before running distributed jobs.
 */
public class RabitTracker implements ITracker {
  // Maybe per tracker logger?
  private static final Log logger = LogFactory.getLog(RabitTracker.class);
  private long handle = 0;
  private Thread tracker_daemon;

  public RabitTracker(int numWorkers) throws XGBoostError {
    this(numWorkers, "");
  }

  public RabitTracker(int numWorkers, String hostIp)
      throws XGBoostError {
    this(numWorkers, hostIp, 0, 300);
  }
  public RabitTracker(int numWorkers, String hostIp, int port, int timeout) throws XGBoostError {
    if (numWorkers < 1) {
      throw new XGBoostError("numWorkers must be greater equal to one");
    }

    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.TrackerCreate(hostIp, numWorkers, port, 0, timeout, out));
    this.handle = out[0];
  }

  public void uncaughtException(Thread t, Throwable e) {
    logger.error("Uncaught exception thrown by worker:", e);
    try {
      Thread.sleep(5000L);
    } catch (InterruptedException ex) {
      logger.error(ex);
    } finally {
      this.tracker_daemon.interrupt();
    }
  }

  /**
   * Get environments that can be used to pass to worker.
   * @return The environment settings.
   */
  public Map<String, Object> workerArgs() throws XGBoostError {
    // fixme: timeout
    String[] args = new String[1];
    XGBoostJNI.checkCall(XGBoostJNI.TrackerWorkerArgs(this.handle, 0, args));
    ObjectMapper mapper = new ObjectMapper();
    TypeReference<Map<String, Object>> typeRef = new TypeReference<Map<String, Object>>() {
    };
    Map<String, Object> config;
    try {
      config = mapper.readValue(args[0], typeRef);
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to get worker arguments.", ex);
    }
    return config;
  }

  public void stop() throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.TrackerFree(this.handle));
  }

  public boolean start() throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.TrackerRun(this.handle));
    this.tracker_daemon = new Thread(() -> {
      try {
        XGBoostJNI.checkCall(XGBoostJNI.TrackerWaitFor(this.handle, 0));
      } catch (XGBoostError ex) {
        logger.error(ex);
        return; // exit the thread
      }
    });
    this.tracker_daemon.setDaemon(true);
    this.tracker_daemon.start();

    return this.tracker_daemon.isAlive();
  }

  public void waitFor(long timeout) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.TrackerWaitFor(this.handle, timeout));
  }
}
