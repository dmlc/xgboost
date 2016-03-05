package ml.dmlc.xgboost4j.java;



import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Distributed RabitTracker, need to be started on driver code before running distributed jobs.
 */
public class RabitTracker {
  // Maybe per tracker logger?
  private static final Log logger = LogFactory.getLog(RabitTracker.class);
  // tracker python file.
  private static String tracker_py = null;
  // environment variable to be pased.
  private Map<String, String> envs = new HashMap<String, String>();
  // number of workers to be submitted.
  private int numWorkers;
  private AtomicReference<Process> trackerProcess = new AtomicReference<Process>();

  static {
    try {
      initTrackerPy();
    } catch (IOException ex) {
      logger.error("load tracker library failed.");
      logger.error(ex);
    }
  }

  /**
   * Tracker logger that logs output from tracker.
   */
  private class TrackerProcessLogger implements Runnable {
    public void run() {

      Log trackerProcessLogger = LogFactory.getLog(TrackerProcessLogger.class);
      BufferedReader reader = new BufferedReader(new InputStreamReader(
              trackerProcess.get().getErrorStream()));
      String line;
      try {
        while ((line = reader.readLine()) != null) {
          trackerProcessLogger.info(line);
        }
      } catch (IOException ex) {
        trackerProcessLogger.error(ex.toString());
      }
    }
  }

  private static void initTrackerPy() throws IOException {
    try {
      tracker_py = NativeLibLoader.createTempFileFromResource("/tracker.py");
    } catch (IOException ioe) {
      logger.trace("cannot access tracker python script");
      throw ioe;
    }
  }


  public RabitTracker(int numWorkers)
    throws XGBoostError {
    if (numWorkers < 1) {
      throw new XGBoostError("numWorkers must be greater equal to one");
    }
    this.numWorkers = numWorkers;
  }

  /**
   * Get environments that can be used to pass to worker.
   * @return The environment settings.
   */
  public Map<String, String> getWorkerEnvs() {
    return envs;
  }

  private void loadEnvs(InputStream ins) throws IOException {
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(ins));
      assert reader.readLine().trim().equals("DMLC_TRACKER_ENV_START");
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.trim().equals("DMLC_TRACKER_ENV_END")) {
          break;
        }
        String[] sep = line.split("=");
        if (sep.length == 2) {
          envs.put(sep[0], sep[1]);
        }
      }
      reader.close();
    } catch (IOException ioe){
      logger.error("cannot get runtime configuration from tracker process");
      ioe.printStackTrace();
      throw ioe;
    }
  }

  private boolean startTrackerProcess() {
    try {
      trackerProcess.set(Runtime.getRuntime().exec("python " + tracker_py +
              " --log-level=DEBUG --num-workers=" + String.valueOf(numWorkers)));
      loadEnvs(trackerProcess.get().getInputStream());
      return true;
    } catch (IOException ioe) {
      ioe.printStackTrace();
      return false;
    }
  }

  private void stop() {
    if (trackerProcess.get() != null) {
      trackerProcess.get().destroy();
    }
  }

  public boolean start() {
    if (startTrackerProcess()) {
      logger.debug("Tracker started, with env=" + envs.toString());
      // also start a tracker logger
      Thread logger_thread = new Thread(new TrackerProcessLogger());
      logger_thread.setDaemon(true);
      logger_thread.start();
      return true;
    } else {
      logger.error("FAULT: failed to start tracker process");
      stop();
      return false;
    }
  }

  public void waitFor() {
    try {
      trackerProcess.get().waitFor();
      logger.info("Tracker Process ends with exit code " + trackerProcess.get().exitValue());
      stop();
    } catch (InterruptedException e) {
      // we should not get here as RabitTracker is accessed in the main thread
      e.printStackTrace();
      logger.error("the RabitTracker thread is terminated unexpectedly");
    }
  }
}
