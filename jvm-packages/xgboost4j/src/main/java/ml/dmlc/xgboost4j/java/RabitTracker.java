package ml.dmlc.xgboost4j.java;



import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Java implementation of the Rabit tracker to coordinate distributed workers.
 * As a wrapper of the Python Rabit tracker, this implementation does not handle timeout for both
 * start() and waitFor() methods (i.e., the timeout is infinite.)
 *
 * For systems lacking Python environment, or for timeout functionality, consider using the Scala
 * Rabit tracker (ml.dmlc.xgboost4j.scala.rabit.RabitTracker) which does not depend on Python, and
 * provides timeout support.
 *
 * The tracker must be started on driver node before running distributed jobs.
 */
public class RabitTracker implements IRabitTracker {
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
        trackerProcess.get().waitFor();
        trackerProcessLogger.info("Tracker Process ends with exit code " +
                trackerProcess.get().exitValue());
      } catch (IOException ex) {
        trackerProcessLogger.error(ex.toString());
      } catch (InterruptedException ie) {
        // we should not get here as RabitTracker is accessed in the main thread
        ie.printStackTrace();
        logger.error("the RabitTracker thread is terminated unexpectedly");
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

  public void uncaughtException(Thread t, Throwable e) {
    logger.error("Uncaught exception thrown by worker:", e);
    try {
      Thread.sleep(5000L);
    } catch (InterruptedException ex) {
      logger.error(ex);
    } finally {
      trackerProcess.get().destroy();
    }
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

  public void stop() {
    if (trackerProcess.get() != null) {
      trackerProcess.get().destroy();
    }
  }

  public boolean start(long timeout) {
    if (timeout > 0L) {
      logger.warn("Python RabitTracker does not support timeout. " +
              "The tracker will wait for all workers to connect indefinitely, unless " +
              "it is interrupted manually. Use the Scala RabitTracker for timeout support.");
    }

    if (startTrackerProcess()) {
      logger.debug("Tracker started, with env=" + envs.toString());
      System.out.println("Tracker started, with env=" + envs.toString());
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

  public int waitFor(long timeout) {
    if (timeout > 0L) {
      logger.warn("Python RabitTracker does not support timeout. " +
              "The tracker will wait for either all workers to finish tasks and send " +
              "shutdown signal, or manual interruptions. " +
              "Use the Scala RabitTracker for timeout support.");
    }

    try {
      trackerProcess.get().waitFor();
      int returnVal = trackerProcess.get().exitValue();
      logger.info("Tracker Process ends with exit code " + returnVal);
      stop();
      return returnVal;
    } catch (InterruptedException e) {
      // we should not get here as RabitTracker is accessed in the main thread
      e.printStackTrace();
      logger.error("the RabitTracker thread is terminated unexpectedly");
      return TrackerStatus.INTERRUPTED.getStatusCode();
    }
  }
}
