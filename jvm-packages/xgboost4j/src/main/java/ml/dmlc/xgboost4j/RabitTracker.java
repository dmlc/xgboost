package ml.dmlc.xgboost4j;



import java.io.*;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Distributed RabitTracker, need to be started on driver code before running distributed jobs.
 */
public class RabitTracker {
  // Maybe per tracker logger?
  private static final Log logger = LogFactory.getLog(RabitTracker.class);
  // tracker python file.
  private static File tracker_py = null;
  // environment variable to be pased.
  private Map<String, String> envs = new HashMap<String, String>();
  // number of workers to be submitted.
  private int num_workers;
  // child process
  private  Process process = null;
  // logger thread
  private Thread logger_thread = null;

  //load native library
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
  private class TrackerLogger implements Runnable {
    public void run() {
      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
      String line;
      try {
        while ((line = reader.readLine()) != null) {
          logger.info(line);
        }
      } catch (IOException ex) {
        logger.error(ex.toString());
      }
    }
  }

  private static synchronized void initTrackerPy() throws IOException {
    tracker_py = FileUtil.createTempFileFromResource("/tracker.py");
  }


  public RabitTracker(int num_workers) {
    this.num_workers = num_workers;
  }

  /**
   * Get environments that can be used to pass to worker.
   * @return The environment settings.
   */
  public Map<String, String> getWorkerEnvs() {
    return envs;
  }

  public void start() throws IOException {
    process = Runtime.getRuntime().exec("python " + tracker_py.getAbsolutePath() +
            " --num-workers=" + new Integer(num_workers).toString());
    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
    assert reader.readLine().trim().equals("DMLC_TRACKER_ENV_START");
    String line;
    while ((line = reader.readLine()) != null) {
      if (line.trim().equals("DMLC_TRACKER_ENV_END")) {
        break;
      }
      String []sep = line.split("=");
      if (sep.length == 2) {
        envs.put(sep[0], sep[1]);
      }
    }
    logger.debug("Tracker started, with env=" + envs.toString());
    // also start a tracker logger
    logger_thread = new Thread(new TrackerLogger());
    logger_thread.setDaemon(true);
    logger_thread.start();
  }

  public void waitFor() throws InterruptedException {
    process.waitFor();
  }
}
