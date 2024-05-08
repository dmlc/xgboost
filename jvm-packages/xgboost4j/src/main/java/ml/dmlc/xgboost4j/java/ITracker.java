package ml.dmlc.xgboost4j.java;

import java.util.Map;

/**
 * Interface for a tracker implementations with three public methods:
 *
 *  - start(timeout): Start the tracker awaiting for worker connections, with a given
 *  timeout value (in seconds).
 *  - workerArgs(): Return the arguments needed to initialize Rabit clients.
 *  - waitFor(timeout): Wait for the task execution by the worker nodes for at most `timeout`
 *  milliseconds.
 *
 * Each implementation is expected to implement a callback function
 *
 *    public void uncaughtException(Threat t, Throwable e) { ... }
 *
 * to interrupt waitFor() in order to prevent the tracker from hanging indefinitely.
 *
 * The Rabit tracker handles connections from distributed workers, assigns ranks to workers, and
 * brokers connections between workers.
 */
public interface ITracker extends Thread.UncaughtExceptionHandler {
  enum TrackerStatus {
    SUCCESS(0), INTERRUPTED(1), TIMEOUT(2), FAILURE(3);

    private int statusCode;

    TrackerStatus(int statusCode) {
      this.statusCode = statusCode;
    }

    public int getStatusCode() {
      return this.statusCode;
    }
  }

  Map<String, Object> workerArgs() throws XGBoostError;

  boolean start() throws XGBoostError;

  void stop() throws XGBoostError;

  void waitFor(long taskExecutionTimeout) throws XGBoostError;
}
