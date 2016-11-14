package ml.dmlc.xgboost4j.java;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Interface for Rabit tracker implementations with three public methods:
 *
 *  - start(timeout): Start the Rabit tracker awaiting for worker connections, with a given
 *  timeout value (in milliseconds.)
 *  - getWorkerEnvs(): Return the environment variables needed to initialize Rabit clients.
 *  - waitFor(timeout): Wait for the task execution by the worker nodes for at most `timeout`
 *  milliseconds.
 *
 * The Rabit tracker handles connections from distributed workers, assigns ranks to workers, and
 * brokers connections between workers.
 */
public interface IRabitTracker {
  Map<String, String> getWorkerEnvs();
  boolean start(long workerConnectionTimeout);
  int waitFor(long taskExecutionTimeout);
}
