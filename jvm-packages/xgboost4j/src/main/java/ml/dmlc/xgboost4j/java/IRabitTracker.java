package ml.dmlc.xgboost4j.java;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Interface for Rabit tracker implementations with three public methods:
 *
 *  - start(): Start the rabit tracker awaiting for worker connections.
 *  - getWorkerEnvs(): Return the environment variables needed to initialize Rabit clients.
 *  - waitFor(): Wait for the task execution by the worker nodes.
 */
public interface IRabitTracker {
  Map<String, String> getWorkerEnvs();
  boolean start();
  // for Scala-implementation compatibility
  boolean start(long timeout, TimeUnit unit);
  int waitFor();
  int waitFor(long timeout, TimeUnit unit);
}
