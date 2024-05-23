/*
 Copyright (c) 2014-2024 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.java;

import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Java implementation of the Rabit tracker to coordinate distributed workers.
 */
public class RabitTracker implements ITracker {
  // Maybe per tracker logger?
  private static final Log logger = LogFactory.getLog(RabitTracker.class);
  private long handle = 0;
  private Thread trackerDaemon;

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
      this.trackerDaemon.interrupt();
    }
  }

  /**
   * Get environments that can be used to pass to worker.
   * @return The environment settings.
   */
  public Map<String, Object> getWorkerArgs() throws XGBoostError {
    // fixme: timeout
    String[] args = new String[1];
    XGBoostJNI.checkCall(XGBoostJNI.TrackerWorkerArgs(this.handle, 0, args));
    ObjectMapper mapper = new ObjectMapper();
    Map<String, Object> config;
    try {
      config = mapper.readValue(args[0], new TypeReference<Map<String, Object>>() {});
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
    this.trackerDaemon = new Thread(() -> {
      try {
        waitFor(0);
      } catch (XGBoostError ex) {
        logger.error(ex);
        return; // exit the thread
      }
    });
    this.trackerDaemon.setDaemon(true);
    this.trackerDaemon.start();

    return this.trackerDaemon.isAlive();
  }

  public void waitFor(long timeout) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.TrackerWaitFor(this.handle, timeout));
  }
}
