/*
 Copyright (c) 2014-2023 by Contributors

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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class ExternalCheckpointManager {

  private Log logger = LogFactory.getLog("ExternalCheckpointManager");
  private String modelSuffix = ".ubj";
  private Path checkpointPath;  // directory for checkpoints
  private FileSystem fs;

  public ExternalCheckpointManager(String checkpointPath, FileSystem fs) throws XGBoostError {
    if (checkpointPath == null || checkpointPath.isEmpty()) {
      throw new XGBoostError("cannot create ExternalCheckpointManager with null or" +
              " empty checkpoint path");
    }
    this.checkpointPath = new Path(checkpointPath);
    this.fs = fs;
  }

  private String getPath(int version) {
    return checkpointPath.toUri().getPath() + "/" + version + modelSuffix;
  }

  private List<Integer> getExistingVersions() throws IOException {
    if (!fs.exists(checkpointPath)) {
      return new ArrayList<>();
    } else {
      // Get integer versions from a list of checkpoint files.
      return Arrays.stream(fs.listStatus(checkpointPath))
              .map(path -> path.getPath().getName())
              .filter(fileName -> fileName.endsWith(modelSuffix))
              .map(fileName -> Integer.valueOf(
                      fileName.substring(0, fileName.length() - modelSuffix.length())))
              .collect(Collectors.toList());
    }
  }

  private Integer latest(List<Integer> versions) {
    return versions.stream()
        .max(Comparator.comparing(Integer::valueOf)).get();
  }

  public void cleanPath() throws IOException {
    fs.delete(checkpointPath, true);
  }

  public Booster loadCheckpointAsBooster() throws IOException, XGBoostError {
    List<Integer> versions = getExistingVersions();
    if (versions.size() > 0) {
      int latestVersion = this.latest(versions);
      String checkpointPath = getPath(latestVersion);
      InputStream in = fs.open(new Path(checkpointPath));
      logger.info("loaded checkpoint from " + checkpointPath);
      Booster booster = XGBoost.loadModel(in);
      return booster;
    } else {
      return null;
    }
  }

  public void updateCheckpoint(Booster boosterToCheckpoint) throws IOException, XGBoostError {
    List<String> prevModelPaths = getExistingVersions().stream()
        .map(this::getPath).collect(Collectors.toList());
    // checkpointing is done after update, so n_rounds - 1 is the current iteration
    // accounting for training continuation.
    Integer iter = boosterToCheckpoint.getNumBoostedRound() - 1;
    String eventualPath = getPath(iter);
    String tempPath = eventualPath + "-" + UUID.randomUUID();
    try (OutputStream out = fs.create(new Path(tempPath), true)) {
      boosterToCheckpoint.saveModel(out);
      fs.rename(new Path(tempPath), new Path(eventualPath));
      logger.info("saving checkpoint with version " + iter);
      prevModelPaths.stream().forEach(path -> {
        try {
          fs.delete(new Path(path), true);
        } catch (IOException e) {
          logger.error("failed to delete outdated checkpoint at " + path, e);
        }
      });
    }
  }

  public void cleanUpHigherVersions(int currentRound) throws IOException {
    getExistingVersions().stream().filter(v -> v > currentRound).forEach(v -> {
      try {
        fs.delete(new Path(getPath(v)), true);
      } catch (IOException e) {
        logger.error("failed to clean checkpoint from other training instance", e);
      }
    });
  }
  // Get a list of iterations that need checkpointing.
  public List<Integer> getCheckpointRounds(
      int firstRound, int checkpointInterval, int numOfRounds)
      throws IOException {
    int end = firstRound + numOfRounds; // exclusive
    int lastRound = end - 1;
    if (end - 1 < 0) {
      throw new IllegalArgumentException("Inavlid `numOfRounds`.");
    }

    List<Integer> arr = new ArrayList<>();
    if (checkpointInterval > 0) {
      for (int i = firstRound; i < end; i += checkpointInterval) {
        arr.add(i);
      }
    }

    if (!arr.contains(lastRound)) {
      arr.add(lastRound);
    }
    return arr;
  }
}
