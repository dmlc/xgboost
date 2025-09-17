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

/**
 * This class contains the methods that are required for managing the state of the training
 * process. The training state is stored in a distributed file system, that consists of
 * UBJ (Universal Binary JSON) model files.
 * The class provides methods for saving, loading and cleaning up checkpoints.
 */
public class ExternalCheckpointManager {

  private Log logger = LogFactory.getLog("ExternalCheckpointManager");
  private String modelSuffix = ".ubj";
  private Path checkpointPath;  // directory for checkpoints
  private FileSystem fs;

  /**
   * This constructor creates a new Expternal Checkpoint Manager at the specified path in the
   * specified file system.
   *
   * @param checkpointPath The directory path where checkpoints will be stored.
   * @param fs The file system to use for storing checkpoints.
   * @throws XGBoostError the error that is thrown is the checkpoint path is null or empty.
   */
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

  /**
   * This method cleans all the directories and files that are present in the checkpoint path.
   * @throws IOException exception that is thrown when there is an error deleting the
   * checkpoint path.
   */
  public void cleanPath() throws IOException {
    fs.delete(checkpointPath, true);
  }

  /**
   * Read the checkpoint from the checkpoint path. Once the checkpoint path is read, we get
   * the latest version of the checkpoint from all the checkpoint versions and lead it
   * into the booster for the purpose of making predictions.
   *
   * @return The booster object that is used for making predictions.
   * @throws IOException Any expection that occurs when reading the checkpoint path.
   * @throws XGBoostError Any exception that occurs when loading the model into the booster.
   */
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

  /**
   * This method updates the booster checkpoint to the the latest or current
   * version and deleted all the previous versions of the checkpoint.
   * @param boosterToCheckpoint The booster object that is to be checkpointed and
   *                            saved as a model file.
   * @throws IOException Any exception that occurs when writing the model file to the
   * checkpoint path.
   * @throws XGBoostError Any exception that occurs when saving the model from the booster.
   */
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

  /**
   * This method cleans up all the checkpoint versions that are higher than the current round.
   * This is useful when multiple training instances are running and we want to make sure that
   * only the checkpoints from the current training instance are retained.
   * @param currentRound The current round of training.
   * @throws IOException Any exception that occurs when deleting the checkpoint files.
   */
  public void cleanUpHigherVersions(int currentRound) throws IOException {
    getExistingVersions().stream().filter(v -> v > currentRound).forEach(v -> {
      try {
        fs.delete(new Path(getPath(v)), true);
      } catch (IOException e) {
        logger.error("failed to clean checkpoint from other training instance", e);
      }
    });
  }

  /**
   * Get a list of iterations that need checkpointing.
   * @param firstRound The first round of training.
   * @param checkpointInterval The interval at which checkpoints are to be saved.
   * @param numOfRounds The number of rounds to be trained.
   * @return A list of integer rounds that need checkpointing.
   * @throws IOException Any exception that occurs when getting the list of rounds.
   */
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
