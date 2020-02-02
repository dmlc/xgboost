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
  private String modelSuffix = ".model";
  private Path checkpointPath;
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
      return Arrays.stream(fs.listStatus(checkpointPath))
              .map(path -> path.getPath().getName())
              .filter(fileName -> fileName.endsWith(modelSuffix))
              .map(fileName -> Integer.valueOf(
                      fileName.substring(0, fileName.length() - modelSuffix.length())))
              .collect(Collectors.toList());
    }
  }

  public void cleanPath() throws IOException {
    fs.delete(checkpointPath, true);
  }

  public Booster loadCheckpointAsBooster() throws IOException, XGBoostError {
    List<Integer> versions = getExistingVersions();
    if (versions.size() > 0) {
      int latestVersion = versions.stream().max(Comparator.comparing(Integer::valueOf)).get();
      String checkpointPath = getPath(latestVersion);
      InputStream in = fs.open(new Path(checkpointPath));
      logger.info("loaded checkpoint from " + checkpointPath);
      Booster booster = XGBoost.loadModel(in);
      booster.setVersion(latestVersion);
      return booster;
    } else {
      return null;
    }
  }

  public void updateCheckpoint(Booster boosterToCheckpoint) throws IOException, XGBoostError {
    List<String> prevModelPaths = getExistingVersions().stream()
            .map(this::getPath).collect(Collectors.toList());
    String eventualPath = getPath(boosterToCheckpoint.getVersion());
    String tempPath = eventualPath + "-" + UUID.randomUUID();
    try (OutputStream out = fs.create(new Path(tempPath), true)) {
      boosterToCheckpoint.saveModel(out);
      fs.rename(new Path(tempPath), new Path(eventualPath));
      logger.info("saving checkpoint with version " + boosterToCheckpoint.getVersion());
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
    getExistingVersions().stream().filter(v -> v / 2 >= currentRound).forEach(v -> {
      try {
        fs.delete(new Path(getPath(v)), true);
      } catch (IOException e) {
        logger.error("failed to clean checkpoint from other training instance", e);
      }
    });
  }

  public List<Integer> getCheckpointRounds(int checkpointInterval, int numOfRounds)
      throws IOException {
    if (checkpointInterval > 0) {
      List<Integer> prevRounds =
              getExistingVersions().stream().map(v -> v / 2).collect(Collectors.toList());
      prevRounds.add(0);
      int firstCheckpointRound = prevRounds.stream()
              .max(Comparator.comparing(Integer::valueOf)).get() + checkpointInterval;
      List<Integer> arr = new ArrayList<>();
      for (int i = firstCheckpointRound; i <= numOfRounds; i += checkpointInterval) {
        arr.add(i);
      }
      arr.add(numOfRounds);
      return arr;
    } else if (checkpointInterval <= 0) {
      List<Integer> l = new ArrayList<Integer>();
      l.add(numOfRounds);
      return l;
    } else {
      throw new IllegalArgumentException("parameters \"checkpoint_path\" should also be set.");
    }
  }
}
