package com.airbnb.common.config;

import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import com.typesafe.config.ConfigParseOptions;
import com.typesafe.config.ConfigResolveOptions;


/**
  * Airbnb extension of HOCON language.
  * https://github.com/typesafehub/config/blob/master/HOCON.md
  */
@Slf4j
public class AirCon {
  protected MacroExecutor executor = null;
  // Since we modify Config via system properties, we have to clean it up. If not, it will affect
  // the next AirCon "load" call.
  private Set<String> keysToCleanUp = new HashSet<>();
  // List of Exceptions.
  static class InvalidMacro extends RuntimeException {
    InvalidMacro(String message) {
      super(message);
    }
  }

  static class InvalidRegexPattern extends RuntimeException {
    InvalidRegexPattern(String message) {
      super(message);
    }
  }

  // A drop-in replacement for ConfigFactory.load(...).
  static public Config load(String resourceBasename) {
    return new AirCon(new AirConMacroExecutor()).loadAndProcess(resourceBasename);
  }

  public static<T> T getOrElse(Config config, String key, T defaultResult) {
    if (config.hasPath(key)) {
      return (T)config.getAnyRef(key);
    } else {
      return defaultResult;
    }
  }

  protected AirCon(MacroExecutor executor) {
    this.executor = executor;
  }

  private Config tryLoadFromFS(String resourceBasename) {
    File file = new File(resourceBasename);
    if(file.exists()) {
      log.info("Loading config from file: " + file);
      return ConfigFactory.load(ConfigFactory.parseFile(file));
    } else {
      log.info("Loading config using classpath loader: " + resourceBasename);
      return ConfigFactory.load(
          resourceBasename,
          ConfigParseOptions.defaults().setAllowMissing(false),
          ConfigResolveOptions.defaults()
      );
    }
  }

  protected Config loadAndProcess(String resourceBasename) {
    // Need this because some other processes might load something else that interfere with.
    ConfigFactory.invalidateCaches();
    Config config = tryLoadFromFS(resourceBasename);
    int numReloads = 1;
    if (config.hasPath("num_reloads_required")) {
      numReloads = config.getInt("num_reloads_required");
      System.out.println("Gonna execute macros " + numReloads + " times.");
    }
    for (int i = 0; i < numReloads; i++) {
      if (maybeApplyAirbnbSpecificMacros(config)) {
        ConfigFactory.invalidateCaches();
        config = tryLoadFromFS(resourceBasename);
      }
    }
    cleanUp();
    return config;
  }

  private boolean maybeApplyAirbnbSpecificMacros(Config config) {
    boolean macroUsed = false;

    for (String key : config.root().keySet()) {
      if (key.startsWith("aircon_get_")) {
        System.out.println("Macro " + key);
        String output = applyAirbnbMacro(config, key);
        System.out.println("... output = " + output);
        if (output != null) {
          System.setProperty(key + ".output", output);
          keysToCleanUp.add(key + ".output");
          macroUsed = true;
        }
      }
    }
    return macroUsed;
  }

  private String applyAirbnbMacro(Config config, String key) {
    Config macroConfig = config.getConfig(key);
    switch (macroConfig.getString("macro")) {
      case "max_string":
        return executor.getMaxString(macroConfig);
      case "min_string":
        return executor.getMinString(macroConfig);
      case "sum":
        return executor.getSum(macroConfig);
      case "matched_latest_directory":
        return executor.getMatchedLatestDirectory(macroConfig);
      case "date_minus_days":
        return executor.getDateMinusDays(macroConfig);
      case "hdfs_partitions_by_dates":
        return executor.getHdfsPartitionsByDates(macroConfig);
      default:
        throw new InvalidMacro("Unknown macro " + macroConfig.getString("macro"));
    }
  }

  private void cleanUp() {
    Properties props = System.getProperties();
    for (String key: keysToCleanUp) {
      props.remove(key);
    }
    System.setProperties(props);
    keysToCleanUp.clear();
  }
}
