package ml.dmlc.xgboost4j.java;

import java.io.IOException;

/**
 * Load an entity.
 */
public interface Loadable {
  boolean load() throws IOException;

  boolean isLoaded();

  String getName();
}
