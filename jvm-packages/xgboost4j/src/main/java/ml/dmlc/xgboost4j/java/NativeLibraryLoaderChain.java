/*
 Copyright (c) 2014 by Contributors

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
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * A simple loader which tries to load all
 * specified libraries in given order.
 */
public class NativeLibraryLoaderChain implements Loadable {

  private static final Log logger = LogFactory.getLog(NativeLibraryLoaderChain.class);

  private final Loadable[] nativeLibs;

  private boolean loaded = false;

  private NativeLibraryLoaderChain(Loadable[] libs) {
    assert libs != null : "Argument `libs` cannot be null.";
    nativeLibs = libs;
  }

  @Override
  public synchronized boolean load() throws IOException {
    if (!isLoaded()) {
      ArrayList<IOException> exs = new ArrayList<>();
      for (Loadable lib : nativeLibs) {
        try {
          if (loaded = lib.load()) break;
        } catch (IOException e) {
          logger.info("Cannot load library: " + lib.toString());
          exs.add(e);
        }
      }
      if (!isLoaded() && !exs.isEmpty()) { // Try to load something but failed
        throw new IOException(exs.get(exs.size()-1));
      }
    }
    return isLoaded();
  }

  @Override
  public boolean isLoaded() {
    return loaded;
  }

  public static NativeLibraryLoaderChain loaderChain(Loadable ...libs) {
    return new NativeLibraryLoaderChain(libs);
  }
}