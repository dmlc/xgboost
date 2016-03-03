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
package ml.dmlc.xgboost4j;

import java.io.*;
import java.lang.reflect.Field;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import ml.dmlc.xgboost4j.FileUtil;


/**
 * class to load native library
 *
 * @author hzx
 */
class NativeLibLoader {
  private static final Log logger = LogFactory.getLog(NativeLibLoader.class);

  private static boolean initialized = false;
  private static final String nativePath = "../lib/";
  private static final String nativeResourcePath = "/lib/";
  private static final String[] libNames = new String[]{"xgboost4j"};

  public static synchronized void initXgBoost() throws IOException {
    if (!initialized) {
      for (String libName : libNames) {
        smartLoad(libName);
      }
      initialized = true;
    }
  }

  /**
   * Loads library from current JAR archive
   * <p/>
   * The file from JAR is copied into system temporary directory and then loaded.
   * The temporary file is deleted after exiting.
   * Method uses String as filename because the pathname is "abstract", not system-dependent.
   * <p/>
   * The restrictions of {@link File#createTempFile(java.lang.String, java.lang.String)} apply to
   * {@code path}.
   *
   * @param path The filename inside JAR as absolute path (beginning with '/'),
   *             e.g. /package/File.ext
   * @throws IOException              If temporary file creation or read/write operation fails
   * @throws IllegalArgumentException If source file (param path) does not exist
   * @throws IllegalArgumentException If the path is not absolute or if the filename is shorter than
   * three characters
   */
  private static void loadLibraryFromJar(String path) throws IOException, IllegalArgumentException{
    File temp = FileUtil.createTempFileFromResource(path);
    // Finally, load the library
    System.load(temp.getAbsolutePath());
  }

  /**
   * load native library, this method will first try to load library from java.library.path, then
   * try to load library in jar package.
   *
   * @param libName library path
   * @throws IOException exception
   */
  private static void smartLoad(String libName) throws IOException {
    addNativeDir(nativePath);
    try {
      System.loadLibrary(libName);
    } catch (UnsatisfiedLinkError e) {
      try {
        String libraryFromJar = nativeResourcePath + System.mapLibraryName(libName);
        loadLibraryFromJar(libraryFromJar);
      } catch (IOException e1) {
        throw e1;
      }
    }
  }

  /**
   * Add libPath to java.library.path, then native library in libPath would be load properly
   *
   * @param libPath library path
   * @throws IOException exception
   */
  private static void addNativeDir(String libPath) throws IOException {
    try {
      Field field = ClassLoader.class.getDeclaredField("usr_paths");
      field.setAccessible(true);
      String[] paths = (String[]) field.get(null);
      for (String path : paths) {
        if (libPath.equals(path)) {
          return;
        }
      }
      String[] tmp = new String[paths.length + 1];
      System.arraycopy(paths, 0, tmp, 0, paths.length);
      tmp[paths.length] = libPath;
      field.set(null, tmp);
    } catch (IllegalAccessException e) {
      logger.error(e.getMessage());
      throw new IOException("Failed to get permissions to set library path");
    } catch (NoSuchFieldException e) {
      logger.error(e.getMessage());
      throw new IOException("Failed to get field handle to set library path");
    }
  }
}
