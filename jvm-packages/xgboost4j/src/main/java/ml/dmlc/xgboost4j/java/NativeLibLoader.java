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

import java.io.*;
import java.lang.reflect.Field;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * class to load native library
 *
 * @author hzx
 */
class NativeLibLoader {
  private static final Log logger = LogFactory.getLog(NativeLibLoader.class);

  private static boolean initialized = false;
  private static final String nativePath = "../../lib/";
  private static final String nativeResourcePath = "/lib/";
  private static final String[] libNames = new String[]{"xgboost4j"};

  static synchronized void initXGBoost() throws IOException {
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
    String temp = createTempFileFromResource(path);
    // Finally, load the library
    System.load(temp);
  }

  /**
   * Create a temp file that copies the resource from current JAR archive
   * <p/>
   * The file from JAR is copied into system temp file.
   * The temporary file is deleted after exiting.
   * Method uses String as filename because the pathname is "abstract", not system-dependent.
   * <p/>
   * The restrictions of {@link File#createTempFile(java.lang.String, java.lang.String)} apply to
   * {@code path}.
   * @param path Path to the resources in the jar
   * @return The created temp file.
   * @throws IOException
   * @throws IllegalArgumentException
   */
  static String createTempFileFromResource(String path) throws
          IOException, IllegalArgumentException {
    // Obtain filename from path
    if (!path.startsWith("/")) {
      throw new IllegalArgumentException("The path has to be absolute (start with '/').");
    }

    String[] parts = path.split("/");
    String filename = (parts.length > 1) ? parts[parts.length - 1] : null;

    // Split filename to prexif and suffix (extension)
    String prefix = "";
    String suffix = null;
    if (filename != null) {
      parts = filename.split("\\.", 2);
      prefix = parts[0];
      suffix = (parts.length > 1) ? "." + parts[parts.length - 1] : null; // Thanks, davs! :-)
    }

    // Check if the filename is okay
    if (filename == null || prefix.length() < 3) {
      throw new IllegalArgumentException("The filename has to be at least 3 characters long.");
    }
    // Prepare temporary file
    File temp = File.createTempFile(prefix, suffix);
    temp.deleteOnExit();

    if (!temp.exists()) {
      throw new FileNotFoundException("File " + temp.getAbsolutePath() + " does not exist.");
    }

    // Prepare buffer for data copying
    byte[] buffer = new byte[1024];
    int readBytes;

    // Open and check input stream
    InputStream is = NativeLibLoader.class.getResourceAsStream(path);
    if (is == null) {
      throw new FileNotFoundException("File " + path + " was not found inside JAR.");
    }

    // Open output stream and copy data between source file in JAR and the temporary file
    OutputStream os = new FileOutputStream(temp);
    try {
      while ((readBytes = is.read(buffer)) != -1) {
        os.write(buffer, 0, readBytes);
      }
    } finally {
      // If read/write fails, close streams safely before throwing an exception
      os.close();
      is.close();
    }
    return temp.getAbsolutePath();
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
      } catch (IOException ioe) {
        logger.error("failed to load library from both native path and jar");
        throw ioe;
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
