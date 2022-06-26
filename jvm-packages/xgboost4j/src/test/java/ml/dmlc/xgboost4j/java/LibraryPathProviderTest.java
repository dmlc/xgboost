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

import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.Arch.X86_64;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.LibraryPathProvider.getLibraryPathFor;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.OS.LINUX;

public class LibraryPathProviderTest {

  @Test
  public void testLibraryPathProviderUsesOsAndArchToResolvePath() {
    String libraryPath = getLibraryPathFor(LINUX, X86_64, "someLibrary");

    assertTrue(libraryPath.startsWith("/lib/linux/x86_64/"));
  }

  @Test
  public void testLibraryPathProviderUsesPropertyValueForPathIfPresent() {
    String propertyName = "xgboostruntime.native.library";

    executeAndRestoreProperty(propertyName, () -> {
      System.setProperty(propertyName, "/my/custom/path/to/my/library");
      String libraryPath = getLibraryPathFor(LINUX, X86_64, "library");

      assertEquals("/my/custom/path/to/my/library", libraryPath);
    });
  }

  private static void executeAndRestoreProperty(String propertyName, Runnable action) {
    String oldValue = System.getProperty(propertyName);

    try {
      action.run();
    } finally {
      if (oldValue != null) {
        System.setProperty(propertyName, oldValue);
      } else {
        System.clearProperty(propertyName);
      }
    }
  }

}
