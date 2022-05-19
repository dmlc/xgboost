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
