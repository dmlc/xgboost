package ml.dmlc.xgboost4j.java;

import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.Arch.X86_64;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.LibraryPathProvider.getLibraryPathFor;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.OS.LINUX;

public class LibraryPathProviderTest {

  @Test
  public void testLibraryPathProvider() {
    String libraryPath = getLibraryPathFor(LINUX, X86_64, "someLibrary");

    assertTrue(libraryPath.startsWith("/lib/linux/x86_64/"));
  }

}
