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
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.Collection;

import static java.util.Arrays.asList;
import static junit.framework.TestCase.assertSame;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.Arch.X86_64;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.Arch.AARCH64;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.Arch.SPARC;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.Arch.detectArch;
import static org.junit.Assert.assertThrows;

/**
 * Test cases for {@link NativeLibLoader.Arch}.
 */
@RunWith(Enclosed.class)
public class ArchDetectionTest {

  private static final String OS_ARCH_PROPERTY = "os.arch";

  @RunWith(Parameterized.class)
  public static class ParameterizedArchDetectionTest {

    private final String osArchValue;
    private final NativeLibLoader.Arch expectedArch;

    public ParameterizedArchDetectionTest(String osArchValue, NativeLibLoader.Arch expectedArch) {
      this.osArchValue = osArchValue;
      this.expectedArch = expectedArch;
    }

    @Parameters
    public static Collection<Object[]> data() {
      return asList(new Object[][]{
        {"x86_64", X86_64},
        {"amd64", X86_64},
        {"aarch64", AARCH64},
        {"arm64", AARCH64},
        {"sparc64", SPARC}
      });
    }

    @Test
    public void testArch() {
      executeAndRestoreProperty(() -> {
        System.setProperty(OS_ARCH_PROPERTY, osArchValue);
        assertSame(detectArch(), expectedArch);
      });
    }
  }

  public static class UnsupportedArchDetectionTest {

    @Test
    public void testUnsupportedArch() {
      executeAndRestoreProperty(() -> {
        System.setProperty(OS_ARCH_PROPERTY, "unsupported");
        assertThrows(IllegalStateException.class, NativeLibLoader.Arch::detectArch);
      });
    }
  }

  private static void executeAndRestoreProperty(Runnable action) {
    String oldValue = System.getProperty(OS_ARCH_PROPERTY);

    try {
      action.run();
    } finally {
      if (oldValue != null) {
        System.setProperty(OS_ARCH_PROPERTY, oldValue);
      } else {
        System.clearProperty(OS_ARCH_PROPERTY);
      }
    }
  }

}
