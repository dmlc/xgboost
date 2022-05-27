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

import ml.dmlc.xgboost4j.java.NativeLibLoader.OS;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.Collection;

import static java.util.Arrays.asList;
import static junit.framework.TestCase.assertSame;
import static ml.dmlc.xgboost4j.java.NativeLibLoader.OS.*;
import static org.junit.Assert.assertThrows;

/**
 * Test cases for {@link OS}.
 */
@RunWith(Enclosed.class)
public class OsDetectionTest {

  private static final String OS_NAME_PROPERTY = "os.name";

  @RunWith(Parameterized.class)
  public static class SupportedOSDetectionTest {

    private final String osNameValue;
    private final OS expectedOS;

    public SupportedOSDetectionTest(String osNameValue, OS expectedOS) {
      this.osNameValue = osNameValue;
      this.expectedOS = expectedOS;
    }

    @Parameters
    public static Collection<Object[]> data() {
      return asList(new Object[][]{
        {"windows", WINDOWS},
        {"mac", MACOS},
        {"darwin", MACOS},
        {"linux", LINUX},
        {"sunos", SOLARIS}
      });
    }

    @Test
    public void getOS() {
      executeAndRestoreProperty(() -> {
        System.setProperty(OS_NAME_PROPERTY, osNameValue);
        assertSame(detectOS(), expectedOS);
      });
    }
  }

  public static class UnsupportedOSDetectionTest {

    @Test
    public void testUnsupportedOs() {
      executeAndRestoreProperty(() -> {
        System.setProperty(OS_NAME_PROPERTY, "unsupported");
        assertThrows(IllegalStateException.class, OS::detectOS);
      });
    }
  }

  private static void executeAndRestoreProperty(Runnable action) {
    String oldValue = System.getProperty(OS_NAME_PROPERTY);

    try {
      action.run();
    } finally {
      if (oldValue != null) {
        System.setProperty(OS_NAME_PROPERTY, oldValue);
      } else {
        System.clearProperty(OS_NAME_PROPERTY);
      }
    }
  }

}
