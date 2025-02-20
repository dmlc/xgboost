/*
 Copyright (c) 2025 by Contributors

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

import junit.framework.TestCase;
import org.junit.Test;

/**
 * Test cases for the config context.
 */
public class ConfigContextTest {
  @Test
  public void testBasic() throws XGBoostError {
    try (ConfigContext ctx = new ConfigContext().set("verbosity", 3)) {
      int v = (int) ConfigContext.get().get("verbosity");
      TestCase.assertEquals(3, v);
    }
    int v = (int) ConfigContext.get().get("verbosity");
    TestCase.assertEquals(1, v);
  }
}
