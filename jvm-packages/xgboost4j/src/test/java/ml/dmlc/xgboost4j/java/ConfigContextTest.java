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

import java.util.HashMap;

/**
 * Test cases for the config context.
 */
public class ConfigContextTest {
  @Test
  public void testBasic() throws XGBoostError {
    try (ConfigContext ctx = new ConfigContext()) {
      TestCase.assertEquals(1, ctx.getConfig("verbosity"));

      ctx.setConfig("verbosity", 3);
      TestCase.assertEquals(3, ctx.getConfig("verbosity"));
    }
  }

  @Test
  public void testWriteMap() throws XGBoostError {
    try (ConfigContext ctx = new ConfigContext()) {
      TestCase.assertEquals(1, ctx.getConfig("verbosity"));
      TestCase.assertEquals(false, ctx.getConfig("use_rmm"));
    }

    HashMap<String, Object> configs = new HashMap<>();
    configs.put("verbosity", 3);
    configs.put("use_rmm", true);
    try (ConfigContext ctx = new ConfigContext(configs)) {
      TestCase.assertEquals(3, ctx.getConfig("verbosity"));
      TestCase.assertEquals(true, ctx.getConfig("use_rmm"));
    }

    try (ConfigContext ctx = new ConfigContext()) {
      TestCase.assertEquals(1, ctx.getConfig("verbosity"));
      TestCase.assertEquals(false, ctx.getConfig("use_rmm"));
    }
  }
}
