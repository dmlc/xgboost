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

import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Global configuration context for XGBoost.
 *
 * @version 3.0.0
 * <p>
 * See the parameter document for supported global configuration. The configuration is
 * restored upon close.
 */
public class ConfigContext implements AutoCloseable {
  private final String initialConfiguration;

  public ConfigContext() throws XGBoostError {
    initialConfiguration = getGlobalConfig();
  }

  /* Set the parameters during initializing */
  public ConfigContext(Map<String, Object> params) throws XGBoostError {
    if (params != null && !params.isEmpty()) {
      initialConfiguration = getGlobalConfig();
      setConfigs(params);
    } else {
      initialConfiguration = null;
    }
  }

  /**
   * Get the global configuration
   */
  private String getGlobalConfig() throws XGBoostError {
    String[] config = new String[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGBGetGlobalConfig(config));
    return config[0];
  }

  public Object getConfig(String name) throws XGBoostError {
    String jconfig = getGlobalConfig();
    ObjectMapper mapper = new ObjectMapper();
    try {
      Map<String, Object> map = mapper.readValue(jconfig,
        new TypeReference<Map<String, Object>>() {
        });
      return map.get(name);
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to get the global config due to a decode error.", ex);
    }
  }

  /** Set one single configuration */
  public void setConfig(String key, Object value) throws XGBoostError {
    HashMap<String, Object> configs = new HashMap<>();
    configs.put(key, value);
    ObjectMapper mapper = new ObjectMapper();
    try {
      String config = mapper.writeValueAsString(configs);
      XGBoostJNI.checkCall(XGBoostJNI.XGBSetGlobalConfig(config));
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to set the global config due to an encode error.", ex);
    }
  }

  /** Set a bunch of configurations */
  public void setConfigs(Map<String, Object> configs) throws XGBoostError {
    ObjectMapper mapper = new ObjectMapper();
    try {
      String config = mapper.writeValueAsString(configs);
      XGBoostJNI.checkCall(XGBoostJNI.XGBSetGlobalConfig(config));
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to set the global config due to an encode error.", ex);
    }
  }

  @Override
  public void close() throws XGBoostError {
    if (initialConfiguration != null) {
      XGBoostJNI.checkCall(XGBoostJNI.XGBSetGlobalConfig(initialConfiguration));
    }
  }
};
