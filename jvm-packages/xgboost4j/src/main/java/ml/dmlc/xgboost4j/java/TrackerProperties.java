package ml.dmlc.xgboost4j.java;

import java.io.*;
import java.net.URL;
import java.util.Properties;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class TrackerProperties {
  private static String PROPERTIES_FILENAME = "xgboost-tracker.properties";
  private static String HOST_IP = "host-ip";

  private static final Log logger = LogFactory.getLog(TrackerProperties.class);
  private static TrackerProperties instance = new TrackerProperties();

  private Properties properties;

  private TrackerProperties() {
    this.properties = new Properties();

    InputStream inputStream = null;

    try {
      URL propertiesFileURL =
          Thread.currentThread().getContextClassLoader().getResource(PROPERTIES_FILENAME);
      if (propertiesFileURL != null){
        inputStream = propertiesFileURL.openStream();
      }
    } catch (IOException e) {
      logger.warn("Could not load " + PROPERTIES_FILENAME + " file. ", e);
    }

    if(inputStream != null){
      try {
        properties.load(inputStream);
        logger.debug("Loaded properties from external source");
      } catch (IOException e) {
        logger.error("Error loading tracker properties file. Skipping and using defaults. ", e);
      }
      try {
        inputStream.close();
      } catch (IOException e) {
        // ignore exception
      }
    }
  }

  public static TrackerProperties getInstance() {
    return instance;
  }

  public String getHostIp(){
    return this.properties.getProperty(HOST_IP);
  }
}
