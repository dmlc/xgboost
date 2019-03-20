package ml.dmlc.xgboost4j.java;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoSerializable;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class BoosterResults implements Serializable, KryoSerializable {
  private Booster booster;
  private String[] logInfos;

  public BoosterResults(Booster booster, String[] logInfos) {
    this.booster = booster;
    this.logInfos = logInfos;
  }

  public Booster getBooster() {
    return this.booster;
  }

  public String[] getLogInfos() {
    return this.logInfos;
  }

  @Override
  public void write(Kryo kryo, Output output) {
    kryo.writeObject(output, booster);
    kryo.writeObject(output, logInfos);
  }

  @Override
  public void read(Kryo kryo, Input input) {
    booster = kryo.readObject(input, Booster.class);
    logInfos = kryo.readObject(input, String[].class);
  }
}
