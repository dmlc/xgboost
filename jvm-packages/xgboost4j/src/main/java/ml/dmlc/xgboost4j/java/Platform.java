package ml.dmlc.xgboost4j.java;

/**
 * A simple OS type wrapper.
 */
public enum Platform {

  OSX("osx"),
  LINUX("linux"),
  WINDOWS("windows"),
  UNKNOWN("unknown");

  private final String name;
  private final int bits;

  Platform(String name) {
    this.name = name;
    this.bits = getBitModel();
  }

  public String getName() {
    return name;
  }

  public String getPlatform() {
    return name + "_" + bits;
  }

  public String getPlatformLibName(String libName) {
    return System.mapLibraryName(libName);
  }

  public static Platform geOSType() {
    String name = System.getProperty("os.name").toLowerCase().trim();
    if (name.startsWith("linux")) {
      return LINUX;
    }
    if (name.startsWith("mac os x")) {
      return OSX;
    }
    if (name.startsWith("win")) {
      return WINDOWS;
    }
    return UNKNOWN;
  }

  private static int getBitModel() {
    String prop = System.getProperty("sun.arch.data.model");
    if (prop == null) {
      prop = System.getProperty("com.ibm.vm.bitmode");
    }
    if (prop != null) {
      return Integer.parseInt(prop);
    }
    return -1;
  }
}
