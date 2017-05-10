package com.airbnb.common.config;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.common.base.Joiner;
import com.typesafe.config.Config;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

public class AirConMacroExecutor implements MacroExecutor {

  static private final DateTimeFormatter DATE_FORMATTER = DateTimeFormat.forPattern("yyyy-MM-dd");

  AirConMacroExecutor() {}

  @Override
  public String getMaxString(Config config) {
    return Collections.max(config.getStringList("input"));
  }

  @Override
  public String getMinString(Config config) {
    return Collections.min(config.getStringList("input"));
  }

  @Override
  public String getSum(Config config) {
    Long sum = 0L;
    for (Integer el : config.getIntList("input")) sum += el;
    return sum.toString();
  }

  @Override
  public String getMatchedLatestDirectory(Config config) {
    final String javaRegexPattern = config.getString("java_regex_pattern");
    final String dirName = config.getString("dir_name");
    return getMatchedLatestDirectory(dirName, javaRegexPattern);
  }

  private String getMatchedLatestDirectory(String dirName, String javaRegexPattern) {
    Pattern pattern = Pattern.compile(javaRegexPattern);
    final String latestFile = getLatestInDirectory(dirName);
    if (latestFile == null) {
      return null;
    }
    Matcher matcher = pattern.matcher(latestFile);

    if (matcher.find()) {
      try {
        return matcher.group(1);
      } catch (IndexOutOfBoundsException e) {
        throw new AirCon.InvalidRegexPattern(
            "There's no group in the regex " + javaRegexPattern);
      }
    } else {
      return null;
    }
  }

  private String getLatestInDirectory(String hdfsDir) {
    List<String> allFileNames = getAllDirOrFiles(hdfsDir);
    if (allFileNames == null) {
      return null;
    }

    if (allFileNames.isEmpty()) {
      return null;
    } else {
      Collections.sort(allFileNames);
      return allFileNames.get(allFileNames.size() - 1);
    }
  }

  protected List<String> getAllDirOrFiles(String hdfsDir) {
    List<String> allFileNames = new ArrayList<>();
    try {
      FileSystem fs = FileSystem.get(new Path(hdfsDir).toUri(), new Configuration());
      FileStatus[] allFiles = fs.listStatus(new Path(hdfsDir));
      for (FileStatus fileStatus : allFiles) {
        String filename = fileStatus.getPath().getName();
        allFileNames.add(filename);
      }
    } catch (IOException e) {
      System.err.println(e.getMessage());
      return null;
    }
    return allFileNames;
  }

  private void assertMacroParam(Config config, String key) {
    if (!config.hasPath(key)) {
      throw new AirCon.InvalidMacro(key + " param is missing");
    }
  }

  @Override
  public String getDateMinusDays(Config config) {
    assertMacroParam(config, "date");
    assertMacroParam(config, "minus_days");
    DateTime date = DateTime.parse(config.getString("date"), DATE_FORMATTER);
    return date.minusDays(config.getInt("minus_days")).toString(DATE_FORMATTER);
  }

  @Override
  public String getHdfsPartitionsByDates(Config config) {
    assertMacroParam(config, "dir_name");
    assertMacroParam(config, "date_regex_pattern");
    assertMacroParam(config, "start_date");
    assertMacroParam(config, "end_date");

    Pattern pattern = Pattern.compile(config.getString("date_regex_pattern"));
    DateTime startDate = DateTime.parse(config.getString("start_date"), DATE_FORMATTER);
    DateTime endDate = DateTime.parse(config.getString("end_date"), DATE_FORMATTER);

    ArrayList<String> matchedNames = new ArrayList<>();
    String dirName = config.getString("dir_name");
    for (String dirOrFileName : getAllDirOrFiles(dirName)) {
      Matcher matcher = pattern.matcher(dirOrFileName);
      if (matcher.find()) {
        try {
          DateTime date = DateTime.parse(matcher.group(1), DATE_FORMATTER);
          if (date.compareTo(startDate) >= 0 && date.compareTo(endDate) <= 0) {
            matchedNames.add(dirName + "/" + dirOrFileName + "/part-*.gz");
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }
    return Joiner.on(",").join(matchedNames);
  }
}
