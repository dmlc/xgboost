/*!
 * Copyright 2014-2019 by Contributors
 * \file config.h
 * \brief helper class to load in configures from file
 * \author Haoda Fu, Hyunsu Cho
 */
#ifndef XGBOOST_COMMON_CONFIG_H_
#define XGBOOST_COMMON_CONFIG_H_

#include <string>
#include <fstream>
#include <istream>
#include <sstream>
#include <vector>
#include <regex>
#include <iterator>
#include <utility>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {
/*!
 * \brief Implementation of config reader
 */
class ConfigParser {
 public:
  /*!
   * \brief Constructor for INI-style configuration parser
   * \param path path to configuration file
   */
  explicit ConfigParser(const std::string path)
      : path_(std::move(path)),
      line_comment_regex_("^#"),
      key_regex_(R"rx(^([^#"'=\r\n\t ]+)[\t ]*=)rx"),
      key_regex_escaped_(R"rx(^(["'])([^"'=\r\n]+)\1[\t ]*=)rx"),
      value_regex_(R"rx(^([^#"'\r\n\t ]+)[\t ]*(?:#.*){0,1}$)rx"),
      value_regex_escaped_(R"rx(^(["'])([^"'\r\n]+)\1[\t ]*(?:#.*){0,1}$)rx")
  {}

  std::string LoadConfigFile(const std::string& path) {
    std::ifstream fin(path, std::ios_base::in | std::ios_base::binary);
    CHECK(fin) << "Failed to open config file: \"" << path << "\"";
    try {
      std::string content{std::istreambuf_iterator<char>(fin),
                          std::istreambuf_iterator<char>()};
      return content;
    } catch (std::ios_base::failure const &e) {
      LOG(FATAL) << "Failed to read config file: \"" << path << "\"\n"
                 << e.what();
    }
    return "";
  }

  /*!
   * \brief Normalize end-of-line in a file so that it uses LF for all
   *        line endings.
   *
   * This is needed because some OSes use CR or CR LF instead.  So we
   * replace all CR with LF.
   *
   * \param p_config_str pointer to configuration
   */
  std::string NormalizeConfigEOL(std::string const& config_str) {
    std::string result;
    std::stringstream ss(config_str);
    for (auto c : config_str) {
      if (c == '\r') {
        result.push_back('\n');
        continue;
      }
      result.push_back(c);
    }
    return result;
  }

  /*!
   * \brief Parse configuration file into key-value pairs.
   * \param path path to configuration file
   * \return list of key-value pairs
   */
  std::vector<std::pair<std::string, std::string>> Parse() {
    std::string content { LoadConfigFile(path_) };
    content = NormalizeConfigEOL(content);
    std::stringstream ss { content };
    std::vector<std::pair<std::string, std::string>> results;
    std::string line;
    std::string key, value;
    // Loop over every line of the configuration file
    while (std::getline(ss, line)) {
      if (ParseKeyValuePair(line, &key, &value)) {
        results.emplace_back(key, value);
      }
    }
    return results;
  }

 private:
  std::string path_;
  const std::regex line_comment_regex_, key_regex_, key_regex_escaped_,
    value_regex_, value_regex_escaped_;

 public:
  /*!
   * \brief Remove leading and trailing whitespaces from a given string
   * \param str string
   * \return Copy of str with leading and trailing whitespaces removed
   */
  static std::string TrimWhitespace(const std::string& str) {
    const auto first_char = str.find_first_not_of(" \t\n\r");
    const auto last_char = str.find_last_not_of(" \t\n\r");
    if (first_char == std::string::npos) {
      // Every character in str is a whitespace
      return std::string();
    }
    CHECK_NE(last_char, std::string::npos);
    const auto substr_len = last_char + 1 - first_char;
    return str.substr(first_char, substr_len);
  }

  /*!
   * \brief Parse a key-value pair from a string representing a line
   * \param str string (cannot be multi-line)
   * \param key place to store the key, if parsing is successful
   * \param value place to store the value, if parsing is successful
   * \return Whether the parsing was successful
   */
  bool ParseKeyValuePair(const std::string& str, std::string* key,
                         std::string* value) {
    std::string buf = TrimWhitespace(str);
    if (buf.empty()) {
      return false;
    }

    /* Match key */
    std::smatch m;
    if (std::regex_search(buf, m, line_comment_regex_)) {
      // This line is a comment
      return false;
    } else if (std::regex_search(buf, m, key_regex_)) {
      // Key doesn't have whitespace or #
      CHECK_EQ(m.size(), 2);
      *key = m[1].str();
    } else if (std::regex_search(buf, m, key_regex_escaped_)) {
      // Key has a whitespace and/or #; it has to be wrapped around a pair of
      // single or double quotes. Example: "foo bar"  'foo#bar'
      CHECK_EQ(m.size(), 3);
      *key = m[2].str();
    } else {
      LOG(FATAL) << "This line is not a valid key-value pair: " << str;
    }

    /* Match value */
    buf = m.suffix().str();
    buf = TrimWhitespace(buf);
    if (std::regex_search(buf, m, value_regex_)) {
      // Value doesn't have whitespace or #
      CHECK_EQ(m.size(), 2);
      *value = m[1].str();
    } else if (std::regex_search(buf, m, value_regex_escaped_)) {
      // Value has a whitespace and/or #; it has to be wrapped around a pair of
      // single or double quotes. Example: "foo bar"  'foo#bar'
      CHECK_EQ(m.size(), 3);
      *value = m[2].str();
    } else {
      LOG(FATAL) << "This line is not a valid key-value pair: " << str;
    }
    return true;
  }
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CONFIG_H_
