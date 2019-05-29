/*!
 * Copyright 2019 by Contributors
 * \file config.h
 * \brief helper class to load in configures from file
 * \author Haoda Fu
 */
#ifndef XGBOOST_COMMON_CONFIG_H_
#define XGBOOST_COMMON_CONFIG_H_

#include <cstdio>
#include <cstring>
#include <string>
#include <istream>
#include <fstream>
#include <vector>
#include <utility>

namespace xgboost {
namespace common {
/*!
 * \brief Implementation of config reader
 */
class ConfigParse {
 public:
  /*!
* \brief constructor
* \param cfgFileName name of configure file
*/

  explicit ConfigParse(const std::string &cfgFileName) {
    fi_.open(cfgFileName);
    if (fi_.fail()) {
      LOG(FATAL) << "cannot open file " << cfgFileName;
    }
  }

  /*!
  * \brief parse the configure file
  */
  std::vector<std::pair<std::string, std::string> > Parse() {
    std::vector<std::pair<std::string, std::string> > results{};
    char delimiter = '=';
    char comment = '#';
    std::string line{};
    std::string name{};
    std::string value{};

    while (!fi_.eof()) {
      std::getline(fi_, line);  // read a line of configure file
      line = line.substr(0, line.find(comment));  // anything beyond # is comment
      size_t delimiterPos = line.find(delimiter);  // find the = sign
      name = line.substr(0, delimiterPos);  // anything before = is the name
      // after this = is the value
      value = line.substr(delimiterPos + 1, line.length() - delimiterPos - 1);

      if (line.empty() || name.empty() || value.empty())
        continue;  // skip a line if # at beginning or there is no value or no name.
      CleanString(&name);  // clean the string
      CleanString(&value);
      results.emplace_back(name, value);
    }
    return results;
  }

  ~ConfigParse() {
    fi_.close();
  }

 private:
  std::ifstream fi_;
  std::string allowableChar =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-./\\";

  /*!
  * \brief remove unnecessary chars.
  */
  void CleanString(std::string * str) {
    size_t firstIndx = str->find_first_of(allowableChar);
    size_t lastIndx = str->find_last_of(allowableChar);
    // this line can be more efficient, but keep as is for simplicity.
    *str = str->substr(firstIndx, lastIndx - firstIndx + 1);
  }
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_CONFIG_H_
