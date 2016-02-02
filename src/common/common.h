/*!
 * Copyright 2015 by Contributors
 * \file common.h
 * \brief Common utilities
 */
#ifndef XGBOOST_COMMON_COMMON_H_
#define XGBOOST_COMMON_COMMON_H_

#include <vector>
#include <string>
#include <sstream>

namespace xgboost {
namespace common {
/*!
 * \brief Split a string by delimiter
 * \param s String to be splitted.
 * \param delim The delimiter.
 */
inline std::vector<std::string> Split(const std::string& s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COMMON_H_
