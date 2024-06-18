/**
 * Copyright 2021-2023, XGBoost contributors
 */
#include "file_iterator.h"

#include <xgboost/logging.h>  // for LogCheck_EQ, LogCheck_LE, CHECK_EQ, CHECK_LE, LOG, LOG_...

#include <filesystem>  // for weakly_canonical, path, u8path
#include <map>         // for map, operator==
#include <ostream>     // for operator<<, basic_ostream, istringstream
#include <vector>      // for vector

#include "../common/common.h"     // for Split
#include "xgboost/string_view.h"  // for operator<<, StringView

namespace xgboost::data {
std::string ValidateFileFormat(std::string const& uri) {
  std::vector<std::string> name_args_cache = common::Split(uri, '#');
  CHECK_LE(name_args_cache.size(), 2)
      << "Only one `#` is allowed in file path for cachefile specification";

  std::vector<std::string> name_args = common::Split(name_args_cache[0], '?');
  StringView msg{"URI parameter `format` is required for loading text data: filename?format=csv"};
  CHECK_EQ(name_args.size(), 2) << msg;

  std::map<std::string, std::string> args;
  std::vector<std::string> arg_list = common::Split(name_args[1], '&');
  for (size_t i = 0; i < arg_list.size(); ++i) {
    std::istringstream is(arg_list[i]);
    std::pair<std::string, std::string> kv;
    CHECK(std::getline(is, kv.first, '=')) << "Invalid uri argument format"
                                           << " for key in arg " << i + 1;
    CHECK(std::getline(is, kv.second)) << "Invalid uri argument format"
                                       << " for value in arg " << i + 1;
    args.insert(kv);
  }
  if (args.find("format") == args.cend()) {
    LOG(FATAL) << msg;
  }

  auto path = common::Split(uri, '?')[0];

  namespace fs = std::filesystem;
  name_args[0] = fs::weakly_canonical(fs::u8path(path)).string();
  if (name_args_cache.size() == 1) {
    return name_args[0] + "?" + name_args[1];
  } else {
    return name_args[0] + "?" + name_args[1] + '#' + name_args_cache[1];
  }
}
}  // namespace xgboost::data
