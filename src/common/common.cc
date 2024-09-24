/**
 * Copyright 2015-2024, XGBoost Contributors
 */
#include "common.h"

#include <dmlc/thread_local.h>  // for ThreadLocalStore

#include <cmath>    // for pow
#include <cstdint>  // for uint8_t
#include <cstdio>   // for snprintf, size_t
#include <string>   // for string
#include <utility>  // for pair

#include "./random.h"  // for GlobalRandomEngine, GlobalRandom

namespace xgboost::common {
/*! \brief thread local entry for random. */
struct RandomThreadLocalEntry {
  /*! \brief the random engine instance. */
  GlobalRandomEngine engine;
};

using RandomThreadLocalStore = dmlc::ThreadLocalStore<RandomThreadLocalEntry>;

GlobalRandomEngine &GlobalRandom() { return RandomThreadLocalStore::Get()->engine; }

void EscapeU8(std::string const &string, std::string *p_buffer) {
  auto &buffer = *p_buffer;
  for (size_t i = 0; i < string.length(); i++) {
    const auto ch = string[i];
    if (ch == '\\') {
      if (i < string.size() && string[i + 1] == 'u') {
        buffer += "\\";
      } else {
        buffer += "\\\\";
      }
    } else if (ch == '"') {
      buffer += "\\\"";
    } else if (ch == '\b') {
      buffer += "\\b";
    } else if (ch == '\f') {
      buffer += "\\f";
    } else if (ch == '\n') {
      buffer += "\\n";
    } else if (ch == '\r') {
      buffer += "\\r";
    } else if (ch == '\t') {
      buffer += "\\t";
    } else if (static_cast<uint8_t>(ch) <= 0x1f) {
      // Unit separator
      char buf[8];
      snprintf(buf, sizeof buf, "\\u%04x", ch);
      buffer += buf;
    } else {
      buffer += ch;
    }
  }
}

std::string HumanMemUnit(std::size_t n_bytes) {
  auto n_bytes_f64 = static_cast<double>(n_bytes);
  double constexpr k1024 = 1024.0;
  using P = std::pair<std::int32_t, StringView>;
  std::stringstream ss;
  for (auto pu : {P{3, "GB"}, P{2, "MB"}, P{1, "KB"}}) {
    auto const [power, unit] = pu;  // NOLINT
    if (n_bytes_f64 >= (std::pow(k1024, power))) {
      ss << (n_bytes_f64 / std::pow(k1024, power)) << unit;
      return ss.str();
    }
  }
  ss << n_bytes_f64 << "B";
  return ss.str();
}
}  // namespace xgboost::common
