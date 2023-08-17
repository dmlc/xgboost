/**
 * Copyright 2015-2023 by Contributors
 */
#include "common.h"

#include <dmlc/thread_local.h>  // for ThreadLocalStore

#include <cstdint>  // for uint8_t
#include <cstdio>   // for snprintf, size_t
#include <string>   // for string

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

#if !defined(XGBOOST_USE_CUDA)
int AllVisibleGPUs() { return 0; }
#endif  // !defined(XGBOOST_USE_CUDA)

}  // namespace xgboost::common
