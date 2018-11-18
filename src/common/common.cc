/*!
 * Copyright 2015-2018 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include <dmlc/thread_local.h>

#include "common.h"
#include "./random.h"

namespace xgboost {
namespace common {
/*! \brief thread local entry for random. */
struct RandomThreadLocalEntry {
  /*! \brief the random engine instance. */
  GlobalRandomEngine engine;
};

using RandomThreadLocalStore = dmlc::ThreadLocalStore<RandomThreadLocalEntry>;

bool EndsWith(std::string const& str, std::string const& end) {
  size_t end_size = end.size();
  if (end_size > str.size()) { return false; }
  std::string substr = str.substr(str.size() - end_size);
  return substr == end;
}

GlobalRandomEngine& GlobalRandom() {
  return RandomThreadLocalStore::Get()->engine;
}
}  // namespace common

#if !defined(XGBOOST_USE_CUDA)
int AllVisibleImpl::AllVisible() {
  return 0;
}
#endif

}  // namespace xgboost
