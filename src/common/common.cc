/*!
 * Copyright 2015 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include <dmlc/thread_local.h>
#include "./random.h"

namespace xgboost {
namespace common {
/*! \brief thread local entry for random. */
struct RandomThreadLocalEntry {
  /*! \brief the random engine instance. */
  GlobalRandomEngine engine;
};

typedef dmlc::ThreadLocalStore<RandomThreadLocalEntry> RandomThreadLocalStore;

GlobalRandomEngine& GlobalRandom() {
  return RandomThreadLocalStore::Get()->engine;
}
}  // namespace common
}  // namespace xgboost
