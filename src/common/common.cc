/*!
 * Copyright 2015 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include "./random.h"
#include "./thread_local.h"

namespace xgboost {
namespace common {
/*! \brief thread local entry for random. */
struct RandomThreadLocalEntry {
  /*! \brief the random engine instance. */
  GlobalRandomEngine engine;
};

typedef ThreadLocalStore<RandomThreadLocalEntry> RandomThreadLocalStore;

GlobalRandomEngine& GlobalRandom() {
  return RandomThreadLocalStore::Get()->engine;
}
}  // namespace common
}  // namespace xgboost
