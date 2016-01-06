/*!
 * Copyright 2015 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include "./random.h"

namespace xgboost {
namespace common {
RandomEngine& GlobalRandom() {
  static RandomEngine inst;
  return inst;
}
}
}  // namespace xgboost
