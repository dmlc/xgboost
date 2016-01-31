/*!
 * Copyright 2015 by Contributors
 * \file common.cc
 * \brief Enable all kinds of global variables in common.
 */
#include "./random.h"

namespace xgboost {
namespace common {
GlobalRandomEngine& GlobalRandom() {
  static GlobalRandomEngine inst;
  return inst;
}
}
}  // namespace xgboost
