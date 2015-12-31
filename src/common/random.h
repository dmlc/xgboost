/*!
 * Copyright 2015 by Contributors
 * \file random.h
 * \brief Utility related to random.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_RANDOM_H_
#define XGBOOST_COMMON_RANDOM_H_

#include <random>

namespace xgboost {
namespace common {
/*!
 * \brief Random Engine
 */
typedef std::mt19937 RandomEngine;
/*!
 * \brief global singleton of a random engine.
 *  Only use this engine when necessary, not thread-safe.
 */
static RandomEngine* GlobalRandom();

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANDOM_H_
