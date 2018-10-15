/*!
 * Copyright 2015 by Contributors
 * \file random.h
 * \brief Utility related to random.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_RANDOM_H_
#define XGBOOST_COMMON_RANDOM_H_

#include <xgboost/logging.h>
#include <algorithm>
#include <vector>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include "host_device_vector.h"

namespace xgboost {
namespace common {
/*!
 * \brief Define mt19937 as default type Random Engine.
 */
using RandomEngine = std::mt19937;

#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
/*!
 * \brief An customized random engine, used to be plugged in PRNG from other systems.
 *  The implementation of this library is not provided by xgboost core library.
 *  Instead the other library can implement this class, which will be used as GlobalRandomEngine
 *  If XGBOOST_RANDOM_CUSTOMIZE = 1, by default this is switched off.
 */
class CustomGlobalRandomEngine {
 public:
  /*! \brief The result type */
  using result_type = uint32_t;
  /*! \brief The minimum of random numbers generated */
  inline static constexpr result_type min() {
    return 0;
  }
  /*! \brief The maximum random numbers generated */
  inline static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }
  /*!
   * \brief seed function, to be implemented
   * \param val The value of the seed.
   */
  void seed(result_type val);
  /*!
   * \return next random number.
   */
  result_type operator()();
};

/*!
 * \brief global random engine
 */
typedef CustomGlobalRandomEngine GlobalRandomEngine;

#else
/*!
 * \brief global random engine
 */
using GlobalRandomEngine = RandomEngine;
#endif

/*!
 * \brief global singleton of a random engine.
 *  This random engine is thread-local and
 *  only visible to current thread.
 */
GlobalRandomEngine& GlobalRandom(); // NOLINT(*)

/**
 * \class ColumnSampler
 *
 * \brief Handles selection of columns due to colsample_bytree and
 * colsample_bylevel parameters. Should be initialised before tree
 * construction and to reset when tree construction is completed.
 */

class ColumnSampler {
  HostDeviceVector<int> feature_set_tree_;
  std::map<int, HostDeviceVector<int>> feature_set_level_;
  float colsample_bylevel_{1.0f};
  float colsample_bytree_{1.0f};

  std::vector<int> ColSample(std::vector<int> features, float colsample) const {
    if (colsample == 1.0f) return features;
    CHECK_GT(features.size(), 0);
    int n = std::max(1, static_cast<int>(colsample * features.size()));

    std::shuffle(features.begin(), features.end(), common::GlobalRandom());
    features.resize(n);
    std::sort(features.begin(), features.end());

    return features;
  }

 public:
  /**
   * \brief Initialise this object before use.
   *
   * \param num_col
   * \param colsample_bylevel
   * \param colsample_bytree
   * \param skip_index_0      (Optional) True to skip index 0.
   */
  void Init(int64_t num_col, float colsample_bylevel, float colsample_bytree,
            bool skip_index_0 = false) {
    this->colsample_bylevel_ = colsample_bylevel;
    this->colsample_bytree_ = colsample_bytree;
    this->Reset();

    int begin_idx = skip_index_0 ? 1 : 0;
    auto& feature_set_h = feature_set_tree_.HostVector();
    feature_set_h.resize(num_col - begin_idx);

    std::iota(feature_set_h.begin(), feature_set_h.end(), begin_idx);
    feature_set_h = ColSample(feature_set_h, this->colsample_bytree_);
  }

  /**
   * \brief Resets this object.
   */
  void Reset() {
    feature_set_tree_.HostVector().clear();
    feature_set_level_.clear();
  }

  HostDeviceVector<int>& GetFeatureSet(int depth) {
    if (this->colsample_bylevel_ == 1.0f) {
      return feature_set_tree_;
    }

    if (feature_set_level_.count(depth) == 0) {
      // Level sampling, level does not yet exist so generate it
      auto& level = feature_set_level_[depth].HostVector();
      level = ColSample(feature_set_tree_.HostVector(), this->colsample_bylevel_);
    }
    // Level sampling
    return feature_set_level_[depth];
  }
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANDOM_H_
