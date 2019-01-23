/*!
 * Copyright 2015 by Contributors
 * \file random.h
 * \brief Utility related to random.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_RANDOM_H_
#define XGBOOST_COMMON_RANDOM_H_

#include <rabit/rabit.h>
#include <xgboost/logging.h>
#include <algorithm>
#include <vector>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>

#include "io.h"

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
 * \brief Handles selection of columns due to colsample_bytree, colsample_bylevel and
 * colsample_bynode parameters. Should be initialised before tree construction and to
 * reset when tree construction is completed.
 */

class ColumnSampler {
  std::shared_ptr<std::vector<int>> feature_set_tree_;
  std::map<int, std::shared_ptr<std::vector<int>>> feature_set_level_;
  float colsample_bylevel_{1.0f};
  float colsample_bytree_{1.0f};
  float colsample_bynode_{1.0f};

  std::shared_ptr<std::vector<int>> ColSample
    (std::shared_ptr<std::vector<int>> p_features, float colsample) const {
    if (colsample == 1.0f) return p_features;
    const auto& features = *p_features;
    CHECK_GT(features.size(), 0);
    int n = std::max(1, static_cast<int>(colsample * features.size()));
    auto p_new_features = std::make_shared<std::vector<int>>();
    auto& new_features = *p_new_features;
    new_features.resize(features.size());
    std::copy(features.begin(), features.end(), new_features.begin());
    std::shuffle(new_features.begin(), new_features.end(), common::GlobalRandom());
    new_features.resize(n);
    std::sort(new_features.begin(), new_features.end());
    // ensure that new_features are the same across ranks
    rabit::Broadcast(&new_features, 0);
    return p_new_features;
  }

 public:
  /**
   * \brief Initialise this object before use.
   *
   * \param num_col
   * \param colsample_bynode
   * \param colsample_bylevel
   * \param colsample_bytree
   * \param skip_index_0      (Optional) True to skip index 0.
   */
  void Init(int64_t num_col, float colsample_bynode, float colsample_bylevel,
            float colsample_bytree, bool skip_index_0 = false) {
    colsample_bylevel_ = colsample_bylevel;
    colsample_bytree_ = colsample_bytree;
    colsample_bynode_ = colsample_bynode;

    if (feature_set_tree_ == nullptr) {
      feature_set_tree_ = std::make_shared<std::vector<int>>();
    }
    Reset();

    int begin_idx = skip_index_0 ? 1 : 0;
    feature_set_tree_->resize(num_col - begin_idx);
    std::iota(feature_set_tree_->begin(), feature_set_tree_->end(), begin_idx);

    feature_set_tree_ = ColSample(feature_set_tree_, colsample_bytree_);
  }

  /**
   * \brief Resets this object.
   */
  void Reset() {
    feature_set_tree_->clear();
    feature_set_level_.clear();
  }

  /**
   * \brief Samples a feature set.
   * 
   * \param depth The tree depth of the node at which to sample.
   * \return The sampled feature set.
   * \note If colsample_bynode_ < 1.0, this method creates a new feature set each time it
   * is called. Therefore, it should be called only once per node.
   */
  std::shared_ptr<std::vector<int>> GetFeatureSet(int depth) {
    if (colsample_bylevel_ == 1.0f && colsample_bynode_ == 1.0f) {
      return feature_set_tree_;
    }

    if (feature_set_level_.count(depth) == 0) {
      // Level sampling, level does not yet exist so generate it
      feature_set_level_[depth] = ColSample(feature_set_tree_, colsample_bylevel_);
    }
    if (colsample_bynode_ == 1.0f) {
      // Level sampling
      return feature_set_level_[depth];
    }
    // Need to sample for the node individually
    return ColSample(feature_set_level_[depth], colsample_bynode_);
  }
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_RANDOM_H_
