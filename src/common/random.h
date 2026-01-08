/**
 * Copyright 2015-2026, XGBoost Contributors
 * \file random.h
 * \brief Utility related to random.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_RANDOM_H_
#define XGBOOST_COMMON_RANDOM_H_

#include <xgboost/logging.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "../collective/broadcast.h"  // for Broadcast
#include "algorithm.h"  // ArgSort
#include "xgboost/context.h"  // Context
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost::common {
/*!
 * \brief Define mt19937 as default type Random Engine.
 */
using RandomEngine = std::mt19937;

/*!
 * \brief global random engine
 */
using GlobalRandomEngine = RandomEngine;

/*!
 * \brief global singleton of a random engine.
 *  This random engine is thread-local and
 *  only visible to current thread.
 */
GlobalRandomEngine& GlobalRandom(); // NOLINT(*)

/*
 * Original paper:
 * Weighted Random Sampling (2005; Efraimidis, Spirakis)
 *
 * Blog:
 * https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
*/
template <typename T>
std::vector<T> WeightedSamplingWithoutReplacement(Context const* ctx, std::vector<T> const& array,
                                                  std::vector<float> const& weights, size_t n) {
  // ES sampling.
  CHECK_EQ(array.size(), weights.size());
  std::vector<float> keys(weights.size());
  std::uniform_real_distribution<float> dist;
  auto& rng = GlobalRandom();
  for (size_t i = 0; i < array.size(); ++i) {
    auto w = std::max(weights.at(i), kRtEps);
    auto u = dist(rng);
    auto k = std::log(u) / w;
    keys[i] = k;
  }
  auto ind = ArgSort<std::size_t>(ctx, keys.data(), keys.data() + keys.size(), std::greater<>{});
  ind.resize(n);

  std::vector<T> results(ind.size());
  for (size_t k = 0; k < ind.size(); ++k) {
    auto idx = ind[k];
    results[k] = array[idx];
  }
  return results;
}

namespace cuda_impl {
void SampleFeature(Context const* ctx, bst_feature_t n_features,
                   std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features,
                   std::shared_ptr<HostDeviceVector<bst_feature_t>> p_new_features,
                   HostDeviceVector<float> const& feature_weights,
                   HostDeviceVector<float>* weight_buffer,
                   HostDeviceVector<bst_feature_t>* idx_buffer, GlobalRandomEngine* grng);

void InitFeatureSet(Context const* ctx,
                    std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features);
}  // namespace cuda_impl

/**
 * \class ColumnSampler
 *
 * \brief Handles selection of columns due to colsample_bytree, colsample_bylevel and
 * colsample_bynode parameters. Should be initialised before tree construction and to
 * reset when tree construction is completed.
 */
class ColumnSampler {
  std::shared_ptr<HostDeviceVector<bst_feature_t>> feature_set_tree_;
  std::map<int, std::shared_ptr<HostDeviceVector<bst_feature_t>>> feature_set_level_;
  HostDeviceVector<float> feature_weights_;
  float colsample_bylevel_{1.0f};
  float colsample_bytree_{1.0f};
  float colsample_bynode_{1.0f};
  GlobalRandomEngine rng_;
  Context const* ctx_;

  // Used for weighted sampling.
  HostDeviceVector<bst_feature_t> idx_buffer_;
  HostDeviceVector<float> weight_buffer_;

 public:
  std::shared_ptr<HostDeviceVector<bst_feature_t>> ColSample(
      std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features, float colsample);
  /**
   * @brief Column sampler constructor.
   * @note This constructor manually sets the rng seed
   */
  explicit ColumnSampler(std::uint32_t seed) { rng_.seed(seed); }

  /**
   * @brief Initialise this object before use.
   *
   * @param num_col
   * @param colsample_bynode  Sampling rate for node.
   * @param colsample_bylevel Sampling rate for tree level.
   * @param colsample_bytree  Sampling rate for tree.
   */
  void Init(Context const* ctx, int64_t num_col, HostDeviceVector<float> const& feature_weights,
            float colsample_bynode, float colsample_bylevel, float colsample_bytree) {
    this->feature_weights_.SetDevice(ctx->Device()), feature_weights.SetDevice(ctx->Device());
    this->feature_weights_.Resize(feature_weights.Size());
    this->feature_weights_.Copy(feature_weights);

    colsample_bylevel_ = colsample_bylevel;
    colsample_bytree_ = colsample_bytree;
    colsample_bynode_ = colsample_bynode;
    ctx_ = ctx;

    if (feature_set_tree_ == nullptr) {
      feature_set_tree_ = std::make_shared<HostDeviceVector<bst_feature_t>>();
    }
    Reset();

    // We process ColumnSampler on host for SYCL. So don't need to push data to device
    if (!ctx->Device().IsSycl()) {
      feature_set_tree_->SetDevice(ctx->Device());
    }
    feature_set_tree_->Resize(num_col);
    if (ctx->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
      cuda_impl::InitFeatureSet(ctx, feature_set_tree_);
#else
      AssertGPUSupport();
#endif
    } else {
      std::iota(feature_set_tree_->HostVector().begin(), feature_set_tree_->HostVector().end(), 0);
    }

    feature_set_tree_ = ColSample(feature_set_tree_, colsample_bytree_);
  }

  /**
   * \brief Resets this object.
   */
  void Reset() {
    feature_set_tree_->Resize(0);
    feature_set_level_.clear();
  }

  /**
   * \brief Samples a feature set.
   *
   * \param depth The tree depth of the node at which to sample.
   * \return The sampled feature set.
   * \note If colsample_bynode_ < 1.0, this method creates a new feature set each time it
   * is called. Therefore, it should be called only once per node.
   * \note With distributed xgboost, this function must be called exactly once for the
   * construction of each tree node, and must be called the same number of times in each
   * process and with the same parameters to return the same feature set across processes.
   */
  std::shared_ptr<HostDeviceVector<bst_feature_t>> GetFeatureSet(int depth) {
    if (colsample_bylevel_ == 1.0f && colsample_bynode_ == 1.0f) {
      return feature_set_tree_;
    }

    if (feature_set_level_.count(depth) == 0) {
      // Level sampling, level does not yet exist so generate it
      feature_set_level_[depth] = ColSample(feature_set_tree_, colsample_bylevel_);
    }
    if (colsample_bynode_ == 1.0f) {
      // Level sampling
      auto ptr = feature_set_level_[depth];
      ptr->SetDevice(this->ctx_->Device());
      return ptr;
    }
    // Need to sample for the node individually
    auto ptr = ColSample(feature_set_level_[depth], colsample_bynode_);
    ptr->SetDevice(this->ctx_->Device());
    return ptr;
  }
};

inline auto MakeColumnSampler(Context const* ctx) {
  std::uint32_t seed = common::GlobalRandom()();
  auto rc = collective::Broadcast(ctx, linalg::MakeVec(&seed, 1), 0);
  collective::SafeColl(rc);
  auto cs = std::make_shared<common::ColumnSampler>(seed);
  return cs;
}
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_RANDOM_H_
