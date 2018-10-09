/*!
 * Copyright 2018 by Contributors
 * \author Rory Mitchell
 */
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include "../common/random.h"

namespace xgboost {
namespace linear {

/**
 * \brief Calculate change in weight for a given feature. Applies l1/l2 penalty normalised by the
 *        number of training instances.
 *
 * \param sum_grad            The sum gradient.
 * \param sum_hess            The sum hess.
 * \param w                   The weight.
 * \param reg_alpha           Unnormalised L1 penalty.
 * \param reg_lambda          Unnormalised L2 penalty.
 *
 * \return  The weight update.
 */
inline double CoordinateDelta(double sum_grad, double sum_hess, double w,
                              double reg_alpha, double reg_lambda) {
  if (sum_hess < 1e-5f) return 0.0f;
  const double sum_grad_l2 = sum_grad + reg_lambda * w;
  const double sum_hess_l2 = sum_hess + reg_lambda;
  const double tmp = w - sum_grad_l2 / sum_hess_l2;
  if (tmp >= 0) {
    return std::max(-(sum_grad_l2 + reg_alpha) / sum_hess_l2, -w);
  } else {
    return std::min(-(sum_grad_l2 - reg_alpha) / sum_hess_l2, -w);
  }
}

/**
 * \brief Calculate update to bias.
 *
 * \param sum_grad  The sum gradient.
 * \param sum_hess  The sum hess.
 *
 * \return  The weight update.
 */
inline double CoordinateDeltaBias(double sum_grad, double sum_hess) {
  return -sum_grad / sum_hess;
}

/**
 * \brief Get the gradient with respect to a single feature.
 *
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param fidx      The target feature.
 * \param gpair     Gradients.
 * \param p_fmat    The feature matrix.
 *
 * \return  The gradient and diagonal Hessian entry for a given feature.
 */
inline std::pair<double, double> GetGradient(int group_idx, int num_group, int fidx,
                                             const std::vector<GradientPair> &gpair,
                                             DMatrix *p_fmat) {
  double sum_grad = 0.0, sum_hess = 0.0;
  for (const auto &batch : p_fmat->GetColumnBatches()) {
    auto col = batch[fidx];
    const auto ndata = static_cast<bst_omp_uint>(col.size());
    for (bst_omp_uint j = 0; j < ndata; ++j) {
      const bst_float v = col[j].fvalue;
      auto &p = gpair[col[j].index * num_group + group_idx];
      if (p.GetHess() < 0.0f) continue;
      sum_grad += p.GetGrad() * v;
      sum_hess += p.GetHess() * v * v;
    }
  }
  return std::make_pair(sum_grad, sum_hess);
}

/**
 * \brief Get the gradient with respect to a single feature. Row-wise multithreaded.
 *
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param fidx      The target feature.
 * \param gpair     Gradients.
 * \param p_fmat    The feature matrix.
 *
 * \return  The gradient and diagonal Hessian entry for a given feature.
 */
inline std::pair<double, double> GetGradientParallel(int group_idx, int num_group, int fidx,
                                                     const std::vector<GradientPair> &gpair,
                                                     DMatrix *p_fmat) {
  double sum_grad = 0.0, sum_hess = 0.0;
  for (const auto &batch : p_fmat->GetColumnBatches()) {
    auto col = batch[fidx];
    const auto ndata = static_cast<bst_omp_uint>(col.size());
#pragma omp parallel for schedule(static) reduction(+ : sum_grad, sum_hess)
    for (bst_omp_uint j = 0; j < ndata; ++j) {
      const bst_float v = col[j].fvalue;
      auto &p = gpair[col[j].index * num_group + group_idx];
      if (p.GetHess() < 0.0f) continue;
      sum_grad += p.GetGrad() * v;
      sum_hess += p.GetHess() * v * v;
    }
  }
  return std::make_pair(sum_grad, sum_hess);
}

/**
 * \brief Get the gradient with respect to the bias. Row-wise multithreaded.
 *
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param gpair     Gradients.
 * \param p_fmat    The feature matrix.
 *
 * \return  The gradient and diagonal Hessian entry for the bias.
 */
inline std::pair<double, double> GetBiasGradientParallel(int group_idx, int num_group,
                                                         const std::vector<GradientPair> &gpair,
                                                         DMatrix *p_fmat) {
  double sum_grad = 0.0, sum_hess = 0.0;
  const auto ndata = static_cast<bst_omp_uint>(p_fmat->Info().num_row_);
#pragma omp parallel for schedule(static) reduction(+ : sum_grad, sum_hess)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    auto &p = gpair[i * num_group + group_idx];
    if (p.GetHess() >= 0.0f) {
      sum_grad += p.GetGrad();
      sum_hess += p.GetHess();
    }
  }
  return std::make_pair(sum_grad, sum_hess);
}

/**
 * \brief Updates the gradient vector with respect to a change in weight.
 *
 * \param fidx      The feature index.
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param dw        The change in weight.
 * \param in_gpair  The gradient vector to be updated.
 * \param p_fmat    The input feature matrix.
 */
inline void UpdateResidualParallel(int fidx, int group_idx, int num_group,
                                   float dw, std::vector<GradientPair> *in_gpair,
                                   DMatrix *p_fmat) {
  if (dw == 0.0f) return;
  for (const auto &batch : p_fmat->GetColumnBatches()) {
    auto col = batch[fidx];
    // update grad value
    const auto num_row = static_cast<bst_omp_uint>(col.size());
#pragma omp parallel for schedule(static)
    for (bst_omp_uint j = 0; j < num_row; ++j) {
      GradientPair &p = (*in_gpair)[col[j].index * num_group + group_idx];
      if (p.GetHess() < 0.0f) continue;
      p += GradientPair(p.GetHess() * col[j].fvalue * dw, 0);
    }
  }
}

/**
 * \brief Updates the gradient vector based on a change in the bias.
 *
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param dbias     The change in bias.
 * \param in_gpair  The gradient vector to be updated.
 * \param p_fmat    The input feature matrix.
 */
inline void UpdateBiasResidualParallel(int group_idx, int num_group, float dbias,
                                       std::vector<GradientPair> *in_gpair,
                                       DMatrix *p_fmat) {
  if (dbias == 0.0f) return;
  const auto ndata = static_cast<bst_omp_uint>(p_fmat->Info().num_row_);
#pragma omp parallel for schedule(static)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    GradientPair &g = (*in_gpair)[i * num_group + group_idx];
    if (g.GetHess() < 0.0f) continue;
    g += GradientPair(g.GetHess() * dbias, 0);
  }
}

/**
 * \brief Abstract class for stateful feature selection or ordering
 *        in coordinate descent algorithms.
 */
class FeatureSelector {
 public:
  /*! \brief factory method */
  static FeatureSelector *Create(int choice);
  /*! \brief virtual destructor */
  virtual ~FeatureSelector() = default;
  /**
   * \brief Setting up the selector state prior to looping through features.
   *
   * \param model  The model.
   * \param gpair  The gpair.
   * \param p_fmat The feature matrix.
   * \param alpha  Regularisation alpha.
   * \param lambda Regularisation lambda.
   * \param param  A parameter with algorithm-dependent use.
   */
  virtual void Setup(const gbm::GBLinearModel &model,
                     const std::vector<GradientPair> &gpair,
                     DMatrix *p_fmat,
                     float alpha, float lambda, int param) {}
  /**
   * \brief Select next coordinate to update.
   *
   * \param iteration The iteration in a loop through features
   * \param model     The model.
   * \param group_idx Zero-based index of the group.
   * \param gpair     The gpair.
   * \param p_fmat    The feature matrix.
   * \param alpha     Regularisation alpha.
   * \param lambda    Regularisation lambda.
   *
   * \return  The index of the selected feature. -1 indicates none selected.
   */
  virtual int NextFeature(int iteration,
                          const gbm::GBLinearModel &model,
                          int group_idx,
                          const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat, float alpha, float lambda) = 0;
};

/**
 * \brief Deterministic selection by cycling through features one at a time.
 */
class CyclicFeatureSelector : public FeatureSelector {
 public:
  int NextFeature(int iteration, const gbm::GBLinearModel &model,
                  int group_idx, const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat, float alpha, float lambda) override {
    return iteration % model.param.num_feature;
  }
};

/**
 * \brief Similar to Cyclic but with random feature shuffling prior to each update.
 * \note Its randomness is controllable by setting a random seed.
 */
class ShuffleFeatureSelector : public FeatureSelector {
 public:
  void Setup(const gbm::GBLinearModel &model,
             const std::vector<GradientPair> &gpair,
             DMatrix *p_fmat, float alpha, float lambda, int param) override {
    if (feat_index_.size() == 0) {
      feat_index_.resize(model.param.num_feature);
      std::iota(feat_index_.begin(), feat_index_.end(), 0);
    }
    std::shuffle(feat_index_.begin(), feat_index_.end(), common::GlobalRandom());
  }

  int NextFeature(int iteration, const gbm::GBLinearModel &model,
                  int group_idx, const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat, float alpha, float lambda) override {
    return feat_index_[iteration % model.param.num_feature];
  }

 protected:
  std::vector<bst_uint> feat_index_;
};

/**
 * \brief A random (with replacement) coordinate selector.
 * \note Its randomness is controllable by setting a random seed.
 */
class RandomFeatureSelector : public FeatureSelector {
 public:
  int NextFeature(int iteration, const gbm::GBLinearModel &model,
                  int group_idx, const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat, float alpha, float lambda) override {
    return common::GlobalRandom()() % model.param.num_feature;
  }
};

/**
 * \brief Select coordinate with the greatest gradient magnitude.
 * \note It has O(num_feature^2) complexity. It is fully deterministic.
 *
 * \note It allows restricting the selection to top_k features per group with
 * the largest magnitude of univariate weight change, by passing the top_k value
 * through the `param` argument of Setup(). That would reduce the complexity to
 * O(num_feature*top_k).
 */
class GreedyFeatureSelector : public FeatureSelector {
 public:
  void Setup(const gbm::GBLinearModel &model,
             const std::vector<GradientPair> &gpair,
             DMatrix *p_fmat, float alpha, float lambda, int param) override {
    top_k_ = static_cast<bst_uint>(param);
    const bst_uint ngroup = model.param.num_output_group;
    if (param <= 0) top_k_ = std::numeric_limits<bst_uint>::max();
    if (counter_.size() == 0) {
      counter_.resize(ngroup);
      gpair_sums_.resize(model.param.num_feature * ngroup);
    }
    for (bst_uint gid = 0u; gid < ngroup; ++gid) {
      counter_[gid] = 0u;
    }
  }

  int NextFeature(int iteration, const gbm::GBLinearModel &model,
                  int group_idx, const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat, float alpha, float lambda) override {
    // k-th selected feature for a group
    auto k = counter_[group_idx]++;
    // stop after either reaching top-K or going through all the features in a group
    if (k >= top_k_ || counter_[group_idx] == model.param.num_feature) return -1;

    const int ngroup = model.param.num_output_group;
    const bst_omp_uint nfeat = model.param.num_feature;
    // Calculate univariate gradient sums
    std::fill(gpair_sums_.begin(), gpair_sums_.end(), std::make_pair(0., 0.));
  for (const auto &batch : p_fmat->GetColumnBatches()) {
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        const auto col = batch[i];
        const bst_uint ndata = col.size();
        auto &sums = gpair_sums_[group_idx * nfeat + i];
        for (bst_uint j = 0u; j < ndata; ++j) {
          const bst_float v = col[j].fvalue;
          auto &p = gpair[col[j].index * ngroup + group_idx];
          if (p.GetHess() < 0.f) continue;
          sums.first += p.GetGrad() * v;
          sums.second += p.GetHess() * v * v;
        }
      }
    }
    // Find a feature with the largest magnitude of weight change
    int best_fidx = 0;
    double best_weight_update = 0.0f;
    for (bst_omp_uint fidx = 0; fidx < nfeat; ++fidx) {
      auto &s = gpair_sums_[group_idx * nfeat + fidx];
      float dw = std::abs(static_cast<bst_float>(
                 CoordinateDelta(s.first, s.second, model[fidx][group_idx], alpha, lambda)));
      if (dw > best_weight_update) {
        best_weight_update = dw;
        best_fidx = fidx;
      }
    }
    return best_fidx;
  }

 protected:
  bst_uint top_k_;
  std::vector<bst_uint> counter_;
  std::vector<std::pair<double, double>> gpair_sums_;
};

/**
 * \brief Thrifty, approximately-greedy feature selector.
 *
 * \note Prior to cyclic updates, reorders features in descending magnitude of
 * their univariate weight changes. This operation is multithreaded and is a
 * linear complexity approximation of the quadratic greedy selection.
 *
 * \note It allows restricting the selection to top_k features per group with
 * the largest magnitude of univariate weight change, by passing the top_k value
 * through the `param` argument of Setup().
 */
class ThriftyFeatureSelector : public FeatureSelector {
 public:
  void Setup(const gbm::GBLinearModel &model,
             const std::vector<GradientPair> &gpair,
             DMatrix *p_fmat, float alpha, float lambda, int param) override {
    top_k_ = static_cast<bst_uint>(param);
    if (param <= 0) top_k_ = std::numeric_limits<bst_uint>::max();
    const bst_uint ngroup = model.param.num_output_group;
    const bst_omp_uint nfeat = model.param.num_feature;

    if (deltaw_.size() == 0) {
      deltaw_.resize(nfeat * ngroup);
      sorted_idx_.resize(nfeat * ngroup);
      counter_.resize(ngroup);
      gpair_sums_.resize(nfeat * ngroup);
    }
    // Calculate univariate gradient sums
    std::fill(gpair_sums_.begin(), gpair_sums_.end(), std::make_pair(0., 0.));
    for (const auto &batch : p_fmat->GetColumnBatches()) {
// column-parallel is usually faster than row-parallel
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        const auto col = batch[i];
        const bst_uint ndata = col.size();
        for (bst_uint gid = 0u; gid < ngroup; ++gid) {
          auto &sums = gpair_sums_[gid * nfeat + i];
          for (bst_uint j = 0u; j < ndata; ++j) {
            const bst_float v = col[j].fvalue;
            auto &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.f) continue;
            sums.first += p.GetGrad() * v;
            sums.second += p.GetHess() * v * v;
          }
        }
      }
    }
    // rank by descending weight magnitude within the groups
    std::fill(deltaw_.begin(), deltaw_.end(), 0.f);
    std::iota(sorted_idx_.begin(), sorted_idx_.end(), 0);
    bst_float *pdeltaw = &deltaw_[0];
    for (bst_uint gid = 0u; gid < ngroup; ++gid) {
      // Calculate univariate weight changes
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        auto ii = gid * nfeat + i;
        auto &s = gpair_sums_[ii];
        deltaw_[ii] = static_cast<bst_float>(CoordinateDelta(
                       s.first, s.second, model[i][gid], alpha, lambda));
      }
      // sort in descending order of deltaw abs values
      auto start = sorted_idx_.begin() + gid * nfeat;
      std::sort(start, start + nfeat,
                [pdeltaw](size_t i, size_t j) {
                  return std::abs(*(pdeltaw + i)) > std::abs(*(pdeltaw + j));
                });
      counter_[gid] = 0u;
    }
  }

  int NextFeature(int iteration, const gbm::GBLinearModel &model,
                  int group_idx, const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat, float alpha, float lambda) override {
    // k-th selected feature for a group
    auto k = counter_[group_idx]++;
    // stop after either reaching top-N or going through all the features in a group
    if (k >= top_k_ || counter_[group_idx] == model.param.num_feature) return -1;
    // note that sorted_idx stores the "long" indices
    const size_t grp_offset = group_idx * model.param.num_feature;
    return static_cast<int>(sorted_idx_[grp_offset + k] - grp_offset);
  }

 protected:
  bst_uint top_k_;
  std::vector<bst_float> deltaw_;
  std::vector<size_t> sorted_idx_;
  std::vector<bst_uint> counter_;
  std::vector<std::pair<double, double>> gpair_sums_;
};

/**
 * \brief A set of available FeatureSelector's
 */
enum FeatureSelectorEnum {
  kCyclic = 0,
  kShuffle,
  kThrifty,
  kGreedy,
  kRandom
};

inline FeatureSelector *FeatureSelector::Create(int choice) {
  switch (choice) {
    case kCyclic:
      return new CyclicFeatureSelector();
    case kShuffle:
      return new ShuffleFeatureSelector();
    case kThrifty:
      return new ThriftyFeatureSelector();
    case kGreedy:
      return new GreedyFeatureSelector();
    case kRandom:
      return new RandomFeatureSelector();
    default:
      LOG(FATAL) << "unknown coordinate selector: " << choice;
  }
  return nullptr;
}

}  // namespace linear
}  // namespace xgboost
