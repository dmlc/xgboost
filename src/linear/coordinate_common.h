/**
 * Copyright 2018-2023 by XGBoost Contributors
 * \author Rory Mitchell
 */
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <limits>

#include "xgboost/data.h"
#include "xgboost/parameter.h"
#include "./param.h"
#include "../gbm/gblinear_model.h"
#include "../common/random.h"
#include "../common/threading_utils.h"

namespace xgboost {
namespace linear {

struct CoordinateParam : public XGBoostParameter<CoordinateParam> {
  int top_k;
  DMLC_DECLARE_PARAMETER(CoordinateParam) {
    DMLC_DECLARE_FIELD(top_k)
        .set_lower_bound(0)
        .set_default(0)
        .describe("The number of top features to select in 'thrifty' feature_selector. "
                  "The value of zero means using all the features.");
  }
};

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
inline std::pair<double, double> GetGradient(Context const *ctx, int group_idx, int num_group,
                                             bst_feature_t fidx,
                                             std::vector<GradientPair> const &gpair,
                                             DMatrix *p_fmat) {
  double sum_grad = 0.0, sum_hess = 0.0;
  for (const auto &batch : p_fmat->GetBatches<CSCPage>(ctx)) {
    auto page = batch.GetView();
    auto col = page[fidx];
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
inline std::pair<double, double> GetGradientParallel(Context const *ctx, int group_idx,
                                                     int num_group, int fidx,
                                                     const std::vector<GradientPair> &gpair,
                                                     DMatrix *p_fmat) {
  std::vector<double> sum_grad_tloc(ctx->Threads(), 0.0);
  std::vector<double> sum_hess_tloc(ctx->Threads(), 0.0);

  for (const auto &batch : p_fmat->GetBatches<CSCPage>(ctx)) {
    auto page = batch.GetView();
    auto col = page[fidx];
    const auto ndata = static_cast<bst_omp_uint>(col.size());
    common::ParallelFor(ndata, ctx->Threads(), [&](size_t j) {
      const bst_float v = col[j].fvalue;
      auto &p = gpair[col[j].index * num_group + group_idx];
      if (p.GetHess() < 0.0f) {
        return;
      }
      auto t_idx = omp_get_thread_num();
      sum_grad_tloc[t_idx] += p.GetGrad() * v;
      sum_hess_tloc[t_idx] += p.GetHess() * v * v;
    });
  }
  double sum_grad =
      std::accumulate(sum_grad_tloc.cbegin(), sum_grad_tloc.cend(), 0.0);
  double sum_hess =
      std::accumulate(sum_hess_tloc.cbegin(), sum_hess_tloc.cend(), 0.0);
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
                                                         DMatrix *p_fmat, int32_t n_threads) {
  const auto ndata = static_cast<bst_omp_uint>(p_fmat->Info().num_row_);
  std::vector<double> sum_grad_tloc(n_threads, 0);
  std::vector<double> sum_hess_tloc(n_threads, 0);

  common::ParallelFor(ndata, n_threads, [&](auto i) {
    auto tid = omp_get_thread_num();
    auto &p = gpair[i * num_group + group_idx];
    if (p.GetHess() >= 0.0f) {
      sum_grad_tloc[tid] += p.GetGrad();
      sum_hess_tloc[tid] += p.GetHess();
    }
  });
  double sum_grad = std::accumulate(sum_grad_tloc.cbegin(), sum_grad_tloc.cend(), 0.0);
  double sum_hess = std::accumulate(sum_hess_tloc.cbegin(), sum_hess_tloc.cend(), 0.0);
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
inline void UpdateResidualParallel(Context const *ctx, bst_feature_t fidx, int group_idx,
                                   int num_group, float dw, std::vector<GradientPair> *in_gpair,
                                   DMatrix *p_fmat) {
  if (dw == 0.0f) return;
  for (const auto &batch : p_fmat->GetBatches<CSCPage>(ctx)) {
    auto page = batch.GetView();
    auto col = page[fidx];
    // update grad value
    const auto num_row = static_cast<bst_omp_uint>(col.size());
    common::ParallelFor(num_row, ctx->Threads(), [&](auto j) {
      GradientPair &p = (*in_gpair)[col[j].index * num_group + group_idx];
      if (p.GetHess() < 0.0f) return;
      p += GradientPair(p.GetHess() * col[j].fvalue * dw, 0);
    });
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
inline void UpdateBiasResidualParallel(Context const *ctx, int group_idx, int num_group,
                                       float dbias, std::vector<GradientPair> *in_gpair,
                                       DMatrix *p_fmat) {
  if (dbias == 0.0f) return;
  const auto ndata = static_cast<bst_omp_uint>(p_fmat->Info().num_row_);
  common::ParallelFor(ndata, ctx->Threads(), [&](auto i) {
    GradientPair &g = (*in_gpair)[i * num_group + group_idx];
    if (g.GetHess() < 0.0f) return;
    g += GradientPair(g.GetHess() * dbias, 0);
  });
}

/**
 * \brief Abstract class for stateful feature selection or ordering
 *        in coordinate descent algorithms.
 */
class FeatureSelector {
 public:
  FeatureSelector() = default;
  /*! \brief factory method */
  static FeatureSelector *Create(int choice);
  /*! \brief virtual destructor */
  virtual ~FeatureSelector() = default;
  /**
   * \brief Setting up the selector state prior to looping through features.
   *
   * \param ctx    The booster context.
   * \param model  The model.
   * \param gpair  The gpair.
   * \param p_fmat The feature matrix.
   * \param alpha  Regularisation alpha.
   * \param lambda Regularisation lambda.
   * \param param  A parameter with algorithm-dependent use.
   */
  virtual void Setup(Context const *, const gbm::GBLinearModel &,
                     const std::vector<GradientPair> &, DMatrix *, float, float, int) {}
  /**
   * \brief Select next coordinate to update.
   *
   * \param ctx       Booster context
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
  virtual int NextFeature(Context const *ctx, int iteration, const gbm::GBLinearModel &model,
                          int group_idx, const std::vector<GradientPair> &gpair, DMatrix *p_fmat,
                          float alpha, float lambda) = 0;
};

/**
 * \brief Deterministic selection by cycling through features one at a time.
 */
class CyclicFeatureSelector : public FeatureSelector {
 public:
  using FeatureSelector::FeatureSelector;
  int NextFeature(Context const *, int iteration, const gbm::GBLinearModel &model, int,
                  const std::vector<GradientPair> &, DMatrix *, float, float) override {
    return iteration % model.learner_model_param->num_feature;
  }
};

/**
 * \brief Similar to Cyclic but with random feature shuffling prior to each update.
 * \note Its randomness is controllable by setting a random seed.
 */
class ShuffleFeatureSelector : public FeatureSelector {
 public:
  using FeatureSelector::FeatureSelector;
  void Setup(Context const *, const gbm::GBLinearModel &model, const std::vector<GradientPair> &,
             DMatrix *, float, float, int) override {
    if (feat_index_.size() == 0) {
      feat_index_.resize(model.learner_model_param->num_feature);
      std::iota(feat_index_.begin(), feat_index_.end(), 0);
    }
    std::shuffle(feat_index_.begin(), feat_index_.end(), common::GlobalRandom());
  }

  int NextFeature(Context const *, int iteration, const gbm::GBLinearModel &model, int,
                  const std::vector<GradientPair> &, DMatrix *, float, float) override {
    return feat_index_[iteration % model.learner_model_param->num_feature];
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
  using FeatureSelector::FeatureSelector;
  int NextFeature(Context const *, int, const gbm::GBLinearModel &model, int,
                  const std::vector<GradientPair> &, DMatrix *, float, float) override {
    return common::GlobalRandom()() % model.learner_model_param->num_feature;
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
  using FeatureSelector::FeatureSelector;
  void Setup(Context const *, const gbm::GBLinearModel &model, const std::vector<GradientPair> &,
             DMatrix *, float, float, int param) override {
    top_k_ = static_cast<bst_uint>(param);
    const bst_uint ngroup = model.learner_model_param->num_output_group;
    if (param <= 0) top_k_ = std::numeric_limits<bst_uint>::max();
    if (counter_.size() == 0) {
      counter_.resize(ngroup);
      gpair_sums_.resize(model.learner_model_param->num_feature * ngroup);
    }
    for (bst_uint gid = 0u; gid < ngroup; ++gid) {
      counter_[gid] = 0u;
    }
  }

  int NextFeature(Context const* ctx, int, const gbm::GBLinearModel &model,
                  int group_idx, const std::vector<GradientPair> &gpair,
                  DMatrix *p_fmat, float alpha, float lambda) override {
    // k-th selected feature for a group
    auto k = counter_[group_idx]++;
    // stop after either reaching top-K or going through all the features in a group
    if (k >= top_k_ || counter_[group_idx] == model.learner_model_param->num_feature) return -1;

    const int ngroup = model.learner_model_param->num_output_group;
    const bst_omp_uint nfeat = model.learner_model_param->num_feature;
    // Calculate univariate gradient sums
    std::fill(gpair_sums_.begin(), gpair_sums_.end(), std::make_pair(0., 0.));
    for (const auto &batch : p_fmat->GetBatches<CSCPage>(ctx)) {
      auto page = batch.GetView();
      common::ParallelFor(nfeat, ctx->Threads(), [&](bst_omp_uint i) {
        const auto col = page[i];
        const bst_uint ndata = col.size();
        auto &sums = gpair_sums_[group_idx * nfeat + i];
        for (bst_uint j = 0u; j < ndata; ++j) {
          const bst_float v = col[j].fvalue;
          auto &p = gpair[col[j].index * ngroup + group_idx];
          if (p.GetHess() < 0.f) continue;
          sums.first += p.GetGrad() * v;
          sums.second += p.GetHess() * v * v;
        }
      });
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
  using FeatureSelector::FeatureSelector;

  void Setup(Context const *ctx, const gbm::GBLinearModel &model,
             const std::vector<GradientPair> &gpair, DMatrix *p_fmat, float alpha, float lambda,
             int param) override {
    top_k_ = static_cast<bst_uint>(param);
    if (param <= 0) top_k_ = std::numeric_limits<bst_uint>::max();
    const bst_uint ngroup = model.learner_model_param->num_output_group;
    const bst_omp_uint nfeat = model.learner_model_param->num_feature;

    if (deltaw_.size() == 0) {
      deltaw_.resize(nfeat * ngroup);
      sorted_idx_.resize(nfeat * ngroup);
      counter_.resize(ngroup);
      gpair_sums_.resize(nfeat * ngroup);
    }
    // Calculate univariate gradient sums
    std::fill(gpair_sums_.begin(), gpair_sums_.end(), std::make_pair(0., 0.));
    for (const auto &batch : p_fmat->GetBatches<CSCPage>(ctx)) {
      auto page = batch.GetView();
      // column-parallel is usually fastaer than row-parallel
      common::ParallelFor(nfeat, ctx->Threads(), [&](auto i) {
        const auto col = page[i];
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
      });
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

  int NextFeature(Context const *, int, const gbm::GBLinearModel &model, int group_idx,
                  const std::vector<GradientPair> &, DMatrix *, float, float) override {
    // k-th selected feature for a group
    auto k = counter_[group_idx]++;
    // stop after either reaching top-N or going through all the features in a group
    if (k >= top_k_ || counter_[group_idx] == model.learner_model_param->num_feature) return -1;
    // note that sorted_idx stores the "long" indices
    const size_t grp_offset = group_idx * model.learner_model_param->num_feature;
    return static_cast<int>(sorted_idx_[grp_offset + k] - grp_offset);
  }

 protected:
  bst_uint top_k_;
  std::vector<bst_float> deltaw_;
  std::vector<size_t> sorted_idx_;
  std::vector<bst_uint> counter_;
  std::vector<std::pair<double, double>> gpair_sums_;
};

inline FeatureSelector *FeatureSelector::Create(int choice) {
  switch (choice) {
    case kCyclic:
      return new CyclicFeatureSelector;
    case kShuffle:
      return new ShuffleFeatureSelector;
    case kThrifty:
      return new ThriftyFeatureSelector;
    case kGreedy:
      return new GreedyFeatureSelector;
    case kRandom:
      return new RandomFeatureSelector;
    default:
      LOG(FATAL) << "unknown coordinate selector: " << choice;
  }
  return nullptr;
}

}  // namespace linear
}  // namespace xgboost
