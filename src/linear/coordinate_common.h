/*!
 * Copyright 2018 by Contributors
 * \author Rory Mitchell
 */
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
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
 * \param reg_lambda          Unnormalised L2 penalty.
 * \param reg_alpha           Unnormalised L1 penalty.
 * \param sum_instance_weight The sum instance weights, used to normalise l1/l2 penalty.
 *
 * \return  The weight update.
 */

inline double CoordinateDelta(double sum_grad, double sum_hess, double w,
                              double reg_lambda, double reg_alpha,
                              double sum_instance_weight) {
  reg_alpha *= sum_instance_weight;
  reg_lambda *= sum_instance_weight;
  if (sum_hess < 1e-5f) return 0.0f;
  double tmp = w - (sum_grad + reg_lambda * w) / (sum_hess + reg_lambda);
  if (tmp >= 0) {
    return std::max(
        -(sum_grad + reg_lambda * w + reg_alpha) / (sum_hess + reg_lambda), -w);
  } else {
    return std::min(
        -(sum_grad + reg_lambda * w - reg_alpha) / (sum_hess + reg_lambda), -w);
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

inline std::pair<double, double> GetGradient(
    int group_idx, int num_group, int fidx, const std::vector<bst_gpair> &gpair,
    DMatrix *p_fmat) {
  double sum_grad = 0.0, sum_hess = 0.0;
  dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
  while (iter->Next()) {
    const ColBatch &batch = iter->Value();
    ColBatch::Inst col = batch[fidx];
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
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
 * \brief Get the gradient with respect to a single feature. Multithreaded.
 *
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param fidx      The target feature.
 * \param gpair     Gradients.
 * \param p_fmat    The feature matrix.
 *
 * \return  The gradient and diagonal Hessian entry for a given feature.
 */

inline std::pair<double, double> GetGradientParallel(
    int group_idx, int num_group, int fidx,

    const std::vector<bst_gpair> &gpair, DMatrix *p_fmat) {
  double sum_grad = 0.0, sum_hess = 0.0;
  dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
  while (iter->Next()) {
    const ColBatch &batch = iter->Value();
    ColBatch::Inst col = batch[fidx];
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
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
 * \brief Get the gradient with respect to the bias. Multithreaded.
 *
 * \param group_idx Zero-based index of the group.
 * \param num_group Number of groups.
 * \param gpair     Gradients.
 * \param p_fmat    The feature matrix.
 *
 * \return  The gradient and diagonal Hessian entry for the bias.
 */

inline std::pair<double, double> GetBiasGradientParallel(
    int group_idx, int num_group, const std::vector<bst_gpair> &gpair,
    DMatrix *p_fmat) {
  const RowSet &rowset = p_fmat->buffered_rowset();
  double sum_grad = 0.0, sum_hess = 0.0;
  const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
#pragma omp parallel for schedule(static) reduction(+ : sum_grad, sum_hess)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    auto &p = gpair[rowset[i] * num_group + group_idx];
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
                                   float dw, std::vector<bst_gpair> *in_gpair,
                                   DMatrix *p_fmat) {
  if (dw == 0.0f) return;
  dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
  while (iter->Next()) {
    const ColBatch &batch = iter->Value();
    ColBatch::Inst col = batch[fidx];
    // update grad value
    const bst_omp_uint num_row = static_cast<bst_omp_uint>(col.length);
#pragma omp parallel for schedule(static)
    for (bst_omp_uint j = 0; j < num_row; ++j) {
      bst_gpair &p = (*in_gpair)[col[j].index * num_group + group_idx];
      if (p.GetHess() < 0.0f) continue;
      p += bst_gpair(p.GetHess() * col[j].fvalue * dw, 0);
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

inline void UpdateBiasResidualParallel(int group_idx, int num_group,
                                       float dbias,
                                       std::vector<bst_gpair> *in_gpair,
                                       DMatrix *p_fmat) {
  if (dbias == 0.0f) return;
  const RowSet &rowset = p_fmat->buffered_rowset();
  const bst_omp_uint ndata = static_cast<bst_omp_uint>(p_fmat->info().num_row);
#pragma omp parallel for schedule(static)
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    bst_gpair &g = (*in_gpair)[rowset[i] * num_group + group_idx];
    if (g.GetHess() < 0.0f) continue;
    g += bst_gpair(g.GetHess() * dbias, 0);
  }
}

/**
 * \class CoordinateSelector
 *
 * \brief Abstract class for stateful feature selection in coordinate descent
 * algorithms.
 */

class CoordinateSelector {
 public:
  static CoordinateSelector *Create(std::string name);
  /*! \brief virtual destructor */
  virtual ~CoordinateSelector() {}

  /**
   * \brief Select next coordinate to update.
   *
   * \param iteration           The iteration.
   * \param model               The model.
   * \param group_idx           Zero-based index of the group.
   * \param gpair               The gpair.
   * \param p_fmat              The feature matrix.
   * \param alpha               Regularisation alpha.
   * \param lambda              Regularisation lambda.
   * \param sum_instance_weight The sum instance weight.
   *
   * \return  The index of the selected feature. -1 indicates the bias term.
   */

  virtual int SelectNextCoordinate(int iteration,
                                   const gbm::GBLinearModel &model,
                                   int group_idx,
                                   const std::vector<bst_gpair> &gpair,
                                   DMatrix *p_fmat, float alpha, float lambda,
                                   double sum_instance_weight) = 0;
};

/**
 * \class CyclicCoordinateSelector
 *
 * \brief Deterministic selection by cycling through coordinates one at a time.
 */

class CyclicCoordinateSelector : public CoordinateSelector {
 public:
  int SelectNextCoordinate(int iteration, const gbm::GBLinearModel &model,
                           int group_idx, const std::vector<bst_gpair> &gpair,
                           DMatrix *p_fmat, float alpha, float lambda,
                           double sum_instance_weight) override {
    return iteration % model.param.num_feature;
  }
};

/**
 * \class RandomCoordinateSelector
 *
 * \brief A random coordinate selector.
 */

class RandomCoordinateSelector : public CoordinateSelector {
 public:
  int SelectNextCoordinate(int iteration, const gbm::GBLinearModel &model,
                           int group_idx, const std::vector<bst_gpair> &gpair,
                           DMatrix *p_fmat, float alpha, float lambda,
                           double sum_instance_weight) override {
    return common::GlobalRandom()() % model.param.num_feature;
  }
};

/**
 * \class GreedyCoordinateSelector
 *
 * \brief Select coordinate with the greatest gradient magnitude.
 */

class GreedyCoordinateSelector : public CoordinateSelector {
 public:
  int SelectNextCoordinate(int iteration, const gbm::GBLinearModel &model,
                           int group_idx, const std::vector<bst_gpair> &gpair,
                           DMatrix *p_fmat, float alpha, float lambda,
                           double sum_instance_weight) override {
    // Find best
    int best_fidx = 0;
    double best_weight_update = 0.0f;

    for (auto fidx = 0; fidx < model.param.num_feature; fidx++) {
      const float w = model[fidx][group_idx];
      auto gradient = GetGradientParallel(
          group_idx, model.param.num_output_group, fidx, gpair, p_fmat);
      float dw = static_cast<float>(
          CoordinateDelta(gradient.first, gradient.second, w, lambda, alpha,
                          sum_instance_weight));
      if (std::abs(dw) > std::abs(best_weight_update)) {
        best_weight_update = dw;
        best_fidx = fidx;
      }
    }
    return best_fidx;
  }
};

inline CoordinateSelector *CoordinateSelector::Create(std::string name) {
  if (name == "cyclic") {
    return new CyclicCoordinateSelector();
  } else if (name == "random") {
    return new RandomCoordinateSelector();
  } else if (name == "greedy") {
    return new GreedyCoordinateSelector();
  } else {
    LOG(FATAL) << name << ": unknown coordinate selector";
  }
  return nullptr;
}

}  // namespace linear
}  // namespace xgboost
