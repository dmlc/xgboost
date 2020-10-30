/*!
 * Copyright 2018-2020 by Contributors
 * \file split_evaluator.h
 * \brief Used for implementing a loss term specific to decision trees. Useful for custom regularisation.
 * \author Henry Gouk
 */

#ifndef XGBOOST_TREE_SPLIT_EVALUATOR_H_
#define XGBOOST_TREE_SPLIT_EVALUATOR_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <utility>
#include <vector>
#include <limits>

#include "xgboost/tree_model.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/generic_parameters.h"
#include "../common/transform.h"
#include "../common/math.h"
#include "param.h"

namespace xgboost {
namespace tree {
class TreeEvaluator {
  // hist and exact use parent id to calculate constraints.
  static constexpr bst_node_t kRootParentId =
      (-1 & static_cast<bst_node_t>((1U << 31) - 1));

  HostDeviceVector<float> lower_bounds_;
  HostDeviceVector<float> upper_bounds_;
  HostDeviceVector<int32_t> monotone_;
  int32_t device_;
  bool has_constraint_;

 public:
  TreeEvaluator(TrainParam const& p, bst_feature_t n_features, int32_t device) {
    device_ = device;
    if (device != GenericParameter::kCpuId) {
      lower_bounds_.SetDevice(device);
      upper_bounds_.SetDevice(device);
      monotone_.SetDevice(device);
    }

    if (p.monotone_constraints.empty()) {
      monotone_.HostVector().resize(n_features, 0);
      has_constraint_ = false;
    } else {
      monotone_.HostVector() = p.monotone_constraints;
      monotone_.HostVector().resize(n_features, 0);
      lower_bounds_.Resize(p.MaxNodes(), -std::numeric_limits<float>::max());
      upper_bounds_.Resize(p.MaxNodes(), std::numeric_limits<float>::max());
      has_constraint_ = true;
    }

    if (device_ != GenericParameter::kCpuId) {
      // Pull to device early.
      lower_bounds_.ConstDeviceSpan();
      upper_bounds_.ConstDeviceSpan();
      monotone_.ConstDeviceSpan();
    }
  }

  template <typename ParamT>
  struct SplitEvaluator {
    common::Span<int const> constraints;
    common::Span<float const> lower;
    common::Span<float const> upper;
    bool has_constraint;

    XGBOOST_DEVICE double CalcSplitGain(const ParamT &param, bst_node_t nidx,
                                        bst_feature_t fidx,
                                        tree::GradStats left,
                                        tree::GradStats right) const {
      int constraint = constraints[fidx];
      const double negative_infinity = -std::numeric_limits<double>::infinity();
      double wleft = this->CalcWeight(nidx, param, left);
      double wright = this->CalcWeight(nidx, param, right);

      double gain = this->CalcGainGivenWeight(nidx, param, left, wleft) +
                    this->CalcGainGivenWeight(nidx, param, right, wright);

      if (constraint == 0) {
        return gain;
      } else if (constraint > 0) {
        return wleft <= wright ? gain : negative_infinity;
      } else {
        return wleft >= wright ? gain : negative_infinity;
      }
    }

    XGBOOST_DEVICE float CalcWeight(bst_node_t nodeid, const ParamT &param,
                                    tree::GradStats stats) const {
      float w = xgboost::tree::CalcWeight(param, stats);
      if (!has_constraint) {
        return w;
      }

      if (nodeid == kRootParentId) {
        return w;
      } else if (w < lower(nodeid)) {
        return lower[nodeid];
      } else if (w > upper(nodeid)) {
        return upper[nodeid];
      } else {
        return w;
      }
    }
    XGBOOST_DEVICE float CalcGainGivenWeight(bst_node_t, ParamT const &p,
                                             tree::GradStats stats, float w) const {
      if (stats.GetHess() <= 0) {
        return .0f;
      }
      // Avoiding tree::CalcGainGivenWeight can significantly reduce avg floating point error.
      if (p.max_delta_step == 0.0f && has_constraint == false) {
        return Sqr(ThresholdL1(stats.sum_grad, p.reg_alpha)) /
               (stats.sum_hess + p.reg_lambda);
      }
      return tree::CalcGainGivenWeight<ParamT, float>(p, stats.sum_grad,
                                                      stats.sum_hess, w);
    }
    XGBOOST_DEVICE float CalcGain(bst_node_t nid, ParamT const &p,
                                  tree::GradStats stats) const {
      return this->CalcGainGivenWeight(nid, p, stats, this->CalcWeight(nid, p, stats));
    }
  };

 public:
  /* Get a view to the evaluator that can be passed down to device. */
  template <typename ParamT = TrainParam> auto GetEvaluator() const {
    if (device_ != GenericParameter::kCpuId) {
      auto constraints = monotone_.ConstDeviceSpan();
      return SplitEvaluator<ParamT>{
          constraints, lower_bounds_.ConstDeviceSpan(),
          upper_bounds_.ConstDeviceSpan(), has_constraint_};
    } else {
      auto constraints = monotone_.ConstHostSpan();
      return SplitEvaluator<ParamT>{constraints, lower_bounds_.ConstHostSpan(),
                                    upper_bounds_.ConstHostSpan(),
                                    has_constraint_};
    }
  }

  template <bool CompiledWithCuda = WITH_CUDA()>
  void AddSplit(bst_node_t nodeid, bst_node_t leftid, bst_node_t rightid,
                bst_feature_t f, float left_weight, float right_weight) {
    if (!has_constraint_) {
      return;
    }
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t, common::Span<float> lower,
                           common::Span<float> upper,
                           common::Span<int> monotone) {
          lower[leftid] = lower[nodeid];
          upper[leftid] = upper[nodeid];

          lower[rightid] = lower[nodeid];
          upper[rightid] = upper[nodeid];
          int32_t c = monotone[f];
          bst_float mid = (left_weight + right_weight) / 2;

          SPAN_CHECK(!common::CheckNAN(mid));

          if (c < 0) {
            lower[leftid] = mid;
            upper[rightid] = mid;
          } else if (c > 0) {
            upper[leftid] = mid;
            lower[rightid] = mid;
          }
        },
        common::Range(0, 1), device_, false)
        .Eval(&lower_bounds_, &upper_bounds_, &monotone_);
  }
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_SPLIT_EVALUATOR_H_
