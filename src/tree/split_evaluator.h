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

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "../common/math.h"
#include "../common/transform.h"
#include "param.h"
#include "xgboost/context.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"

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
    if (device != Context::kCpuId) {
      lower_bounds_.SetDevice(device);
      upper_bounds_.SetDevice(device);
      monotone_.SetDevice(device);
    }

    if (p.monotone_constraints.empty()) {
      monotone_.HostVector().resize(n_features, 0);
      has_constraint_ = false;
    } else {
      CHECK_LE(p.monotone_constraints.size(), n_features)
          << "The size of monotone constraint should be less or equal to the number of features.";
      monotone_.HostVector() = p.monotone_constraints;
      monotone_.HostVector().resize(n_features, 0);
      // Initialised to some small size, can grow if needed
      lower_bounds_.Resize(256, -std::numeric_limits<float>::max());
      upper_bounds_.Resize(256, std::numeric_limits<float>::max());
      has_constraint_ = true;
    }

    if (device_ != Context::kCpuId) {
      // Pull to device early.
      lower_bounds_.ConstDeviceSpan();
      upper_bounds_.ConstDeviceSpan();
      monotone_.ConstDeviceSpan();
    }
  }

  template <typename ParamT>
  struct SplitEvaluator {
    const int* constraints;
    const float* lower;
    const float* upper;
    bool has_constraint;

    template <typename GradientSumT>
    XGBOOST_DEVICE float CalcSplitGain(const ParamT& param, bst_node_t nidx, bst_feature_t fidx,
                                       GradientSumT const& left, GradientSumT const& right) const {
      int constraint = has_constraint ? constraints[fidx] : 0;
      const float negative_infinity = -std::numeric_limits<float>::infinity();
      float wleft = this->CalcWeight(nidx, param, left);
      float wright = this->CalcWeight(nidx, param, right);

      float gain = this->CalcGainGivenWeight(param, left, wleft) +
                    this->CalcGainGivenWeight(param, right, wright);

      if (constraint == 0) {
        return gain;
      } else if (constraint > 0) {
        return wleft <= wright ? gain : negative_infinity;
      } else {
        return wleft >= wright ? gain : negative_infinity;
      }
    }

    template <typename GradientSumT>
    XGBOOST_DEVICE float CalcWeight(bst_node_t nodeid, const ParamT &param,
                                    GradientSumT const& stats) const {
      float w = ::xgboost::tree::CalcWeight(param, stats);
      if (!has_constraint) {
        return w;
      }

      if (nodeid == kRootParentId) {
        return w;
      } else if (w < lower[nodeid]) {
        return lower[nodeid];
      } else if (w > upper[nodeid]) {
        return upper[nodeid];
      } else {
        return w;
      }
    }

    template <typename GradientSumT>
    XGBOOST_DEVICE float CalcWeightCat(ParamT const& param, GradientSumT const& stats) const {
      // FIXME(jiamingy): This is a temporary solution until we have categorical feature
      // specific regularization parameters.  During sorting we should try to avoid any
      // regularization.
      return ::xgboost::tree::CalcWeight(param, stats);
    }

    // Fast floating point division instruction on device
    XGBOOST_DEVICE float Divide(float a, float b) const {
#ifdef __CUDA_ARCH__
      return __fdividef(a, b);
#else
      return a / b;
#endif
    }

    template <typename GradientSumT>
    XGBOOST_DEVICE float CalcGainGivenWeight(ParamT const& p, GradientSumT const& stats,
                                             float w) const {
      if (stats.GetHess() <= 0) {
        return .0f;
      }
      // Avoiding tree::CalcGainGivenWeight can significantly reduce avg floating point error.
      if (p.max_delta_step == 0.0f && has_constraint == false) {
        return Divide(common::Sqr(ThresholdL1(stats.GetGrad(), p.reg_alpha)),
                      (stats.GetHess() + p.reg_lambda));
      }
      return tree::CalcGainGivenWeight<ParamT, float>(p, stats.GetGrad(),
                                                      stats.GetHess(), w);
    }
    template <typename GradientSumT>
    XGBOOST_DEVICE float CalcGain(bst_node_t nid, ParamT const &p,
                                  GradientSumT const& stats) const {
      return this->CalcGainGivenWeight(p, stats, this->CalcWeight(nid, p, stats));
    }
  };

 public:
  /* Get a view to the evaluator that can be passed down to device. */
  template <typename ParamT = TrainParam> auto GetEvaluator() const {
    if (device_ != Context::kCpuId) {
      auto constraints = monotone_.ConstDevicePointer();
      return SplitEvaluator<ParamT>{constraints, lower_bounds_.ConstDevicePointer(),
                                    upper_bounds_.ConstDevicePointer(), has_constraint_};
    } else {
      auto constraints = monotone_.ConstHostPointer();
      return SplitEvaluator<ParamT>{constraints, lower_bounds_.ConstHostPointer(),
                                    upper_bounds_.ConstHostPointer(), has_constraint_};
    }
  }

  template <bool CompiledWithCuda = WITH_CUDA()>
  void AddSplit(bst_node_t nodeid, bst_node_t leftid, bst_node_t rightid,
                bst_feature_t f, float left_weight, float right_weight) {
    if (!has_constraint_) {
      return;
    }

    size_t max_nidx = std::max(leftid, rightid);
    if (lower_bounds_.Size() <= max_nidx) {
      lower_bounds_.Resize(max_nidx * 2 + 1, -std::numeric_limits<float>::max());
    }
    if (upper_bounds_.Size() <= max_nidx) {
      upper_bounds_.Resize(max_nidx * 2 + 1, std::numeric_limits<float>::max());
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
        common::Range(0, 1), 1, device_)
        .Eval(&lower_bounds_, &upper_bounds_, &monotone_);
  }
};

enum SplitType {
  // numerical split
  kNum = 0,
  // onehot encoding based categorical split
  kOneHot = 1,
  // partition-based categorical split
  kPart = 2
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_SPLIT_EVALUATOR_H_
