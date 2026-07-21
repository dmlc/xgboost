/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file split_evaluator.h
 * \brief Used for implementing a loss term specific to decision trees. Useful for custom regularisation.
 * \author Henry Gouk
 */

#ifndef XGBOOST_TREE_SPLIT_EVALUATOR_H_
#define XGBOOST_TREE_SPLIT_EVALUATOR_H_

#include <xgboost/base.h>

#include <algorithm>    // for any_of
#include <cstddef>      // for size_t
#include <limits>       // for numeric_limits
#include <type_traits>  // for void_t, false_type, declval

#include "../common/math.h"
#include "../common/transform.h"
#include "param.h"  // for TrainParam
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost::tree {
namespace split_impl {
template <typename T, typename = void>
struct IsVectorGradientSum : std::false_type {};

template <typename T>
struct IsVectorGradientSum<T,
                           std::void_t<decltype(std::declval<T const&>().Size()),
                                       decltype(std::declval<T const&>()(std::size_t{}).GetGrad()),
                                       decltype(std::declval<T const&>()(std::size_t{}).GetHess())>>
    : std::true_type {};

template <typename... T>
using EnableVecGrad = std::enable_if_t<(split_impl::IsVectorGradientSum<T>::value && ...), int>;

template <typename... T>
using EnableScaGrad = std::enable_if_t<!(split_impl::IsVectorGradientSum<T>::value || ...), int>;
}  // namespace split_impl

class TreeEvaluator {
  HostDeviceVector<float> lower_bounds_;
  HostDeviceVector<float> upper_bounds_;
  HostDeviceVector<int32_t> monotone_;
  DeviceOrd device_;
  bool has_constraint_;

 public:
  TreeEvaluator(TrainParam const& p, bst_feature_t n_features, DeviceOrd device) {
    device_ = device;
    if (device.IsCUDA()) {
      lower_bounds_.SetDevice(device);
      upper_bounds_.SetDevice(device);
      monotone_.SetDevice(device);
    }

    if (!p.monotone_constraints.empty()) {
      CHECK_LE(p.monotone_constraints.size(), n_features)
          << "The size of monotone constraint should be less or equal to the number of features.";
    }
    monotone_.HostVector() = p.monotone_constraints;
    monotone_.HostVector().resize(n_features, 0);
    has_constraint_ = std::any_of(p.monotone_constraints.cbegin(), p.monotone_constraints.cend(),
                                  [](auto v) { return v != 0; });
    if (has_constraint_) {
      // Initialised to some small size, can grow if needed
      lower_bounds_.Resize(256, -std::numeric_limits<float>::max());
      upper_bounds_.Resize(256, std::numeric_limits<float>::max());
    }

    if (device_.IsCUDA()) {
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

    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcSplitGain(ParamT const& param, bst_node_t nidx, bst_feature_t fidx,
                                       GradientSumT const& left, GradientSumT const& right) const {
      constexpr float kNegInf = -std::numeric_limits<float>::infinity();
      auto left_hess = left.GetHess();
      auto right_hess = right.GetHess();
      if (!IsValidSplit(param, left_hess, right_hess)) {
        return kNegInf;
      }

      int constraint = has_constraint ? constraints[fidx] : 0;
      float wleft = this->CalcWeight(nidx, param, left);
      float wright = this->CalcWeight(nidx, param, right);

      float gain = this->CalcGainGivenWeight(param, left, wleft) +
                   this->CalcGainGivenWeight(param, right, wright);

      if (constraint == 0) {
        // no constraint
        return gain;
      } else if (constraint > 0) {
        return wleft <= wright ? gain : kNegInf;
      } else {
        return wleft >= wright ? gain : kNegInf;
      }
    }

    template <typename LeftGradientSumT, typename RightGradientSumT,
              split_impl::EnableVecGrad<LeftGradientSumT, RightGradientSumT> = 0>
    XGBOOST_DEVICE double CalcSplitGain(ParamT const& param, [[maybe_unused]] bst_node_t nidx,
                                        [[maybe_unused]] bst_feature_t fidx,
                                        LeftGradientSumT const& left,
                                        RightGradientSumT const& right) const {
      auto n_targets = left.Size();
      double left_hess{0.0};
      double right_hess{0.0};
      double gain{0.0};
      for (std::size_t t = 0; t < n_targets; ++t) {
        auto const left_t = left(t);
        auto const right_t = right(t);
        left_hess += left_t.GetHess();
        right_hess += right_t.GetHess();
        gain += tree::CalcGain(param, left_t.GetGrad(), left_t.GetHess());
        gain += tree::CalcGain(param, right_t.GetGrad(), right_t.GetHess());
      }

      auto k = static_cast<double>(n_targets);
      if (!IsValidSplit(param, left_hess / k, right_hess / k)) {
        return -std::numeric_limits<double>::infinity();
      }
      return gain;
    }

    // Weight
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcWeight(bst_node_t nidx, ParamT const& param,
                                    GradientSumT const& stats) const {
      // boxed by max_delta_step
      float w = ::xgboost::tree::CalcWeight(param, stats);
      if (!has_constraint) {
        return w;
      }
      // Calculate bound weight, boxed by monotone constraint
      if (w < lower[nidx]) {
        return lower[nidx];
      } else if (w > upper[nidx]) {
        return upper[nidx];
      } else {
        return w;
      }
    }

    template <typename GradientSumT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE void CalcWeight(bst_node_t nidx, ParamT const& param, GradientSumT const& stats,
                                   linalg::VectorView<float> out) const {
      for (std::size_t t = 0; t < stats.Size(); ++t) {
        out(t) = this->CalcWeight(nidx, param, stats(t));
      }
    }

    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcWeightCat(ParamT const& param, GradientSumT const& stats) const {
      // FIXME(jiamingy): This is a temporary solution until we have categorical feature
      // specific regularization parameters.  During sorting we should try to avoid any
      // regularization.
      return ::xgboost::tree::CalcWeight(param, stats);
    }
    template <typename GradientSumT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE void CalcWeightCat(ParamT const& param, GradientSumT const& stats,
                                      linalg::VectorView<float> out) const {
      for (std::size_t t = 0; t < stats.Size(); ++t) {
        out(t) = this->CalcWeightCat(param, stats(t));
      }
    }

    // Gain given weight
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcGainGivenWeight(ParamT const& p, GradientSumT const& stats,
                                             float w) const {
      if (stats.GetHess() <= 0) {
        return .0f;
      }
      return tree::CalcGainGivenWeight<ParamT>(p, stats.GetGrad(), stats.GetHess(), w);
    }

    template <typename GradientSumT, typename WeightT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE double CalcGainGivenWeight(ParamT const& p, GradientSumT const& stats,
                                              WeightT const& weight) const {
      double gain{0.0};
      for (std::size_t t = 0; t < stats.Size(); ++t) {
        auto const stats_t = stats(t);
        gain += tree::CalcGainGivenWeight(p, stats_t.GetGrad(), stats_t.GetHess(), weight(t));
      }
      return gain;
    }

    // Gain
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcGain(bst_node_t nidx, ParamT const& p,
                                  GradientSumT const& stats) const {
      return this->CalcGainGivenWeight(p, stats, this->CalcWeight(nidx, p, stats));
    }

    template <typename GradientSumT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE double CalcGain(bst_node_t nidx, ParamT const& p,
                                   GradientSumT const& stats) const {
      double gain{0.0};
      for (std::size_t t = 0, n_targets = stats.Size(); t < n_targets; ++t) {
        auto const stats_t = stats(t);
        auto weight = this->CalcWeight(nidx, p, stats_t);
        gain += tree::CalcGainGivenWeight(p, stats_t.GetGrad(), stats_t.GetHess(), weight);
      }
      return gain;
    }
  };

 public:
  /* Get a view to the evaluator that can be passed down to device. */
  template <typename ParamT = TrainParam>
  auto GetEvaluator() const {
    if (device_.IsCUDA()) {
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
  void AddSplit(bst_node_t nodeid, bst_node_t leftid, bst_node_t rightid, bst_feature_t f,
                float left_weight, float right_weight) {
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
        [=] XGBOOST_DEVICE(size_t, common::Span<float> lower, common::Span<float> upper,
                           common::Span<int> monotone) {
          lower[leftid] = lower[nodeid];
          upper[leftid] = upper[nodeid];

          lower[rightid] = lower[nodeid];
          upper[rightid] = upper[nodeid];
          int32_t c = monotone[f];
          float mid = (left_weight + right_weight) / 2.0f;

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
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_SPLIT_EVALUATOR_H_
