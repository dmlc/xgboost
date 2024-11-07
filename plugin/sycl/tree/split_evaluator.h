/*!
 * Copyright 2018-2024 by Contributors
 */

#ifndef PLUGIN_SYCL_TREE_SPLIT_EVALUATOR_H_
#define PLUGIN_SYCL_TREE_SPLIT_EVALUATOR_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <utility>
#include <vector>
#include <limits>

#include "param.h"
#include "../data.h"

#include "xgboost/tree_model.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/context.h"
#include "../../src/common/transform.h"
#include "../../src/common/math.h"
#include "../../src/tree/param.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace tree {

/*! \brief SYCL implementation of TreeEvaluator, with USM memory for temporary buffer to access on device.
 *         It also contains own implementation of SplitEvaluator for device compilation, because some of the
           functions from the original SplitEvaluator are currently not supported
 */

template<typename GradType>
class TreeEvaluator {
  // hist and exact use parent id to calculate constraints.
  static constexpr bst_node_t kRootParentId =
      (-1 & static_cast<bst_node_t>((1U << 31) - 1));

  USMVector<GradType> lower_bounds_;
  USMVector<GradType> upper_bounds_;
  USMVector<int> monotone_;
  TrainParam param_;
  ::sycl::queue qu_;
  bool has_constraint_;

 public:
  void Reset(::sycl::queue qu, xgboost::tree::TrainParam const& p, bst_feature_t n_features) {
    qu_ = qu;

    has_constraint_ = false;
    for (const auto& constraint : p.monotone_constraints) {
      if (constraint != 0) {
        has_constraint_ = true;
        break;
      }
    }

    if (has_constraint_) {
      monotone_.Resize(&qu_, n_features, 0);
      qu_.memcpy(monotone_.Data(), p.monotone_constraints.data(),
                 sizeof(int) * p.monotone_constraints.size());
      qu_.wait();

      lower_bounds_.Resize(&qu_, p.MaxNodes(), std::numeric_limits<GradType>::lowest());
      upper_bounds_.Resize(&qu_, p.MaxNodes(), std::numeric_limits<GradType>::max());
    }
    param_ = TrainParam(p);
  }

  bool HasConstraint() const {
    return has_constraint_;
  }

  TreeEvaluator(::sycl::queue qu, xgboost::tree::TrainParam const& p, bst_feature_t n_features) {
    Reset(qu, p, n_features);
  }

  struct SplitEvaluator {
    const int* constraints;
    const GradType* lower;
    const GradType* upper;
    bool has_constraint;
    TrainParam param;

    GradType CalcSplitGain(bst_node_t nidx,
                        bst_feature_t fidx,
                        const GradStats<GradType>& left,
                        const GradStats<GradType>& right) const {
      const GradType negative_infinity = -std::numeric_limits<GradType>::infinity();
      GradType wleft = this->CalcWeight(nidx, left);
      GradType wright = this->CalcWeight(nidx, right);

      GradType gain = this->CalcGainGivenWeight(nidx, left,  wleft) +
                      this->CalcGainGivenWeight(nidx, right, wright);
      if (!has_constraint) {
        return gain;
      }

      int constraint = constraints[fidx];
      if (constraint == 0) {
        return gain;
      } else if (constraint > 0) {
        return wleft <= wright ? gain : negative_infinity;
      } else {
        return wleft >= wright ? gain : negative_infinity;
      }
    }

    inline static GradType ThresholdL1(GradType w, float alpha) {
      if (w > + alpha) {
        return w - alpha;
      }
      if (w < - alpha) {
        return w + alpha;
      }
      return 0.0;
    }

    inline GradType CalcWeight(GradType sum_grad, GradType sum_hess) const {
      if (sum_hess < param.min_child_weight || sum_hess <= 0.0) {
        return 0.0;
      }
      GradType dw = -this->ThresholdL1(sum_grad, param.reg_alpha) / (sum_hess + param.reg_lambda);
      if (param.max_delta_step != 0.0f && std::abs(dw) > param.max_delta_step) {
        dw = ::sycl::copysign((GradType)param.max_delta_step, dw);
      }
      return dw;
    }

    inline GradType CalcWeight(bst_node_t nodeid, const GradStats<GradType>& stats) const {
      GradType w = this->CalcWeight(stats.GetGrad(), stats.GetHess());
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

    inline GradType CalcGainGivenWeight(GradType sum_grad, GradType sum_hess, GradType w) const {
      return -(2.0f * sum_grad * w + (sum_hess + param.reg_lambda) * xgboost::common::Sqr(w));
    }

    inline GradType CalcGainGivenWeight(bst_node_t nid, const GradStats<GradType>& stats,
                                        GradType w) const {
      if (stats.GetHess() <= 0) {
        return .0f;
      }
      // Avoiding tree::CalcGainGivenWeight can significantly reduce avg floating point error.
      if (param.max_delta_step == 0.0f && has_constraint == false) {
        return xgboost::common::Sqr(this->ThresholdL1(stats.GetGrad(), param.reg_alpha)) /
               (stats.GetHess() + param.reg_lambda);
      }
      return this->CalcGainGivenWeight(stats.GetGrad(), stats.GetHess(), w);
    }

    GradType CalcGain(bst_node_t nid, const GradStats<GradType>& stats) const {
      return this->CalcGainGivenWeight(nid, stats, this->CalcWeight(nid, stats));
    }
  };

 public:
  /* Get a view to the evaluator that can be passed down to device. */
  auto GetEvaluator() const {
    return SplitEvaluator{monotone_.DataConst(),
                          lower_bounds_.DataConst(),
                          upper_bounds_.DataConst(),
                          has_constraint_,
                          param_};
  }

  void AddSplit(bst_node_t nodeid, bst_node_t leftid, bst_node_t rightid,
                bst_feature_t f, GradType left_weight, GradType right_weight) {
    if (!has_constraint_) {
      return;
    }

    lower_bounds_[leftid] = lower_bounds_[nodeid];
    upper_bounds_[leftid] = upper_bounds_[nodeid];

    lower_bounds_[rightid] = lower_bounds_[nodeid];
    upper_bounds_[rightid] = upper_bounds_[nodeid];
    int32_t c = monotone_[f];
    GradType mid = (left_weight + right_weight) / 2;

    if (c < 0) {
      lower_bounds_[leftid] = mid;
      upper_bounds_[rightid] = mid;
    } else if (c > 0) {
      upper_bounds_[leftid] = mid;
      lower_bounds_[rightid] = mid;
    }
  }
};
}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_SPLIT_EVALUATOR_H_
