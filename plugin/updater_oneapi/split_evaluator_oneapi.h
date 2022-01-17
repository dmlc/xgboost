/*!
 * Copyright 2018-2021 by Contributors
 */

#ifndef XGBOOST_TREE_SPLIT_EVALUATOR_ONEAPI_H_
#define XGBOOST_TREE_SPLIT_EVALUATOR_ONEAPI_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <utility>
#include <vector>
#include <limits>

#include "param_oneapi.h"

#include "xgboost/tree_model.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/generic_parameters.h"
#include "../../src/common/transform.h"
#include "../../src/common/math.h"
#include "../../src/tree/param.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace tree {

/*! \brief OneAPI implementation of TreeEvaluator, with USM memory for temporary buffer to access on device.
 *         It also contains own implementation of SplitEvaluator for device compilation, because some of the
           functions from the original SplitEvaluator are currently not supported
 */

template<typename GradType>
class TreeEvaluatorOneAPI {
  // hist and exact use parent id to calculate constraints.
  static constexpr bst_node_t kRootParentId =
      (-1 & static_cast<bst_node_t>((1U << 31) - 1));

  USMVector<GradType> lower_bounds_;
  USMVector<GradType> upper_bounds_;
  USMVector<int32_t> monotone_;
  TrainParamOneAPI param_;
  sycl::queue qu_;
  bool has_constraint_;

 public:
  TreeEvaluatorOneAPI(sycl::queue qu, TrainParam const& p, bst_feature_t n_features) {
    qu_ = qu;
    if (p.monotone_constraints.empty()) {
      monotone_.Resize(qu_, n_features, 0);
      has_constraint_ = false;
    } else {
      monotone_ = USMVector<int32_t>(qu_, p.monotone_constraints);
      monotone_.Resize(qu_, n_features, 0);
      lower_bounds_.Resize(qu_, p.MaxNodes(), -std::numeric_limits<GradType>::max());
      upper_bounds_.Resize(qu_, p.MaxNodes(), std::numeric_limits<GradType>::max());
      has_constraint_ = true;
    }
    param_ = TrainParamOneAPI(p);
  }

  struct SplitEvaluator {
    int* constraints;
    GradType* lower;
    GradType* upper;
    bool has_constraint;
    TrainParamOneAPI param;

    GradType CalcSplitGain(bst_node_t nidx,
                        bst_feature_t fidx,
                        const GradStatsOneAPI<GradType>& left,
                        const GradStatsOneAPI<GradType>& right) const {
      int constraint = constraints[fidx];
      const GradType negative_infinity = -std::numeric_limits<GradType>::infinity();
      GradType wleft = this->CalcWeight(nidx, left);
      GradType wright = this->CalcWeight(nidx, right);

      GradType gain = this->CalcGainGivenWeight(nidx, left, wleft) +
                    this->CalcGainGivenWeight(nidx, right, wright);
      if (constraint == 0) {
        return gain;
      } else if (constraint > 0) {
        return wleft <= wright ? gain : negative_infinity;
      } else {
        return wleft >= wright ? gain : negative_infinity;
      }
    }

    inline GradType ThresholdL1OneAPI(GradType w, GradType alpha) const {
      if (w > + alpha) {
        return w - alpha;
      }
      if (w < - alpha) {
        return w + alpha;
      }
      return 0.0;
    }

    inline GradType CalcWeightOneAPI(GradType sum_grad, GradType sum_hess) const {
      if (sum_hess < param.min_child_weight || sum_hess <= 0.0) {
        return 0.0;
      }
      GradType dw = -this->ThresholdL1OneAPI(sum_grad, param.reg_alpha) / (sum_hess + param.reg_lambda);
      if (param.max_delta_step != 0.0f && std::abs(dw) > param.max_delta_step) {
        dw = sycl::copysign((GradType)param.max_delta_step, dw);
      }         
      return dw;
    }

    inline GradType CalcWeight(bst_node_t nodeid, const GradStatsOneAPI<GradType>& stats) const {
      GradType w = this->CalcWeightOneAPI(stats.GetGrad(), stats.GetHess());
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

    inline GradType Sqr(GradType a) const { return a * a; }

    inline GradType CalcGainGivenWeight(GradType sum_grad, GradType sum_hess, GradType w) const {
      return -(2.0f * sum_grad * w + (sum_hess + param.reg_lambda) * this->Sqr(w));
    }

    inline GradType CalcGainGivenWeight(bst_node_t nid, const GradStatsOneAPI<GradType>& stats, GradType w) const {
      if (stats.GetHess() <= 0) {
        return .0f;
      }
      // Avoiding tree::CalcGainGivenWeight can significantly reduce avg floating point error.
      if (param.max_delta_step == 0.0f && has_constraint == false) {
        return this->Sqr(this->ThresholdL1OneAPI(stats.sum_grad, param.reg_alpha)) /
               (stats.sum_hess + param.reg_lambda);
      }
      return this->CalcGainGivenWeight(stats.sum_grad, stats.sum_hess, w);
    }

    GradType CalcGain(bst_node_t nid, const GradStatsOneAPI<GradType>& stats) const {
      return this->CalcGainGivenWeight(nid, stats, this->CalcWeight(nid, stats));
    }
  };

 public:
  /* Get a view to the evaluator that can be passed down to device. */
  auto GetEvaluator() {
    return SplitEvaluator{monotone_.Data(),
                          lower_bounds_.Data(),
                          upper_bounds_.Data(),
                          has_constraint_,
                          param_};
  }

  void AddSplit(bst_node_t nodeid, bst_node_t leftid, bst_node_t rightid,
                bst_feature_t f, GradType left_weight, GradType right_weight) {
    if (!has_constraint_) {
      return;
    }
    GradType* lower = lower_bounds_.Data();
    GradType* upper = upper_bounds_.Data();
    int* monotone = monotone_.Data();
    qu_.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<>(sycl::range<1>(1), [=](sycl::item<1> pid) {
        lower[leftid] = lower[nodeid];
        upper[leftid] = upper[nodeid];

        lower[rightid] = lower[nodeid];
        upper[rightid] = upper[nodeid];
        int32_t c = monotone[f];
        GradType mid = (left_weight + right_weight) / 2;

        if (c < 0) {
          lower[leftid] = mid;
          upper[rightid] = mid;
        } else if (c > 0) {
          upper[leftid] = mid;
          lower[rightid] = mid;
        }
      });
    }).wait();
  }
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_SPLIT_EVALUATOR_ONEAPI_H_
