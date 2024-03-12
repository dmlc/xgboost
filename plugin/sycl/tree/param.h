/*!
 * Copyright 2014-2023 by Contributors
 */
#ifndef PLUGIN_SYCL_TREE_PARAM_H_
#define PLUGIN_SYCL_TREE_PARAM_H_


#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>


#include "xgboost/parameter.h"
#include "xgboost/data.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../src/tree/param.h"
#pragma GCC diagnostic pop

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace tree {


/*! \brief Wrapper for necessary training parameters for regression tree to access on device */
struct TrainParam {
  float min_child_weight;
  float reg_lambda;
  float reg_alpha;
  float max_delta_step;


  TrainParam() {}


  explicit TrainParam(const xgboost::tree::TrainParam& param) {
    reg_lambda = param.reg_lambda;
    reg_alpha = param.reg_alpha;
    min_child_weight = param.min_child_weight;
    max_delta_step = param.max_delta_step;
  }
};

/*! \brief core statistics used for tree construction */
template<typename GradType>
struct GradStats {
  /*! \brief sum gradient statistics */
  GradType sum_grad { 0 };
  /*! \brief sum hessian statistics */
  GradType sum_hess { 0 };

 public:
  GradType GetGrad() const { return sum_grad; }
  GradType GetHess() const { return sum_hess; }

  GradStats<GradType>& operator+= (const GradStats<GradType>& rhs) {
    sum_grad += rhs.sum_grad;
    sum_hess += rhs.sum_hess;

    return *this;
  }

  GradStats<GradType>& operator-= (const GradStats<GradType>& rhs) {
    sum_grad -= rhs.sum_grad;
    sum_hess -= rhs.sum_hess;

    return *this;
  }

  friend GradStats<GradType> operator+ (GradStats<GradType> lhs,
                                        const GradStats<GradType> rhs) {
    lhs += rhs;
    return lhs;
  }

  friend GradStats<GradType> operator- (GradStats<GradType> lhs,
                                        const GradStats<GradType> rhs) {
    lhs -= rhs;
    return lhs;
  }


  friend std::ostream& operator<<(std::ostream& os, GradStats s) {
    os << s.GetGrad() << "/" << s.GetHess();
    return os;
  }

  GradStats() {}

  template <typename GpairT>
  explicit GradStats(const GpairT &sum)
      : sum_grad(sum.GetGrad()), sum_hess(sum.GetHess()) {}
  explicit GradStats(const GradType grad, const GradType hess)
      : sum_grad(grad), sum_hess(hess) {}
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_TREE_PARAM_H_
