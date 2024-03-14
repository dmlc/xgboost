/*!
 * Copyright 2014-2024 by Contributors
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
/* The original structure xgboost::tree::TrainParam can't be used,
 * since std::vector are not copyable on sycl-devices.
 */
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

template <typename GradType>
using GradStats = xgboost::detail::GradientPairInternal<GradType>;

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_TREE_PARAM_H_
