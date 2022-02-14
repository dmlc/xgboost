/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#pragma once
#include <thrust/random.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>
#include <vector>
#include "../common/categorical.h"
#include "../common/device_helpers.cuh"
#include "../common/random.h"
#include "param.h"

namespace xgboost {
namespace tree {

struct GPUTrainingParam {
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // L2 regularization factor
  float reg_lambda;
  // L1 regularization factor
  float reg_alpha;
  // maximum delta update we can add in weight estimation
  // this parameter can be used to stabilize update
  // default=0 means no constraint on weight delta
  float max_delta_step;
  float learning_rate;
  uint32_t max_cat_to_onehot;

  GPUTrainingParam() = default;

  XGBOOST_DEVICE explicit GPUTrainingParam(const TrainParam& param)
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step),
        learning_rate{param.learning_rate},
        max_cat_to_onehot{param.max_cat_to_onehot} {}
};

/**
 * @enum DefaultDirection node.cuh
 * @brief Default direction to be followed in case of missing values
 */
enum DefaultDirection {
  /** move to left child */
  kLeftDir = 0,
  /** move to right child */
  kRightDir
};

struct DeviceSplitCandidate {
  float loss_chg {-FLT_MAX};
  DefaultDirection dir {kLeftDir};
  int findex {-1};
  float fvalue {0};

  common::CatBitField split_cats;
  bool is_cat { false };

  GradientPairPrecise left_sum;
  GradientPairPrecise right_sum;

  XGBOOST_DEVICE DeviceSplitCandidate() {}  // NOLINT

  template <typename ParamT>
  XGBOOST_DEVICE void Update(const DeviceSplitCandidate& other,
                             const ParamT& param) {
    if (other.loss_chg > loss_chg &&
        other.left_sum.GetHess() >= param.min_child_weight &&
        other.right_sum.GetHess() >= param.min_child_weight) {
      *this = other;
    }
  }
  /**
   * \brief The largest encoded category in the split bitset
   */
  bst_cat_t MaxCat() const {
    // Reuse the fvalue for categorical values.
    return static_cast<bst_cat_t>(fvalue);
  }
  /**
   * \brief Return the best threshold for cat split, reset the value after return.
   */
  XGBOOST_DEVICE size_t PopBestThresh() {
    // fvalue is also being used for storing the threshold for categorical split
    auto best_thresh = static_cast<size_t>(this->fvalue);
    this->fvalue = 0;
    return best_thresh;
  }

  template <typename T>
  XGBOOST_DEVICE void SetCat(T c) {
    this->split_cats.Set(common::AsCat(c));
    fvalue = std::max(this->fvalue, static_cast<float>(c));
  }

  XGBOOST_DEVICE void Update(float loss_chg_in, DefaultDirection dir_in,
                             float fvalue_in, int findex_in,
                             GradientPairPrecise left_sum_in,
                             GradientPairPrecise right_sum_in,
                             bool cat,
                             const GPUTrainingParam& param) {
    if (loss_chg_in > loss_chg &&
        left_sum_in.GetHess() >= param.min_child_weight &&
        right_sum_in.GetHess() >= param.min_child_weight) {
      loss_chg = loss_chg_in;
      dir = dir_in;
      fvalue = fvalue_in;
      is_cat = cat;
      left_sum = left_sum_in;
      right_sum = right_sum_in;
      findex = findex_in;
    }
  }
  XGBOOST_DEVICE bool IsValid() const { return loss_chg > 0.0f; }

  friend std::ostream& operator<<(std::ostream& os, DeviceSplitCandidate const& c) {
    os << "loss_chg:" << c.loss_chg << ", "
       << "dir: " << c.dir << ", "
       << "findex: " << c.findex << ", "
       << "fvalue: " << c.fvalue << ", "
       << "is_cat: " << c.is_cat << ", "
       << "left sum: " << c.left_sum << ", "
       << "right sum: " << c.right_sum << std::endl;
    return os;
  }
};

template <typename T>
struct SumCallbackOp {
  // Running prefix
  T running_total;
  // Constructor
  XGBOOST_DEVICE SumCallbackOp() : running_total(T()) {}
  XGBOOST_DEVICE T operator()(T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};
}  // namespace tree
}  // namespace xgboost
