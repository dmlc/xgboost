/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include <xgboost/tree_model.h>
#include <cfloat>
#include <tuple>  // The linter is not very smart and thinks we need this

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

  __host__ __device__ GPUTrainingParam() {}

  __host__ __device__ GPUTrainingParam(const TrainParam &param) 
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step) {}

  __host__ __device__ GPUTrainingParam(float min_child_weight_in,
                                       float reg_lambda_in, float reg_alpha_in,
                                       float max_delta_step_in)
      : min_child_weight(min_child_weight_in),
        reg_lambda(reg_lambda_in),
        reg_alpha(reg_alpha_in),
        max_delta_step(max_delta_step_in) {}
};

struct Split {
  float loss_chg;
  bool missing_left;
  float fvalue;
  int findex;
  bst_gpair left_sum;
  bst_gpair right_sum;

  __host__ __device__ Split()
      : loss_chg(-FLT_MAX), missing_left(true), fvalue(0), findex(-1) {}

  __device__ void Update(float loss_chg_in, bool missing_left_in,
                         float fvalue_in, int findex_in, bst_gpair left_sum_in,
                         bst_gpair right_sum_in,
                         const GPUTrainingParam &param) {
    if (loss_chg_in > loss_chg &&
        left_sum_in.hess>= param.min_child_weight &&
        right_sum_in.hess>= param.min_child_weight) {
      loss_chg = loss_chg_in;
      missing_left = missing_left_in;
      fvalue = fvalue_in;
      left_sum = left_sum_in;
      right_sum = right_sum_in;
      findex = findex_in;
    }
  }

  // Does not check minimum weight
  __device__ void Update(Split &s) {  // NOLINT
    if (s.loss_chg > loss_chg) {
      loss_chg = s.loss_chg;
      missing_left = s.missing_left;
      fvalue = s.fvalue;
      findex = s.findex;
      left_sum = s.left_sum;
      right_sum = s.right_sum;
    }
  }

  __device__ bool IsValid()const 
  {
    return loss_chg > 0.0f;
  }
  //__host__ __device__ void Print() {
  //  printf("Loss: %1.4f\n", loss_chg);
  //  printf("Missing left: %d\n", missing_left);
  //  printf("fvalue: %1.4f\n", fvalue);
  //  printf("Left sum: ");
  //  left_sum.print();

  //  printf("Right sum: ");
  //  right_sum.print();
  //}
};

struct split_reduce_op {
  template <typename T>
  __device__ __forceinline__ T operator()(T &a, T b) {  // NOLINT
    b.Update(a);
    return b;
  }
};

/**
 * @enum DefaultDirection node.cuh
 * @brief Default direction to be followed in case of missing values
 */
enum DefaultDirection {
  /** move to left child */
  LeftDir = 0,
  /** move to right child */
  RightDir
};
typedef int node_id_t;

/** used to assign default id to a Node */
static const int UNUSED_NODE = -1;


struct DeviceDenseNode {
  bst_gpair sum_gradients;
  float root_gain;
  float weight;

  //Split split;

  //__host__ __device__ Node() : weight(0), root_gain(0) {}

  //__host__ __device__ Node(bst_gpair sum_gradients_in, float root_gain_in,
  //                         float weight_in) {
  //  sum_gradients = sum_gradients_in;
  //  root_gain = root_gain_in;
  //  weight = weight_in;
  //}

  /** default direction for missing values */
  DefaultDirection dir;
  /** threshold value for comparison */
  float fvalue;
  /** \brief The feature index. */
  int fidx;
  /** node id (used as key for reduce/scan) */
  node_id_t idx;

  //__host__ __device__ Node() : weight(0), root_gain(0) {}
  HOST_DEV_INLINE DeviceDenseNode()
      : sum_gradients(),
        root_gain(-FLT_MAX),
        weight(-FLT_MAX),
        dir(LeftDir),
        fvalue(0.f),
        fidx(UNUSED_NODE),
        idx(UNUSED_NODE) {}

  HOST_DEV_INLINE DeviceDenseNode(bst_gpair sum_gradients,node_id_t nidx,const GPUTrainingParam &param)
      : sum_gradients(sum_gradients),
        dir(LeftDir),
        fvalue(0.f),
        fidx(UNUSED_NODE),
        idx(nidx)
  {
    this->root_gain = CalcGain(param, sum_gradients.grad, sum_gradients.hess);
    this->weight = CalcWeight(param, sum_gradients.grad, sum_gradients.hess);
  }

  HOST_DEV_INLINE void  SetSplit(float fvalue,int fidx,DefaultDirection dir)
  {
    this->fvalue = fvalue;
    this->fidx = fidx;
    this->dir = dir;
  }

  //__host__ __device__ bool IsLeaf() { return split.loss_chg == -FLT_MAX; }
  /** Tells whether this node is part of the decision tree */
  HOST_DEV_INLINE bool IsUnused() const { return (idx == UNUSED_NODE); }
  //__host__ __device__ bool IsLeaf() { return split.loss_chg == -FLT_MAX; }
  /** Tells whether this node is a leaf of the decision tree */
  HOST_DEV_INLINE bool IsLeaf() const {
    return (!IsUnused() && (fidx == UNUSED_NODE));
  }
};

}  // namespace tree
}  // namespace xgboost
