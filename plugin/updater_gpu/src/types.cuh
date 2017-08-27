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

  /** Tells whether this node is part of the decision tree */
  HOST_DEV_INLINE bool IsUnused() const { return (idx == UNUSED_NODE); }

  /** Tells whether this node is a leaf of the decision tree */
  HOST_DEV_INLINE bool IsLeaf() const {
    return (!IsUnused() && (fidx == UNUSED_NODE));
  }
};

}  // namespace tree
}  // namespace xgboost
