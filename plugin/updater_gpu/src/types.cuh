/*!
 * Copyright 2016 Rory mitchell
*/
#pragma once
#include <xgboost/base.h>

namespace xgboost {
namespace tree {

typedef int16_t NodeIdT;

// gpair type defined with device accessible functions
struct gpu_gpair {
  float _grad;
  float _hess;

  __host__ __device__ __forceinline__ float grad() const { return _grad; }

  __host__ __device__ __forceinline__ float hess() const { return _hess; }

  __host__ __device__ gpu_gpair() : _grad(0), _hess(0) {}

  __host__ __device__ gpu_gpair(float g, float h) : _grad(g), _hess(h) {}

  __host__ __device__ gpu_gpair(bst_gpair gpair)
      : _grad(gpair.grad), _hess(gpair.hess) {}

  __host__ __device__ bool operator==(const gpu_gpair &rhs) const {
    return (_grad == rhs._grad) && (_hess == rhs._hess);
  }

  __host__ __device__ bool operator!=(const gpu_gpair &rhs) const {
    return !(*this == rhs);
  }

  __host__ __device__ gpu_gpair &operator+=(const gpu_gpair &rhs) {
    _grad += rhs._grad;
    _hess += rhs._hess;
    return *this;
  }

  __host__ __device__ gpu_gpair operator+(const gpu_gpair &rhs) const {
    gpu_gpair g;
    g._grad = _grad + rhs._grad;
    g._hess = _hess + rhs._hess;
    return g;
  }

  __host__ __device__ gpu_gpair &operator-=(const gpu_gpair &rhs) {
    _grad -= rhs._grad;
    _hess -= rhs._hess;
    return *this;
  }

  __host__ __device__ gpu_gpair operator-(const gpu_gpair &rhs) const {
    gpu_gpair g;
    g._grad = _grad - rhs._grad;
    g._hess = _hess - rhs._hess;
    return g;
  }

  friend std::ostream &operator<<(std::ostream &os, const gpu_gpair &g) {
    os << g.grad() << "/" << g.hess();
    return os;
  }

  __host__ __device__ void print() const {
    printf("%1.4f/%1.4f\n", grad(), hess());
  }

  __host__ __device__ bool approximate_compare(const gpu_gpair &b,
                                               float g_eps = 0.1,
                                               float h_eps = 0.1) const {
    float gdiff = abs(this->grad() - b.grad());
    float hdiff = abs(this->hess() - b.hess());

    return (gdiff <= g_eps) && (hdiff <= h_eps);
  }
};

struct Item {
  bst_uint instance_id;
  float fvalue;
  gpu_gpair gpair;
};

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

  __host__ __device__ GPUTrainingParam(float min_child_weight_in,
                                       float reg_lambda_in, float reg_alpha_in,
                                       float max_delta_step_in)
      : min_child_weight(min_child_weight_in), reg_lambda(reg_lambda_in),
        reg_alpha(reg_alpha_in), max_delta_step(max_delta_step_in) {}
};

struct Split {
  float loss_chg;
  bool missing_left;
  float fvalue;
  int findex;
  gpu_gpair left_sum;
  gpu_gpair right_sum;

  __host__ __device__ Split()
      : loss_chg(-FLT_MAX), missing_left(true), fvalue(0) {}

  __device__ void Update(float loss_chg_in, bool missing_left_in,
                         float fvalue_in, int findex_in, gpu_gpair left_sum_in,
                         gpu_gpair right_sum_in,
                         const GPUTrainingParam &param) {
    if (loss_chg_in > loss_chg && left_sum_in.hess() > param.min_child_weight &&
        right_sum_in.hess() > param.min_child_weight) {
      loss_chg = loss_chg_in;
      missing_left = missing_left_in;
      fvalue = fvalue_in;
      left_sum = left_sum_in;
      right_sum = right_sum_in;
      findex = findex_in;
    }
  }

  // Does not check minimum weight
  __device__ void Update(Split &s) { // NOLINT
    if (s.loss_chg > loss_chg) {
      loss_chg = s.loss_chg;
      missing_left = s.missing_left;
      fvalue = s.fvalue;
      findex = s.findex;
      left_sum = s.left_sum;
      right_sum = s.right_sum;
    }
  }

  __device__ void Print() {
    printf("Loss: %1.4f\n", loss_chg);
    printf("Missing left: %d\n", missing_left);
    printf("fvalue: %1.4f\n", fvalue);
    printf("Left sum: ");
    left_sum.print();

    printf("Right sum: ");
    right_sum.print();
  }
};

struct split_reduce_op {
  template <typename T>
  __device__ __forceinline__ T operator()(T &a, T b) { // NOLINT
    b.Update(a);
    return b;
  }
};

struct Node {
  gpu_gpair sum_gradients;
  float root_gain;
  float weight;

  Split split;

  __host__ __device__ Node() : weight(0), root_gain(0) {}

  __host__ __device__ Node(gpu_gpair sum_gradients_in, float root_gain_in,
                           float weight_in) {
    sum_gradients = sum_gradients_in;
    root_gain = root_gain_in;
    weight = weight_in;
  }

  __host__ __device__ bool IsLeaf() { return split.loss_chg == -FLT_MAX; }
};
}  // namespace tree
}  // namespace xgboost
