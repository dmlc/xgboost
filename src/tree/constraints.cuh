/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file Various constraints used in GPU_Hist.
 */
#ifndef XGBOOST_TREE_CONSTRAINTS_H_
#define XGBOOST_TREE_CONSTRAINTS_H_

#include <dmlc/json.h>

#include <cinttypes>
#include <vector>

#include "param.h"
#include "xgboost/span.h"
#include "../common/bitfield.h"
#include "../common/device_helpers.cuh"

namespace xgboost {

// This class implements monotonic constraints, L1, L2 regularization.
struct ValueConstraint {
  double lower_bound;
  double upper_bound;
  XGBOOST_DEVICE ValueConstraint()
      : lower_bound(-std::numeric_limits<double>::max()),
        upper_bound(std::numeric_limits<double>::max()) {}
  inline static void Init(tree::TrainParam *param, unsigned num_feature) {
    param->monotone_constraints.resize(num_feature, 0);
  }
  template <typename ParamT, typename GpairT>
  XGBOOST_DEVICE inline double CalcWeight(const ParamT &param, GpairT stats) const {
    double w = xgboost::tree::CalcWeight(param, stats);
    if (w < lower_bound) {
      return lower_bound;
    }
    if (w > upper_bound) {
      return upper_bound;
    }
    return w;
  }

  template <typename ParamT>
  XGBOOST_DEVICE inline double CalcGain(const ParamT &param, tree::GradStats stats) const {
    return tree::CalcGainGivenWeight<ParamT, float>(param, stats.sum_grad, stats.sum_hess,
                                                    CalcWeight(param, stats));
  }

  template <typename ParamT>
  XGBOOST_DEVICE inline double CalcSplitGain(const ParamT &param, int constraint,
                                             tree::GradStats left, tree::GradStats right) const {
    const double negative_infinity = -std::numeric_limits<double>::infinity();
    double wleft = CalcWeight(param, left);
    double wright = CalcWeight(param, right);
    double gain =
      tree::CalcGainGivenWeight<ParamT, float>(param, left.sum_grad, left.sum_hess, wleft) +
      tree::CalcGainGivenWeight<ParamT, float>(param, right.sum_grad, right.sum_hess, wright);
    if (constraint == 0) {
      return gain;
    } else if (constraint > 0) {
      return wleft <= wright ? gain : negative_infinity;
    } else {
      return wleft >= wright ? gain : negative_infinity;
    }
  }
  template <typename GpairT>
  void SetChild(const tree::TrainParam &param, bst_uint split_index,
                       GpairT left, GpairT right, ValueConstraint *cleft,
                       ValueConstraint *cright) {
    int c = param.monotone_constraints.at(split_index);
    *cleft = *this;
    *cright = *this;
    if (c == 0) {
      return;
    }
    double wleft = CalcWeight(param, left);
    double wright = CalcWeight(param, right);
    double mid = (wleft + wright) / 2;
    CHECK(!std::isnan(mid));
    if (c < 0) {
      cleft->lower_bound = mid;
      cright->upper_bound = mid;
    } else {
      cleft->upper_bound = mid;
      cright->lower_bound = mid;
    }
  }
};

// Feature interaction constraints built for GPU Hist updater.
struct FeatureInteractionConstraintDevice {
 protected:
  // Whether interaction constraint is used.
  bool has_constraint_;
  // n interaction sets.
  size_t n_sets_;

  // The parsed feature interaction constraints as CSR.
  dh::device_vector<bst_feature_t> d_fconstraints_;
  common::Span<bst_feature_t> s_fconstraints_;
  dh::device_vector<size_t> d_fconstraints_ptr_;
  common::Span<size_t> s_fconstraints_ptr_;
  /* Interaction sets for each feature as CSR.  For an input like:
   * [[0, 1], [1, 2]], this will have values:
   *
   * fid:                                |0 | 1  | 2|
   * sets a feature belongs to(d_sets_): |0 |0, 1| 1|
   *
   * d_sets_ptr_:                        |0, 1, 3, 4|
   */
  dh::device_vector<bst_feature_t> d_sets_;
  common::Span<bst_feature_t> s_sets_;
  dh::device_vector<size_t> d_sets_ptr_;
  common::Span<size_t> s_sets_ptr_;

  // Allowed features attached to each node, have n_nodes bitfields,
  // each of size n_features.
  std::vector<dh::device_vector<LBitField64::value_type>> node_constraints_storage_;
  std::vector<LBitField64> node_constraints_;
  common::Span<LBitField64> s_node_constraints_;

  // buffer storing return feature list from Query, of size n_features.
  dh::device_vector<bst_feature_t> result_buffer_;
  common::Span<bst_feature_t> s_result_buffer_;

  // Temp buffers, one bit for each possible feature.
  dh::device_vector<LBitField64::value_type> output_buffer_bits_storage_;
  LBitField64 output_buffer_bits_;
  dh::device_vector<LBitField64::value_type> input_buffer_bits_storage_;
  LBitField64 input_buffer_bits_;
  /*
   * Combined features from all interaction sets that one feature belongs to.
   * For an input with [[0, 1], [1, 2]], the feature 1 belongs to sets {0, 1}
   */
  dh::device_vector<LBitField64::value_type> d_feature_buffer_storage_;
  LBitField64 feature_buffer_;  // of Size n features.

  // Clear out all temp buffers except for `feature_buffer_', which is
  // handled in `Split'.
  void ClearBuffers();

 public:
  size_t Features() const;
  FeatureInteractionConstraintDevice() = default;
  void Configure(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(tree::TrainParam const& param, int32_t const n_features);
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice const& that) = default;
  FeatureInteractionConstraintDevice(FeatureInteractionConstraintDevice&& that) = default;
  /*! \brief Reset before constructing a new tree. */
  void Reset();
  /*! \brief Return a list of features given node id */
  common::Span<bst_feature_t> QueryNode(int32_t nid);
  /*!
   * \brief Return a list of selected features from given feature_list and node id.
   *
   * \param feature_list A list of features
   * \param nid node id
   *
   * \return A list of features picked from `feature_list' that conform to constraints in
   * node.
   */
  common::Span<bst_feature_t> Query(common::Span<bst_feature_t> feature_list, int32_t nid);
  /*! \brief Apply split for node_id. */
  void Split(bst_node_t node_id, bst_feature_t feature_id, bst_node_t left_id, bst_node_t right_id);
};

}      // namespace xgboost
#endif  // XGBOOST_TREE_CONSTRAINTS_H_
