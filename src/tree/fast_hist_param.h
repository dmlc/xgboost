/*!
 * Copyright 2017 by Contributors
 * \file updater_fast_hist.h
 * \brief parameters for histogram-based training
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_TREE_FAST_HIST_PARAM_H_
#define XGBOOST_TREE_FAST_HIST_PARAM_H_

namespace xgboost {
namespace tree {

/*! \brief training parameters for histogram-based training */
struct FastHistParam : public dmlc::Parameter<FastHistParam> {
  int colmat_dtype;
  // percentage threshold for treating a feature as sparse
  // e.g. 0.2 indicates a feature with fewer than 20% nonzeros is considered sparse
  double sparse_threshold;
  // use feature grouping? (default yes)
  int enable_feature_grouping;
  // when grouping features, how many "conflicts" to allow.
  // conflict is when an instance has nonzero values for two or more features
  // default is 0, meaning features should be strictly complementary
  double max_conflict_rate;
  // when grouping features, how much effort to expend to prevent singleton groups
  // we'll try to insert each feature into existing groups before creating a new group
  // for that feature; to save time, only up to (max_search_group) of existing groups
  // will be considered. If set to zero, ALL existing groups will be examined
  unsigned max_search_group;

  // declare the parameters
  DMLC_DECLARE_PARAMETER(FastHistParam) {
    DMLC_DECLARE_FIELD(sparse_threshold).set_range(0, 1.0).set_default(0.2)
        .describe("percentage threshold for treating a feature as sparse");
    DMLC_DECLARE_FIELD(enable_feature_grouping).set_lower_bound(0).set_default(0)
        .describe("if >0, enable feature grouping to ameliorate work imbalance "
                  "among worker threads");
    DMLC_DECLARE_FIELD(max_conflict_rate).set_range(0, 1.0).set_default(0)
        .describe("when grouping features, how many \"conflicts\" to allow."
       "conflict is when an instance has nonzero values for two or more features."
       "default is 0, meaning features should be strictly complementary.");
    DMLC_DECLARE_FIELD(max_search_group).set_lower_bound(0).set_default(100)
        .describe("when grouping features, how much effort to expend to prevent "
                  "singleton groups. We'll try to insert each feature into existing "
                  "groups before creating a new group for that feature; to save time, "
                  "only up to (max_search_group) of existing groups will be "
                  "considered. If set to zero, ALL existing groups will be examined.");
  }
};

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_FAST_HIST_PARAM_H_
