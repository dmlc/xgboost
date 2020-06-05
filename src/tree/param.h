/*!
 * Copyright 2014-2019 by Contributors
 * \file param.h
 * \brief training parameters, statistics used to support tree construction.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_PARAM_H_
#define XGBOOST_TREE_PARAM_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "xgboost/parameter.h"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {

/*! \brief training parameters for regression tree */
struct TrainParam : public XGBoostParameter<TrainParam> {
  // learning step size for a time
  float learning_rate;
  // minimum loss change required for a split
  float min_split_loss;
  // maximum depth of a tree
  int max_depth;
  // maximum number of leaves
  int max_leaves;
  // if using histogram based algorithm, maximum number of bins per feature
  int max_bin;
  // growing policy
  enum TreeGrowPolicy { kDepthWise = 0, kLossGuide = 1 };
  int grow_policy;

  //----- the rest parameters are less important ----
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // L2 regularization factor
  float reg_lambda;
  // L1 regularization factor
  float reg_alpha;
  // default direction choice
  int default_direction;
  // maximum delta update we can add in weight estimation
  // this parameter can be used to stabilize update
  // default=0 means no constraint on weight delta
  float max_delta_step;
  // whether we want to do subsample
  float subsample;
  // sampling method
  enum SamplingMethod { kUniform = 0, kGradientBased = 1 };
  int sampling_method;
  // whether to subsample columns in each split (node)
  float colsample_bynode;
  // whether to subsample columns in each level
  float colsample_bylevel;
  // whether to subsample columns during tree construction
  float colsample_bytree;
  // accuracy of sketch
  float sketch_eps;
  // accuracy of sketch
  float sketch_ratio;
  // option to open cacheline optimization
  bool cache_opt;
  // whether refresh updater needs to update the leaf values
  bool refresh_leaf;

  std::vector<int> monotone_constraints;
  // Stored as a JSON string.
  std::string interaction_constraints;

  // the criteria to use for ranking splits
  std::string split_evaluator;

  // ------ From cpu quantile histogram -------.
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
  DMLC_DECLARE_PARAMETER(TrainParam) {
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.3f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_FIELD(min_split_loss)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe(
            "Minimum loss reduction required to make a further partition.");
    DMLC_DECLARE_FIELD(max_depth)
        .set_lower_bound(0)
        .set_default(6)
        .describe(
            "Maximum depth of the tree; 0 indicates no limit; a limit is required "
            "for depthwise policy");
    DMLC_DECLARE_FIELD(max_leaves).set_lower_bound(0).set_default(0).describe(
        "Maximum number of leaves; 0 indicates no limit.");
    DMLC_DECLARE_FIELD(max_bin).set_lower_bound(2).set_default(256).describe(
        "if using histogram-based algorithm, maximum number of bins per feature");
    DMLC_DECLARE_FIELD(grow_policy)
        .set_default(kDepthWise)
        .add_enum("depthwise", kDepthWise)
        .add_enum("lossguide", kLossGuide)
        .describe(
            "Tree growing policy. 0: favor splitting at nodes closest to the node, "
            "i.e. grow depth-wise. 1: favor splitting at nodes with highest loss "
            "change. (cf. LightGBM)");
    DMLC_DECLARE_FIELD(min_child_weight)
        .set_lower_bound(0.0f)
        .set_default(1.0f)
        .describe("Minimum sum of instance weight(hessian) needed in a child.");
    DMLC_DECLARE_FIELD(reg_lambda)
        .set_lower_bound(0.0f)
        .set_default(1.0f)
        .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_alpha)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("L1 regularization on leaf weight");
    DMLC_DECLARE_FIELD(default_direction)
        .set_default(0)
        .add_enum("learn", 0)
        .add_enum("left", 1)
        .add_enum("right", 2)
        .describe("Default direction choice when encountering a missing value");
    DMLC_DECLARE_FIELD(max_delta_step)
        .set_lower_bound(0.0f)
        .set_default(0.0f)
        .describe("Maximum delta step we allow each tree's weight estimate to be. "\
                  "If the value is set to 0, it means there is no constraint");
    DMLC_DECLARE_FIELD(subsample)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("Row subsample ratio of training instance.");
    DMLC_DECLARE_FIELD(sampling_method)
        .set_default(kUniform)
        .add_enum("uniform", kUniform)
        .add_enum("gradient_based", kGradientBased)
        .describe(
            "Sampling method. 0: select random training instances uniformly. "
            "1: select random training instances with higher probability when the "
            "gradient and hessian are larger. (cf. CatBoost)");
    DMLC_DECLARE_FIELD(colsample_bynode)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("Subsample ratio of columns, resample on each node (split).");
    DMLC_DECLARE_FIELD(colsample_bylevel)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("Subsample ratio of columns, resample on each level.");
    DMLC_DECLARE_FIELD(colsample_bytree)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("Subsample ratio of columns, resample on each tree construction.");
    DMLC_DECLARE_FIELD(sketch_eps)
        .set_range(0.0f, 1.0f)
        .set_default(0.03f)
        .describe("EXP Param: Sketch accuracy of approximate algorithm.");
    DMLC_DECLARE_FIELD(sketch_ratio)
        .set_lower_bound(0.0f)
        .set_default(2.0f)
        .describe("EXP Param: Sketch accuracy related parameter of approximate algorithm.");
    DMLC_DECLARE_FIELD(cache_opt)
        .set_default(true)
        .describe("EXP Param: Cache aware optimization.");
    DMLC_DECLARE_FIELD(refresh_leaf)
        .set_default(true)
        .describe("Whether the refresh updater needs to update leaf values.");
    DMLC_DECLARE_FIELD(monotone_constraints)
        .set_default(std::vector<int>())
        .describe("Constraint of variable monotonicity");
    DMLC_DECLARE_FIELD(interaction_constraints)
        .set_default("")
        .describe("Constraints for interaction representing permitted interactions."
                  "The constraints must be specified in the form of a nest list,"
                  "e.g. [[0, 1], [2, 3, 4]], where each inner list is a group of"
                  "indices of features that are allowed to interact with each other."
                  "See tutorial for more information");
    DMLC_DECLARE_FIELD(split_evaluator)
        .set_default("elastic_net,monotonic")
        .describe("The criteria to use for ranking splits");

    // ------ From cpu quantile histogram -------.
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

    // add alias of parameters
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
    DMLC_DECLARE_ALIAS(min_split_loss, gamma);
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }

  /*! \brief given the loss change, whether we need to invoke pruning */
  bool NeedPrune(double loss_chg, int depth) const {
    return loss_chg < this->min_split_loss ||
           (this->max_depth != 0 && depth > this->max_depth);
  }
  /*! \brief maximum sketch size */
  inline unsigned MaxSketchSize() const {
    auto ret = static_cast<unsigned>(sketch_ratio / sketch_eps);
    CHECK_GT(ret, 0U);
    return ret;
  }

  bst_node_t MaxNodes() const {
    if (this->max_depth == 0 && this->max_leaves == 0) {
      LOG(FATAL) << "Max leaves and max depth cannot both be unconstrained.";
    }
    bst_node_t n_nodes{0};
    if (this->max_leaves > 0) {
      n_nodes = this->max_leaves * 2 - 1;
    } else {
      n_nodes = (1 << (this->max_depth + 1)) - 1;
    }
    CHECK_NE(n_nodes, 0);
    return n_nodes;
  }
};

/*! \brief Loss functions */

// functions for L1 cost
template <typename T1, typename T2>
XGBOOST_DEVICE inline static T1 ThresholdL1(T1 w, T2 alpha) {
  if (w > + alpha) {
    return w - alpha;
  }
  if (w < - alpha) {
    return w + alpha;
  }
  return 0.0;
}

template <typename T>
XGBOOST_DEVICE inline static T Sqr(T a) { return a * a; }

// calculate the cost of loss function
template <typename TrainingParams, typename T>
XGBOOST_DEVICE inline T CalcGainGivenWeight(const TrainingParams &p,
                                            T sum_grad, T sum_hess, T w) {
  return -(T(2.0) * sum_grad * w + (sum_hess + p.reg_lambda) * Sqr(w));
}

// calculate the cost of loss function
template <typename TrainingParams, typename T>
XGBOOST_DEVICE inline T CalcGain(const TrainingParams &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight) {
    return T(0.0);
  }
  if (p.max_delta_step == 0.0f) {
    if (p.reg_alpha == 0.0f) {
      return Sqr(sum_grad) / (sum_hess + p.reg_lambda);
    } else {
      return Sqr(ThresholdL1(sum_grad, p.reg_alpha)) /
          (sum_hess + p.reg_lambda);
    }
  } else {
    T w = CalcWeight(p, sum_grad, sum_hess);
    T ret = CalcGainGivenWeight(p, sum_grad, sum_hess, w);
    if (p.reg_alpha == 0.0f) {
      return ret;
    } else {
      return ret + p.reg_alpha * std::abs(w);
    }
  }
}

template <typename TrainingParams,
          typename StatT, typename T = decltype(StatT().GetHess())>
XGBOOST_DEVICE inline T CalcGain(const TrainingParams &p, StatT stat) {
  return CalcGain(p, stat.GetGrad(), stat.GetHess());
}

// calculate weight given the statistics
template <typename TrainingParams, typename T>
XGBOOST_DEVICE inline T CalcWeight(const TrainingParams &p, T sum_grad,
                                   T sum_hess) {
  if (sum_hess < p.min_child_weight || sum_hess <= 0.0) {
    return 0.0;
  }
  T dw;
  if (p.reg_alpha == 0.0f) {
    dw = -sum_grad / (sum_hess + p.reg_lambda);
  } else {
    dw = -ThresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda);
  }
  if (p.max_delta_step != 0.0f) {
    if (dw > p.max_delta_step) {
      dw = p.max_delta_step;
    }
    if (dw < -p.max_delta_step) {
      dw = -p.max_delta_step;
    }
  }
  return dw;
}

// Used in gpu code where GradientPair is used for gradient sum, not GradStats.
template <typename TrainingParams, typename GpairT>
XGBOOST_DEVICE inline float CalcWeight(const TrainingParams &p, GpairT sum_grad) {
  return CalcWeight(p, sum_grad.GetGrad(), sum_grad.GetHess());
}

/*! \brief core statistics used for tree construction */
struct XGBOOST_ALIGNAS(16) GradStats {
  using GradType = double;
  /*! \brief sum gradient statistics */
  GradType sum_grad { 0 };
  /*! \brief sum hessian statistics */
  GradType sum_hess { 0 };

 public:
  XGBOOST_DEVICE GradType GetGrad() const { return sum_grad; }
  XGBOOST_DEVICE GradType GetHess() const { return sum_hess; }

  friend std::ostream& operator<<(std::ostream& os, GradStats s) {
    os << s.GetGrad() << "/" << s.GetHess();
    return os;
  }

  XGBOOST_DEVICE GradStats() {
    static_assert(sizeof(GradStats) == 16,
                  "Size of GradStats is not 16 bytes.");
  }

  template <typename GpairT>
  XGBOOST_DEVICE explicit GradStats(const GpairT &sum)
      : sum_grad(sum.GetGrad()), sum_hess(sum.GetHess()) {}
  explicit GradStats(const GradType grad, const GradType hess)
      : sum_grad(grad), sum_hess(hess) {}
  /*!
   * \brief accumulate statistics
   * \param p the gradient pair
   */
  inline void Add(GradientPair p) { this->Add(p.GetGrad(), p.GetHess()); }

  /*! \brief add statistics to the data */
  inline void Add(const GradStats& b) {
    sum_grad += b.sum_grad;
    sum_hess += b.sum_hess;
  }
  /*! \brief same as add, reduce is used in All Reduce */
  inline static void Reduce(GradStats& a, const GradStats& b) { // NOLINT(*)
    a.Add(b);
  }
  /*! \brief set current value to a - b */
  inline void SetSubstract(const GradStats& a, const GradStats& b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }
  /*! \return whether the statistics is not used yet */
  inline bool Empty() const { return sum_hess == 0.0; }
  /*! \brief add statistics to the data */
  inline void Add(GradType grad, GradType hess) {
    sum_grad += grad;
    sum_hess += hess;
  }
};

/*!
 * \brief statistics that is helpful to store
 *   and represent a split solution for the tree
 */
template<typename GradientT>
struct SplitEntryContainer {
  /*! \brief loss change after split this node */
  bst_float loss_chg {0.0f};
  /*! \brief split index */
  bst_feature_t sindex{0};
  bst_float split_value{0.0f};

  GradientT left_sum;
  GradientT right_sum;

  SplitEntryContainer() = default;

  friend std::ostream& operator<<(std::ostream& os, SplitEntryContainer const& s) {
    os << "loss_chg: " << s.loss_chg << ", "
       << "split index: " << s.SplitIndex() << ", "
       << "split value: " << s.split_value << ", "
       << "left_sum: " << s.left_sum << ", "
       << "right_sum: " << s.right_sum;
    return os;
  }
  /*!\return feature index to split on */
  bst_feature_t SplitIndex() const { return sindex & ((1U << 31) - 1U); }
  /*!\return whether missing value goes to left branch */
  bool DefaultLeft() const { return (sindex >> 31) != 0; }
  /*!
   * \brief decides whether we can replace current entry with the given statistics
   *
   *   This function gives better priority to lower index when loss_chg == new_loss_chg.
   *   Not the best way, but helps to give consistent result during multi-thread
   *   execution.
   *
   * \param new_loss_chg the loss reduction get through the split
   * \param split_index the feature index where the split is on
   */
  bool NeedReplace(bst_float new_loss_chg, unsigned split_index) const {
    if (std::isinf(new_loss_chg)) {  // in some cases new_loss_chg can be NaN or Inf,
                                         // for example when lambda = 0 & min_child_weight = 0
                                         // skip value in this case
      return false;
    } else if (this->SplitIndex() <= split_index) {
      return new_loss_chg > this->loss_chg;
    } else {
      return !(this->loss_chg > new_loss_chg);
    }
  }
  /*!
   * \brief update the split entry, replace it if e is better
   * \param e candidate split solution
   * \return whether the proposed split is better and can replace current split
   */
  inline bool Update(const SplitEntryContainer &e) {
    if (this->NeedReplace(e.loss_chg, e.SplitIndex())) {
      this->loss_chg = e.loss_chg;
      this->sindex = e.sindex;
      this->split_value = e.split_value;
      this->left_sum = e.left_sum;
      this->right_sum = e.right_sum;
      return true;
    } else {
      return false;
    }
  }
  /*!
   * \brief update the split entry, replace it if e is better
   * \param new_loss_chg loss reduction of new candidate
   * \param split_index feature index to split on
   * \param new_split_value the split point
   * \param default_left whether the missing value goes to left
   * \return whether the proposed split is better and can replace current split
   */
  bool Update(bst_float new_loss_chg, unsigned split_index,
              bst_float new_split_value, bool default_left,
              const GradientT &left_sum,
              const GradientT &right_sum) {
    if (this->NeedReplace(new_loss_chg, split_index)) {
      this->loss_chg = new_loss_chg;
      if (default_left) {
        split_index |= (1U << 31);
      }
      this->sindex = split_index;
      this->split_value = new_split_value;
      this->left_sum = left_sum;
      this->right_sum = right_sum;
      return true;
    } else {
      return false;
    }
  }

  /*! \brief same as update, used by AllReduce*/
  inline static void Reduce(SplitEntryContainer &dst,         // NOLINT(*)
                            const SplitEntryContainer &src) { // NOLINT(*)
    dst.Update(src);
  }
};

using SplitEntry = SplitEntryContainer<GradStats>;
}  // namespace tree

/*
 * \brief Parse the interaction constraints from string.
 * \param constraint_str String storing the interfaction constraints:
 *
 *  Example input string:
 *
 *    "[[1, 2], [3, 4]]""
 *
 * \param p_out Pointer to output
 */
void ParseInteractionConstraint(
    std::string const &constraint_str,
    std::vector<std::vector<xgboost::bst_feature_t>> *p_out);
}  // namespace xgboost

// define string serializer for vector, to get the arguments
namespace std {
inline std::ostream &operator<<(std::ostream &os, const std::vector<int> &t) {
  os << '(';
  for (auto it = t.begin(); it != t.end(); ++it) {
    if (it != t.begin()) {
      os << ',';
    }
    os << *it;
  }
  // python style tuple
  if (t.size() == 1) {
    os << ',';
  }
  os << ')';
  return os;
}

std::istream &operator>>(std::istream &is, std::vector<int> &t);
}  // namespace std

#endif  // XGBOOST_TREE_PARAM_H_
