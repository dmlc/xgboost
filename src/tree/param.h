/*!
 * Copyright 2014 by Contributors
 * \file param.h
 * \brief training parameters, statistics used to support tree construction.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_PARAM_H_
#define XGBOOST_TREE_PARAM_H_

#include <vector>
#include <cstring>

namespace xgboost {
namespace tree {

/*! \brief training parameters for regression tree */
struct TrainParam : public dmlc::Parameter<TrainParam> {
  // learning step size for a time
  float learning_rate;
  // minimum loss change required for a split
  float min_split_loss;
  // maximum depth of a tree
  int max_depth;
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
  // whether to subsample columns each split, in each level
  float colsample_bylevel;
  // whether to subsample columns during tree construction
  float colsample_bytree;
  // speed optimization for dense column
  float opt_dense_col;
  // accuracy of sketch
  float sketch_eps;
  // accuracy of sketch
  float sketch_ratio;
  // leaf vector size
  int size_leaf_vector;
  // option for parallelization
  int parallel_option;
  // option to open cacheline optimization
  bool cache_opt;
  // number of threads to be used for tree construction,
  // if OpenMP is enabled, if equals 0, use system default
  int nthread;
  // whether to not print info during training.
  bool silent;
  // declare the parameters
  DMLC_DECLARE_PARAMETER(TrainParam) {
    DMLC_DECLARE_FIELD(learning_rate).set_lower_bound(0.0f).set_default(0.3f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_FIELD(min_split_loss).set_lower_bound(0.0f).set_default(0.0f)
        .describe("Minimum loss reduction required to make a further partition.");
    DMLC_DECLARE_FIELD(max_depth).set_lower_bound(0).set_default(6)
        .describe("Maximum depth of the tree.");
    DMLC_DECLARE_FIELD(min_child_weight).set_lower_bound(0.0f).set_default(1.0f)
        .describe("Minimum sum of instance weight(hessian) needed in a child.");
    DMLC_DECLARE_FIELD(reg_lambda).set_lower_bound(0.0f).set_default(1.0f)
        .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_alpha).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L1 regularization on leaf weight");
    DMLC_DECLARE_FIELD(default_direction).set_default(0)
        .add_enum("learn", 0)
        .add_enum("left", 1)
        .add_enum("right", 2)
        .describe("Default direction choice when encountering a missing value");
    DMLC_DECLARE_FIELD(max_delta_step).set_lower_bound(0.0f).set_default(0.0f)
        .describe("Maximum delta step we allow each tree's weight estimate to be. "\
                  "If the value is set to 0, it means there is no constraint");
    DMLC_DECLARE_FIELD(subsample).set_range(0.0f, 1.0f).set_default(1.0f)
        .describe("Row subsample ratio of training instance.");
    DMLC_DECLARE_FIELD(colsample_bylevel).set_range(0.0f, 1.0f).set_default(1.0f)
        .describe("Subsample ratio of columns, resample on each level.");
    DMLC_DECLARE_FIELD(colsample_bytree).set_range(0.0f, 1.0f).set_default(1.0f)
        .describe("Subsample ratio of columns, resample on each tree construction.");
    DMLC_DECLARE_FIELD(opt_dense_col).set_range(0.0f, 1.0f).set_default(1.0f)
        .describe("EXP Param: speed optimization for dense column.");
    DMLC_DECLARE_FIELD(sketch_eps).set_range(0.0f, 1.0f).set_default(0.1f)
        .describe("EXP Param: Sketch accuracy of approximate algorithm.");
    DMLC_DECLARE_FIELD(sketch_ratio).set_lower_bound(0.0f).set_default(2.0f)
        .describe("EXP Param: Sketch accuracy related parameter of approximate algorithm.");
    DMLC_DECLARE_FIELD(size_leaf_vector).set_lower_bound(0).set_default(0)
        .describe("Size of leaf vectors, reserved for vector trees");
    DMLC_DECLARE_FIELD(parallel_option).set_default(0)
        .describe("Different types of parallelization algorithm.");
    DMLC_DECLARE_FIELD(cache_opt).set_default(true)
        .describe("EXP Param: Cache aware optimization.");
    DMLC_DECLARE_FIELD(nthread).set_default(0)
        .describe("Number of threads used for training.");
    DMLC_DECLARE_FIELD(silent).set_default(false)
        .describe("Not print information during trainig.");
    // add alias of parameters
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
    DMLC_DECLARE_ALIAS(min_split_loss, gamma);
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }

  // calculate the cost of loss function
  inline double CalcGain(double sum_grad, double sum_hess) const {
    if (sum_hess < min_child_weight) return 0.0;
    if (max_delta_step == 0.0f) {
      if (reg_alpha == 0.0f) {
        return Sqr(sum_grad) / (sum_hess + reg_lambda);
      } else {
        return Sqr(ThresholdL1(sum_grad, reg_alpha)) / (sum_hess + reg_lambda);
      }
    } else {
      double w = CalcWeight(sum_grad, sum_hess);
      double ret = sum_grad * w + 0.5 * (sum_hess + reg_lambda) * Sqr(w);
      if (reg_alpha == 0.0f) {
        return - 2.0 * ret;
      } else {
        return - 2.0 * (ret + reg_alpha * std::abs(w));
      }
    }
  }
  // calculate cost of loss function with four statistics
  inline double CalcGain(double sum_grad, double sum_hess,
                         double test_grad, double test_hess) const {
    double w = CalcWeight(sum_grad, sum_hess);
    double ret = test_grad * w  + 0.5 * (test_hess + reg_lambda) * Sqr(w);
    if (reg_alpha == 0.0f) {
      return - 2.0 * ret;
    } else {
      return - 2.0 * (ret + reg_alpha * std::abs(w));
    }
  }
  // calculate weight given the statistics
  inline double CalcWeight(double sum_grad, double sum_hess) const {
    if (sum_hess < min_child_weight) return 0.0;
    double dw;
    if (reg_alpha == 0.0f) {
      dw = -sum_grad / (sum_hess + reg_lambda);
    } else {
      dw = -ThresholdL1(sum_grad, reg_alpha) / (sum_hess + reg_lambda);
    }
    if (max_delta_step != 0.0f) {
      if (dw > max_delta_step) dw = max_delta_step;
      if (dw < -max_delta_step) dw = -max_delta_step;
    }
    return dw;
  }
  /*! \brief whether need forward small to big search: default right */
  inline bool need_forward_search(float col_density, bool indicator) const {
    return this->default_direction == 2 ||
        (default_direction == 0 && (col_density < opt_dense_col) && !indicator);
  }
  /*! \brief whether need backward big to small search: default left */
  inline bool need_backward_search(float col_density, bool indicator) const {
    return this->default_direction != 2;
  }
  /*! \brief given the loss change, whether we need to invoke pruning */
  inline bool need_prune(double loss_chg, int depth) const {
    return loss_chg < this->min_split_loss;
  }
  /*! \brief whether we can split with current hessian */
  inline bool cannot_split(double sum_hess, int depth) const {
    return sum_hess < this->min_child_weight * 2.0;
  }
  /*! \brief maximum sketch size */
  inline unsigned max_sketch_size() const {
    unsigned ret = static_cast<unsigned>(sketch_ratio / sketch_eps);
    CHECK_GT(ret, 0);
    return ret;
  }

 protected:
  // functions for L1 cost
  inline static double ThresholdL1(double w, double lambda) {
    if (w > +lambda) return w - lambda;
    if (w < -lambda) return w + lambda;
    return 0.0;
  }
  inline static double Sqr(double a) {
    return a * a;
  }
};

/*! \brief core statistics used for tree construction */
struct GradStats {
  /*! \brief sum gradient statistics */
  double sum_grad;
  /*! \brief sum hessian statistics */
  double sum_hess;
  /*!
   * \brief whether this is simply statistics and we only need to call
   *   Add(gpair), instead of Add(gpair, info, ridx)
   */
  static const int kSimpleStats = 1;
  /*! \brief constructor, the object must be cleared during construction */
  explicit GradStats(const TrainParam& param) {
    this->Clear();
  }
  /*! \brief clear the statistics */
  inline void Clear() {
    sum_grad = sum_hess = 0.0f;
  }
  /*! \brief check if necessary information is ready */
  inline static void CheckInfo(const MetaInfo& info) {
  }
  /*!
   * \brief accumulate statistics
   * \param p the gradient pair
   */
  inline void Add(bst_gpair p) {
    this->Add(p.grad, p.hess);
  }
  /*!
   * \brief accumulate statistics, more complicated version
   * \param gpair the vector storing the gradient statistics
   * \param info the additional information
   * \param ridx instance index of this instance
   */
  inline void Add(const std::vector<bst_gpair>& gpair,
                  const MetaInfo& info,
                  bst_uint ridx) {
    const bst_gpair& b = gpair[ridx];
    this->Add(b.grad, b.hess);
  }
  /*! \brief calculate leaf weight */
  inline double CalcWeight(const TrainParam& param) const {
    return param.CalcWeight(sum_grad, sum_hess);
  }
  /*! \brief calculate gain of the solution */
  inline double CalcGain(const TrainParam& param) const {
    return param.CalcGain(sum_grad, sum_hess);
  }
  /*! \brief add statistics to the data */
  inline void Add(const GradStats& b) {
    this->Add(b.sum_grad, b.sum_hess);
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
  inline bool Empty() const {
    return sum_hess == 0.0;
  }
  /*! \brief set leaf vector value based on statistics */
  inline void SetLeafVec(const TrainParam& param, bst_float *vec) const {
  }
  // constructor to allow inheritance
  GradStats() {}
  /*! \brief add statistics to the data */
  inline void Add(double grad, double hess) {
    sum_grad += grad; sum_hess += hess;
  }
};

/*!
 * \brief statistics that is helpful to store
 *   and represent a split solution for the tree
 */
struct SplitEntry {
  /*! \brief loss change after split this node */
  bst_float loss_chg;
  /*! \brief split index */
  unsigned sindex;
  /*! \brief split value */
  float split_value;
  /*! \brief constructor */
  SplitEntry() : loss_chg(0.0f), sindex(0), split_value(0.0f) {}
  /*!
   * \brief decides whether we can replace current entry with the given statistics
   *   This function gives better priority to lower index when loss_chg == new_loss_chg.
   *   Not the best way, but helps to give consistent result during multi-thread execution.
   * \param new_loss_chg the loss reduction get through the split
   * \param split_index the feature index where the split is on
   */
  inline bool NeedReplace(bst_float new_loss_chg, unsigned split_index) const {
    if (this->split_index() <= split_index) {
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
  inline bool Update(const SplitEntry& e) {
    if (this->NeedReplace(e.loss_chg, e.split_index())) {
      this->loss_chg = e.loss_chg;
      this->sindex = e.sindex;
      this->split_value = e.split_value;
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
  inline bool Update(bst_float new_loss_chg, unsigned split_index,
                     float new_split_value, bool default_left) {
    if (this->NeedReplace(new_loss_chg, split_index)) {
      this->loss_chg = new_loss_chg;
      if (default_left) split_index |= (1U << 31);
      this->sindex = split_index;
      this->split_value = new_split_value;
      return true;
    } else {
      return false;
    }
  }
  /*! \brief same as update, used by AllReduce*/
  inline static void Reduce(SplitEntry& dst, const SplitEntry& src) { // NOLINT(*)
    dst.Update(src);
  }
  /*!\return feature index to split on */
  inline unsigned split_index() const {
    return sindex & ((1U << 31) - 1U);
  }
  /*!\return whether missing value goes to left branch */
  inline bool default_left() const {
    return (sindex >> 31) != 0;
  }
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_PARAM_H_
