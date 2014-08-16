#ifndef XGBOOST_TREE_PARAM_H_
#define XGBOOST_TREE_PARAM_H_
/*!
 * \file param.h
 * \brief training parameters, statistics used to support tree construction
 * \author Tianqi Chen
 */
#include <cstring>
#include "../data.h"

namespace xgboost {
namespace tree {

/*! \brief core statistics used for tree construction */
struct GradStats {
  /*! \brief sum gradient statistics */
  double sum_grad;
  /*! \brief sum hessian statistics */
  double sum_hess;
  /*! \brief constructor */
  GradStats(void) {
    this->Clear();
  }
  /*! \brief clear the statistics */
  inline void Clear(void) {
    sum_grad = sum_hess = 0.0f;
  }
  /*! \brief add statistics to the data */
  inline void Add(double grad, double hess) {
    sum_grad += grad; sum_hess += hess;
  }
  /*! \brief add statistics to the data */
  inline void Add(const bst_gpair& b) {
    this->Add(b.grad, b.hess);
  }
  /*! \brief add statistics to the data */
  inline void Add(const GradStats &b) {
    this->Add(b.sum_grad, b.sum_hess);
  }
  /*! \brief substract the statistics by b */
  inline GradStats Substract(const GradStats &b) const {
    GradStats res;
    res.sum_grad = this->sum_grad - b.sum_grad;
    res.sum_hess = this->sum_hess - b.sum_hess;
    return res;
  }
  /*! \return whether the statistics is not used yet */
  inline bool Empty(void) const {
    return sum_hess == 0.0;
  }
};

/*! \brief training parameters for regression tree */
struct TrainParam{
  // learning step size for a time
  float learning_rate;
  // minimum loss change required for a split
  float min_split_loss;
  // maximum depth of a tree
  int max_depth;
  //----- the rest parameters are less important ----
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // weight decay parameter used to control leaf fitting
  float reg_lambda;
  // reg method
  int reg_method;
  // default direction choice
  int default_direction;
  // whether we want to do subsample
  float subsample;
  // whether to subsample columns each split, in each level
  float colsample_bylevel;
  // whether to subsample columns during tree construction
  float colsample_bytree;
  // speed optimization for dense column
  float opt_dense_col;
  // number of threads to be used for tree construction,
  // if OpenMP is enabled, if equals 0, use system default
  int nthread;
  /*! \brief constructor */
  TrainParam(void) {
    learning_rate = 0.3f;
    min_child_weight = 1.0f;
    max_depth = 6;
    reg_lambda = 1.0f;
    reg_method = 2;
    default_direction = 0;
    subsample = 1.0f;
    colsample_bytree = 1.0f;
    colsample_bylevel = 1.0f;
    opt_dense_col = 1.0f;
    nthread = 0;
  }
  /*! 
   * \brief set parameters from outside 
   * \param name name of the parameter
   * \param val  value of the parameter
   */            
  inline void SetParam(const char *name, const char *val) {
    // sync-names
    if (!strcmp(name, "gamma")) min_split_loss = static_cast<float>(atof(val));
    if (!strcmp(name, "eta")) learning_rate = static_cast<float>(atof(val));
    if (!strcmp(name, "lambda")) reg_lambda = static_cast<float>(atof(val));
    if (!strcmp(name, "learning_rate")) learning_rate = static_cast<float>(atof(val));
    if (!strcmp(name, "min_child_weight")) min_child_weight = static_cast<float>(atof(val));
    if (!strcmp(name, "min_split_loss")) min_split_loss = static_cast<float>(atof(val));
    if (!strcmp(name, "reg_lambda")) reg_lambda = static_cast<float>(atof(val));
    if (!strcmp(name, "reg_method")) reg_method = static_cast<float>(atof(val));
    if (!strcmp(name, "subsample")) subsample = static_cast<float>(atof(val));
    if (!strcmp(name, "colsample_bylevel")) colsample_bylevel = static_cast<float>(atof(val));
    if (!strcmp(name, "colsample_bytree")) colsample_bytree  = static_cast<float>(atof(val));
    if (!strcmp(name, "opt_dense_col")) opt_dense_col = static_cast<float>(atof(val));
    if (!strcmp(name, "max_depth")) max_depth = atoi(val);
    if (!strcmp(name, "nthread")) nthread = atoi(val);
    if (!strcmp(name, "default_direction")) {
      if (!strcmp(val, "learn")) default_direction = 0;
      if (!strcmp(val, "left")) default_direction = 1;
      if (!strcmp(val, "right")) default_direction = 2;
    }
  }
  // calculate the cost of loss function
  inline double CalcGain(double sum_grad, double sum_hess) const {
    if (sum_hess < min_child_weight) {
      return 0.0;
    }
    switch (reg_method) {
      case 1 : return Sqr(ThresholdL1(sum_grad, reg_lambda)) / sum_hess;
      case 2 : return Sqr(sum_grad) / (sum_hess + reg_lambda);
      case 3 : return
          Sqr(ThresholdL1(sum_grad, 0.5 * reg_lambda)) /
          (sum_hess + 0.5 * reg_lambda);
      default: return Sqr(sum_grad) / sum_hess;
    }
  }
  // calculate weight given the statistics
  inline double CalcWeight(double sum_grad, double sum_hess) const {
    if (sum_hess < min_child_weight) {
      return 0.0;
    } else {
      switch (reg_method) {
        case 1: return - ThresholdL1(sum_grad, reg_lambda) / sum_hess;
        case 2: return - sum_grad / (sum_hess + reg_lambda);
        case 3: return
            - ThresholdL1(sum_grad, 0.5 * reg_lambda) /
            (sum_hess + 0.5 * reg_lambda);
        default: return - sum_grad / sum_hess;
      }
    }
  }
  /*! \brief whether need forward small to big search: default right */
  inline bool need_forward_search(float col_density = 0.0f) const {
    return this->default_direction == 2 ||
        (default_direction == 0 && (col_density < opt_dense_col));
  }
  /*! \brief whether need backward big to small search: default left */
  inline bool need_backward_search(float col_density = 0.0f) const {
    return this->default_direction != 2;
  }
  /*! \brief given the loss change, whether we need to invode prunning */
  inline bool need_prune(double loss_chg, int depth) const {
    return loss_chg < this->min_split_loss;
  }
  /*! \brief whether we can split with current hessian */
  inline bool cannot_split(double sum_hess, int depth) const {
    return sum_hess < this->min_child_weight * 2.0;
  }
  // code support for template data
  inline double CalcWeight(const GradStats &d) const {
    return this->CalcWeight(d.sum_grad, d.sum_hess);
  }
  inline double CalcGain(const GradStats &d) const {
    return this->CalcGain(d.sum_grad, d.sum_hess);
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

/*! 
 * \brief statistics that is helpful to store 
 *   and represent a split solution for the tree
 */
struct SplitEntry{
  /*! \brief loss change after split this node */
  bst_float loss_chg;
  /*! \brief split index */
  unsigned sindex;
  /*! \brief split value */
  float split_value;
  /*! \brief constructor */
  SplitEntry(void) : loss_chg(0.0f), sindex(0), split_value(0.0f) {}
  /*! 
   * \brief decides whether a we can replace current entry with the statistics given 
   *   This function gives better priority to lower index when loss_chg equals
   *    not the best way, but helps to give consistent result during multi-thread execution
   * \param loss_chg the loss reduction get through the split
   * \param split_index the feature index where the split is on 
   */
  inline bool NeedReplace(bst_float loss_chg, unsigned split_index) const {
    if (this->split_index() <= split_index) {
      return loss_chg > this->loss_chg;
    } else {
      return !(this->loss_chg > loss_chg);
    }
  }
  /*! 
   * \brief update the split entry, replace it if e is better
   * \param e candidate split solution
   * \return whether the proposed split is better and can replace current split
   */
  inline bool Update(const SplitEntry &e) {
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
   * \param loss_chg loss reduction of new candidate
   * \param split_index feature index to split on
   * \param split_value the split point
   * \param default_left whether the missing value goes to left
   * \return whether the proposed split is better and can replace current split
   */
  inline bool Update(bst_float loss_chg, unsigned split_index,
                     float split_value, bool default_left) {
    if (this->NeedReplace(loss_chg, split_index)) {
      this->loss_chg = loss_chg;
      if (default_left) split_index |= (1U << 31);
      this->sindex = split_index;
      this->split_value = split_value;
      return true;
    } else {
      return false;
    }
  }
  /*!\return feature index to split on */
  inline unsigned split_index(void) const {
    return sindex & ((1U << 31) - 1U);
  }
  /*!\return whether missing value goes to left branch */
  inline bool default_left(void) const {
    return (sindex >> 31) != 0;
  }
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_PARAM_H_
