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
  // L2 regularization factor
  float reg_lambda;
  // L1 regularization factor
  float reg_alpha;
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
  // leaf vector size
  int size_leaf_vector;
  // number of threads to be used for tree construction,
  // if OpenMP is enabled, if equals 0, use system default
  int nthread;
  /*! \brief constructor */
  TrainParam(void) {
    learning_rate = 0.3f;
    min_child_weight = 1.0f;
    max_depth = 6;
    reg_lambda = 1.0f;
    reg_alpha = 0.0f;
    default_direction = 0;
    subsample = 1.0f;
    colsample_bytree = 1.0f;
    colsample_bylevel = 1.0f;
    opt_dense_col = 1.0f;
    nthread = 0;
    size_leaf_vector = 0;
  }
  /*! 
   * \brief set parameters from outside 
   * \param name name of the parameter
   * \param val  value of the parameter
   */            
  inline void SetParam(const char *name, const char *val) {
    using namespace std;
    // sync-names
    if (!strcmp(name, "gamma")) min_split_loss = static_cast<float>(atof(val));
    if (!strcmp(name, "eta")) learning_rate = static_cast<float>(atof(val));
    if (!strcmp(name, "lambda")) reg_lambda = static_cast<float>(atof(val));
    if (!strcmp(name, "alpha")) reg_alpha = static_cast<float>(atof(val));
    if (!strcmp(name, "learning_rate")) learning_rate = static_cast<float>(atof(val));
    if (!strcmp(name, "min_child_weight")) min_child_weight = static_cast<float>(atof(val));
    if (!strcmp(name, "min_split_loss")) min_split_loss = static_cast<float>(atof(val));
    if (!strcmp(name, "reg_lambda")) reg_lambda = static_cast<float>(atof(val));
    if (!strcmp(name, "reg_alpha")) reg_alpha = static_cast<float>(atof(val));
    if (!strcmp(name, "subsample")) subsample = static_cast<float>(atof(val));
    if (!strcmp(name, "colsample_bylevel")) colsample_bylevel = static_cast<float>(atof(val));
    if (!strcmp(name, "colsample_bytree")) colsample_bytree  = static_cast<float>(atof(val));
    if (!strcmp(name, "opt_dense_col")) opt_dense_col = static_cast<float>(atof(val));
    if (!strcmp(name, "size_leaf_vector")) size_leaf_vector = atoi(val);
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
    if (sum_hess < min_child_weight) return 0.0;
    if (reg_alpha == 0.0f) {
      return Sqr(sum_grad) / (sum_hess + reg_lambda);
    } else {
      return Sqr(ThresholdL1(sum_grad, reg_alpha)) / (sum_hess + reg_lambda); 
    }
  }
  // calculate cost of loss function with four stati
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
    if (reg_alpha == 0.0f) {
      return -sum_grad / (sum_hess + reg_lambda);
    } else {
      return -ThresholdL1(sum_grad, reg_alpha) / (sum_hess + reg_lambda);
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
  /*! \brief constructor, the object must be cleared during construction */
  explicit GradStats(const TrainParam &param) {
    this->Clear();
  }
  /*! \brief clear the statistics */
  inline void Clear(void) {
    sum_grad = sum_hess = 0.0f;
  }
  /*! \brief check if necessary information is ready */
  inline static void CheckInfo(const BoosterInfo &info) {
  }
  /*!
   * \brief accumulate statistics,
   * \param gpair the vector storing the gradient statistics
   * \param info the additional information 
   * \param ridx instance index of this instance
   */
  inline void Add(const std::vector<bst_gpair> &gpair,
                  const BoosterInfo &info,
                  bst_uint ridx) {
    const bst_gpair &b = gpair[ridx];
    this->Add(b.grad, b.hess);
  }
  /*! \brief caculate leaf weight */
  inline double CalcWeight(const TrainParam &param) const {
    return param.CalcWeight(sum_grad, sum_hess);
  }
  /*! \brief calculate gain of the solution */
  inline double CalcGain(const TrainParam &param) const {
    return param.CalcGain(sum_grad, sum_hess);
  }
  /*! \brief add statistics to the data */
  inline void Add(const GradStats &b) {
    this->Add(b.sum_grad, b.sum_hess);
  }
  /*! \brief set current value to a - b */
  inline void SetSubstract(const GradStats &a, const GradStats &b) {
    sum_grad = a.sum_grad - b.sum_grad;
    sum_hess = a.sum_hess - b.sum_hess;
  }
  /*! \return whether the statistics is not used yet */
  inline bool Empty(void) const {
    return sum_hess == 0.0;
  }
  /*! \brief set leaf vector value based on statistics */
  inline void SetLeafVec(const TrainParam &param, bst_float *vec) const{
  }  
  // constructor to allow inheritance
  GradStats(void) {}
  /*! \brief add statistics to the data */
  inline void Add(double grad, double hess) {
    sum_grad += grad; sum_hess += hess;
  }
};

/*! \brief vectorized cv statistics */
template<unsigned vsize>
struct CVGradStats : public GradStats {
  // additional statistics
  GradStats train[vsize], valid[vsize];
  // constructor
  explicit CVGradStats(const TrainParam &param) {
    utils::Check(param.size_leaf_vector == vsize,
                 "CVGradStats: vsize must match size_leaf_vector");
    this->Clear();
  }
  /*! \brief check if necessary information is ready */
  inline static void CheckInfo(const BoosterInfo &info) {
    utils::Check(info.fold_index.size() != 0,
                 "CVGradStats: require fold_index");
  }
  /*! \brief clear the statistics */
  inline void Clear(void) {
    GradStats::Clear();
    for (unsigned i = 0; i < vsize; ++i) {
      train[i].Clear(); valid[i].Clear();
    }
  }
  inline void Add(const std::vector<bst_gpair> &gpair,
                  const BoosterInfo &info,
                  bst_uint ridx) {
    GradStats::Add(gpair[ridx].grad, gpair[ridx].hess);
    const size_t step = info.fold_index.size();
    for (unsigned i = 0; i < vsize; ++i) {
      const bst_gpair &b = gpair[(i + 1) * step + ridx];
      if (info.fold_index[ridx] == i) {
        valid[i].Add(b.grad, b.hess);
      } else {
        train[i].Add(b.grad, b.hess);
      }
    }
  }
  /*! \brief calculate gain of the solution */
  inline double CalcGain(const TrainParam &param) const {
    double ret = 0.0;
    for (unsigned i = 0; i < vsize; ++i) {
      ret += param.CalcGain(train[i].sum_grad,
                            train[i].sum_hess,
                            vsize * valid[i].sum_grad,
                            vsize * valid[i].sum_hess);      
    }
    return ret / vsize;
  }
  /*! \brief add statistics to the data */
  inline void Add(const CVGradStats &b) {
    GradStats::Add(b);
    for (unsigned i = 0; i < vsize; ++i) {
      train[i].Add(b.train[i]);
      valid[i].Add(b.valid[i]);
    }
  }
  /*! \brief set current value to a - b */
  inline void SetSubstract(const CVGradStats &a, const CVGradStats &b) {
    GradStats::SetSubstract(a, b);
    for (int i = 0; i < vsize; ++i) {
      train[i].SetSubstract(a.train[i], b.train[i]);
      valid[i].SetSubstract(a.valid[i], b.valid[i]);
    }
  }
  /*! \brief set leaf vector value based on statistics */
  inline void SetLeafVec(const TrainParam &param, bst_float *vec) const{
    for (int i = 0; i < vsize; ++i) {
      vec[i] = param.learning_rate *
          param.CalcWeight(train[i].sum_grad, train[i].sum_hess);
    }
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
