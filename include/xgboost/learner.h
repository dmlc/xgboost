/*!
 * Copyright 2015 by Contributors
 * \file learner.h
 * \brief Learner interface that integrates objective, gbm and evaluation together.
 *  This is the user facing XGBoost training module.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_LEARNER_H_
#define XGBOOST_LEARNER_H_

#include <rabit/rabit.h>
#include <utility>
#include <map>
#include <string>
#include <vector>
#include "./base.h"
#include "./gbm.h"
#include "./metric.h"
#include "./objective.h"

namespace xgboost {
/*!
 * \brief Learner class that does training and prediction.
 *  This is the user facing module of xgboost training.
 *  The Load/Save function corresponds to the model used in python/R.
 *  \code
 *
 *  std::unique_ptr<Learner> learner(new Learner::Create(cache_mats));
 *  learner.Configure(configs);
 *
 *  for (int iter = 0; iter < max_iter; ++iter) {
 *    learner->UpdateOneIter(iter, train_mat);
 *    LOG(INFO) << learner->EvalOneIter(iter, data_sets, data_names);
 *  }
 *
 *  \endcode
 */
class Learner : public rabit::Serializable {
 public:
  /*! \brief virtual destructor */
  ~Learner() override = default;
  /*!
   * \brief set configuration from pair iterators.
   * \param begin The beginning iterator.
   * \param end The end iterator.
   * \tparam PairIter iterator<std::pair<std::string, std::string> >
   */
  template<typename PairIter>
  inline void Configure(PairIter begin, PairIter end);
  /*!
   * \brief Set the configuration of gradient boosting.
   *  User must call configure once before InitModel and Training.
   *
   * \param cfg configurations on both training and model parameters.
   */
  virtual void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) = 0;
  /*!
   * \brief Initialize the model using the specified configurations via Configure.
   *  An model have to be either Loaded or initialized before Update/Predict/Save can be called.
   */
  virtual void InitModel() = 0;
  /*!
   * \brief load model from stream
   * \param fi input stream.
   */
  void Load(dmlc::Stream* fi) override = 0;
  /*!
   * \brief save model to stream.
   * \param fo output stream
   */
  void Save(dmlc::Stream* fo) const override = 0;
  /*!
   * \brief update the model for one iteration
   *  With the specified objective function.
   * \param iter current iteration number
   * \param train reference to the data matrix.
   */
  virtual void UpdateOneIter(int iter, DMatrix* train) = 0;
  /*!
   * \brief Do customized gradient boosting with in_gpair.
   *  in_gair can be mutated after this call.
   * \param iter current iteration number
   * \param train reference to the data matrix.
   * \param in_gpair The input gradient statistics.
   */
  virtual void BoostOneIter(int iter,
                            DMatrix* train,
                            HostDeviceVector<GradientPair>* in_gpair) = 0;
  /*!
   * \brief evaluate the model for specific iteration using the configured metrics.
   * \param iter iteration number
   * \param data_sets datasets to be evaluated.
   * \param data_names name of each dataset
   * \return a string corresponding to the evaluation result
   */
  virtual std::string EvalOneIter(int iter,
                                  const std::vector<DMatrix*>& data_sets,
                                  const std::vector<std::string>& data_names) = 0;
  /*!
   * \brief get prediction given the model.
   * \param data input data
   * \param output_margin whether to only predict margin value instead of transformed prediction
   * \param out_preds output vector that stores the prediction
   * \param ntree_limit limit number of trees used for boosted tree
   *   predictor, when it equals 0, this means we are using all the trees
   * \param pred_leaf whether to only predict the leaf index of each tree in a boosted tree predictor
   * \param pred_contribs whether to only predict the feature contributions
   * \param approx_contribs whether to approximate the feature contributions for speed
   * \param pred_interactions whether to compute the feature pair contributions
   */
  virtual void Predict(DMatrix* data,
                       bool output_margin,
                       HostDeviceVector<bst_float> *out_preds,
                       unsigned ntree_limit = 0,
                       bool pred_leaf = false,
                       bool pred_contribs = false,
                       bool approx_contribs = false,
                       bool pred_interactions = false) const = 0;

  /*!
   * \brief Set additional attribute to the Booster.
   *  The property will be saved along the booster.
   * \param key The key of the property.
   * \param value The value of the property.
   */
  virtual void SetAttr(const std::string& key, const std::string& value) = 0;
  /*!
   * \brief Get attribute from the booster.
   *  The property will be saved along the booster.
   * \param key The key of the attribute.
   * \param out The output value.
   * \return Whether the key exists among booster's attributes.
   */
  virtual bool GetAttr(const std::string& key, std::string* out) const = 0;
  /*!
   * \brief Delete an attribute from the booster.
   * \param key The key of the attribute.
   * \return Whether the key was found among booster's attributes.
   */
  virtual bool DelAttr(const std::string& key) = 0;
  /*!
   * \brief Get a vector of attribute names from the booster.
   * \return vector of attribute name strings.
   */
  virtual std::vector<std::string> GetAttrNames() const = 0;
  /*!
   * \return whether the model allow lazy checkpoint in rabit.
   */
  bool AllowLazyCheckPoint() const;
  /*!
   * \brief dump the model in the requested format
   * \param fmap feature map that may help give interpretations of feature
   * \param with_stats extra statistics while dumping model
   * \param format the format to dump the model in
   * \return a vector of dump for boosters.
   */
  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const;
  /*!
   * \brief online prediction function, predict score for one instance at a time
   *  NOTE: use the batch prediction interface if possible, batch prediction is usually
   *        more efficient than online prediction
   *        This function is NOT threadsafe, make sure you only call from one thread.
   *
   * \param inst the instance you want to predict
   * \param output_margin whether to only predict margin value instead of transformed prediction
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction
   */
  inline void Predict(const SparsePage::Inst &inst,
                      bool output_margin,
                      HostDeviceVector<bst_float> *out_preds,
                      unsigned ntree_limit = 0) const;
  /*!
   * \brief Create a new instance of learner.
   * \param cache_data The matrix to cache the prediction.
   * \return Created learner.
   */
  static Learner* Create(const std::vector<std::shared_ptr<DMatrix> >& cache_data);

  /*!
   * \brief Get configuration arguments currently stored by the learner
   * \return Key-value pairs representing configuration arguments
   */
  virtual const std::map<std::string, std::string>& GetConfigurationArguments() const = 0;

 protected:
  /*! \brief internal base score of the model */
  bst_float base_score_;
  /*! \brief objective function */
  std::unique_ptr<ObjFunction> obj_;
  /*! \brief The gradient booster used by the model*/
  std::unique_ptr<GradientBooster> gbm_;
  /*! \brief The evaluation metrics used to evaluate the model. */
  std::vector<std::unique_ptr<Metric> > metrics_;
};

// implementation of inline functions.
inline void Learner::Predict(const SparsePage::Inst& inst,
                             bool output_margin,
                             HostDeviceVector<bst_float>* out_preds,
                             unsigned ntree_limit) const {
  gbm_->PredictInstance(inst, &out_preds->HostVector(), ntree_limit);
  if (!output_margin) {
    obj_->PredTransform(out_preds);
  }
}

// implementing configure.
template<typename PairIter>
inline void Learner::Configure(PairIter begin, PairIter end) {
  std::vector<std::pair<std::string, std::string> > vec(begin, end);
  this->Configure(vec);
}

}  // namespace xgboost
#endif  // XGBOOST_LEARNER_H_
