/*!
 * Copyright 2015-2021 by Contributors
 * \file learner.h
 * \brief Learner interface that integrates objective, gbm and evaluation together.
 *  This is the user facing XGBoost training module.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_LEARNER_H_
#define XGBOOST_LEARNER_H_

#include <dmlc/any.h>
#include <xgboost/base.h>
#include <xgboost/feature_map.h>
#include <xgboost/predictor.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/model.h>

#include <utility>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace xgboost {

class Metric;
class GradientBooster;
class ObjFunction;
class DMatrix;
class Json;

enum class PredictionType : std::uint8_t {  // NOLINT
  kValue = 0,
  kMargin = 1,
  kContribution = 2,
  kApproxContribution = 3,
  kInteraction = 4,
  kLeaf = 5
};

/*! \brief entry to to easily hold returning information */
struct XGBAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief returning float vector. */
  std::vector<bst_float> ret_vec_float;
  /*! \brief temp variable of gradient pairs. */
  std::vector<GradientPair> tmp_gpair;
  /*! \brief Temp variable for returing prediction result. */
  PredictionCacheEntry prediction_entry;
  /*! \brief Temp variable for returing prediction shape. */
  std::vector<bst_ulong> prediction_shape;
};

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
class Learner : public Model, public Configurable, public dmlc::Serializable {
 public:
  /*! \brief virtual destructor */
  ~Learner() override;
  /*!
   * \brief Configure Learner based on set parameters.
   */
  virtual void Configure() = 0;
  /*!
   * \brief update the model for one iteration
   *  With the specified objective function.
   * \param iter current iteration number
   * \param train reference to the data matrix.
   */
  virtual void UpdateOneIter(int iter, std::shared_ptr<DMatrix> train) = 0;
  /*!
   * \brief Do customized gradient boosting with in_gpair.
   *  in_gair can be mutated after this call.
   * \param iter current iteration number
   * \param train reference to the data matrix.
   * \param in_gpair The input gradient statistics.
   */
  virtual void BoostOneIter(int iter,
                            std::shared_ptr<DMatrix> train,
                            HostDeviceVector<GradientPair>* in_gpair) = 0;
  /*!
   * \brief evaluate the model for specific iteration using the configured metrics.
   * \param iter iteration number
   * \param data_sets datasets to be evaluated.
   * \param data_names name of each dataset
   * \return a string corresponding to the evaluation result
   */
  virtual std::string EvalOneIter(int iter,
                                  const std::vector<std::shared_ptr<DMatrix>>& data_sets,
                                  const std::vector<std::string>& data_names) = 0;
  /*!
   * \brief get prediction given the model.
   * \param data input data
   * \param output_margin whether to only predict margin value instead of transformed prediction
   * \param out_preds output vector that stores the prediction
   * \param layer_begin Beginning of boosted tree layer used for prediction.
   * \param layer_end   End of booster layer. 0 means do not limit trees.
   * \param training Whether the prediction result is used for training
   * \param pred_leaf whether to only predict the leaf index of each tree in a boosted tree predictor
   * \param pred_contribs whether to only predict the feature contributions
   * \param approx_contribs whether to approximate the feature contributions for speed
   * \param pred_interactions whether to compute the feature pair contributions
   */
  virtual void Predict(std::shared_ptr<DMatrix> data,
                       bool output_margin,
                       HostDeviceVector<bst_float> *out_preds,
                       unsigned layer_begin,
                       unsigned layer_end,
                       bool training = false,
                       bool pred_leaf = false,
                       bool pred_contribs = false,
                       bool approx_contribs = false,
                       bool pred_interactions = false) = 0;

  /*!
   * \brief Inplace prediction.
   *
   * \param          x           A type erased data adapter.
   * \param          p_m         An optional Proxy DMatrix object storing meta info like
   *                             base margin.  Can be nullptr.
   * \param          type        Prediction type.
   * \param          missing     Missing value in the data.
   * \param [in,out] out_preds   Pointer to output prediction vector.
   * \param          layer_begin Beginning of boosted tree layer used for prediction.
   * \param          layer_end   End of booster layer. 0 means do not limit trees.
   */
  virtual void InplacePredict(dmlc::any const &x,
                              std::shared_ptr<DMatrix> p_m,
                              PredictionType type,
                              float missing,
                              HostDeviceVector<bst_float> **out_preds,
                              uint32_t layer_begin, uint32_t layer_end) = 0;

  /*
   * \brief Get number of boosted rounds from gradient booster.
   */
  virtual int32_t BoostedRounds() const = 0;
  virtual uint32_t Groups() const = 0;

  void LoadModel(Json const& in) override = 0;
  void SaveModel(Json* out) const override = 0;

  virtual void LoadModel(dmlc::Stream* fi) = 0;
  virtual void SaveModel(dmlc::Stream* fo) const = 0;

  /*!
   * \brief Set multiple parameters at once.
   *
   * \param args parameters.
   */
  virtual void SetParams(Args const& args) = 0;
  /*!
   * \brief Set parameter for booster
   *
   *  The property will NOT be saved along with booster
   *
   * \param key   The key of parameter
   * \param value The value of parameter
   */
  virtual void SetParam(const std::string& key, const std::string& value) = 0;

  /*!
   * \brief Get the number of features of the booster.
   * \return number of features
   */
  virtual uint32_t GetNumFeature() const = 0;

  /*!
   * \brief Set additional attribute to the Booster.
   *
   *  The property will be saved along the booster.
   *
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
   * \brief Slice the model.
   *
   * See InplacePredict for layer parameters.
   *
   * \param step step size between slice.
   * \param out_of_bound Return true if end layer is out of bound.
   *
   * \return a sliced model.
   */
  virtual Learner *Slice(int32_t begin_layer, int32_t end_layer, int32_t step,
                         bool *out_of_bound) = 0;
  /*!
   * \brief dump the model in the requested format
   * \param fmap feature map that may help give interpretations of feature
   * \param with_stats extra statistics while dumping model
   * \param format the format to dump the model in
   * \return a vector of dump for boosters.
   */
  virtual std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                             bool with_stats,
                                             std::string format) = 0;

  virtual XGBAPIThreadLocalEntry& GetThreadLocal() const = 0;
  /*!
   * \brief Create a new instance of learner.
   * \param cache_data The matrix to cache the prediction.
   * \return Created learner.
   */
  static Learner* Create(const std::vector<std::shared_ptr<DMatrix> >& cache_data);

  virtual GenericParameter const& GetGenericParameter() const = 0;
  /*!
   * \brief Get configuration arguments currently stored by the learner
   * \return Key-value pairs representing configuration arguments
   */
  virtual const std::map<std::string, std::string>& GetConfigurationArguments() const = 0;

 protected:
  /*! \brief objective function */
  std::unique_ptr<ObjFunction> obj_;
  /*! \brief The gradient booster used by the model*/
  std::unique_ptr<GradientBooster> gbm_;
  /*! \brief The evaluation metrics used to evaluate the model. */
  std::vector<std::unique_ptr<Metric> > metrics_;
  /*! \brief Training parameter. */
  GenericParameter generic_parameters_;
};

struct LearnerModelParamLegacy;

/*
 * \brief Basic Model Parameters, used to describe the booster.
 */
struct LearnerModelParam {
  /* \brief global bias */
  bst_float base_score { 0.5f };
  /* \brief number of features  */
  uint32_t num_feature { 0 };
  /* \brief number of classes, if it is multi-class classification  */
  uint32_t num_output_group { 0 };

  LearnerModelParam() = default;
  // As the old `LearnerModelParamLegacy` is still used by binary IO, we keep
  // this one as an immutable copy.
  LearnerModelParam(LearnerModelParamLegacy const& user_param, float base_margin);
  /* \brief Whether this parameter is initialized with LearnerModelParamLegacy. */
  bool Initialized() const { return num_feature != 0; }
};

}  // namespace xgboost
#endif  // XGBOOST_LEARNER_H_
