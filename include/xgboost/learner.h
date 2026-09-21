/**
 * Copyright 2015-2026, XGBoost Contributors
 *
 * \brief Learner interface that integrates objective, gbm and evaluation together.
 *  This is the user facing XGBoost training module.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_LEARNER_H_
#define XGBOOST_LEARNER_H_

#include <dmlc/io.h>           // for Serializable
#include <xgboost/base.h>      // for bst_feature_t, bst_target_t, bst_float, Args, GradientPair, ..
#include <xgboost/context.h>   // for Context
#include <xgboost/gradient.h>  // for GradientContainer
#include <xgboost/linalg.h>    // for Vector, VectorView
#include <xgboost/metric.h>    // for Metric
#include <xgboost/model.h>     // for Configurable, Model
#include <xgboost/span.h>      // for Span
#include <xgboost/task.h>      // for ObjInfo

#include <algorithm>  // for max
#include <cstdint>    // for int32_t, uint32_t, uint8_t
#include <memory>     // for shared_ptr, unique_ptr
#include <string>     // for string
#include <utility>    // for move
#include <vector>     // for vector

namespace xgboost {
class FeatureMap;
class Metric;
class GradientBooster;
class ObjFunction;
class DMatrix;
class Json;
struct XGBAPIThreadLocalEntry;
template <typename T>
class HostDeviceVector;
class CatContainer;

enum class PredictionType : std::uint8_t {  // NOLINT
  kValue = 0,
  kMargin = 1,
  kContribution = 2,
  kApproxContribution = 3,
  kInteraction = 4,
  kApproxInteraction = 5,
  kLeaf = 6
};

/**
 * @brief Learner class that does training and prediction.
 *  This is the user facing module of xgboost training.
 *  The Load/Save function corresponds to the model used in python/R.
 *  @code
 *
 *  std::unique_ptr<Learner> learner{Learner::Create(cache_mats)};
 *  learner->Configure(configs);
 *
 *  for (int iter = 0; iter < max_iter; ++iter) {
 *    learner->UpdateOneIter(iter, train_mat);
 *    LOG(INFO) << learner->EvalOneIter(iter, data_sets, data_names);
 *  }
 *
 *  @endcode
 */
class Learner : public Model, public Configurable, public dmlc::Serializable {
 public:
  ~Learner() override;
  /*!
   * \brief Configure Learner with a parameter batch.
   * \param args Ordered parameter batch. Repeated parameters are preserved.
   */
  virtual void Configure(Args const& args = {}) = 0;
  /*!
   * \brief update the model for one iteration
   *  With the specified objective function.
   * \param iter current iteration number
   * \param train reference to the data matrix.
   */
  virtual void UpdateOneIter(std::int32_t iter, std::shared_ptr<DMatrix> train) = 0;
  /**
   * @brief Do customized gradient boosting with in_gpair.
   *
   * @note in_gpair can be mutated after this call.
   *
   * @param iter current iteration number
   * @param train reference to the data matrix.
   * @param in_gpair The input gradient statistics.
   */
  virtual void BoostOneIter(std::int32_t iter, std::shared_ptr<DMatrix> train,
                            GradientContainer* in_gpair) = 0;
  /*!
   * \brief evaluate the model for specific iteration using the configured metrics.
   * \param iter iteration number
   * \param data_sets datasets to be evaluated.
   * \param data_names name of each dataset
   * \return a string corresponding to the evaluation result
   */
  virtual std::string EvalOneIter(int iter, const std::vector<std::shared_ptr<DMatrix>>& data_sets,
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
   * @param strict_shape Used by predict leaf to validate whether it's actually possible
   *        to have uniform shape.
   */
  virtual void Predict(std::shared_ptr<DMatrix> data, bool output_margin,
                       HostDeviceVector<bst_float>* out_preds, bst_layer_t layer_begin,
                       bst_layer_t layer_end, bool training = false, bool pred_leaf = false,
                       bool pred_contribs = false, bool approx_contribs = false,
                       bool pred_interactions = false, bool strict_shape = false) = 0;

  /*!
   * \brief Inplace prediction.
   *
   * \param          p_fmat      A proxy DMatrix that contains the data and related meta info.
   * \param          type        Prediction type.
   * \param          missing     Missing value in the data.
   * \param [in,out] out_preds   Pointer to output prediction vector.
   * \param          layer_begin Beginning of boosted tree layer used for prediction.
   * \param          layer_end   End of booster layer. 0 means do not limit trees.
   */
  virtual void InplacePredict(std::shared_ptr<DMatrix> p_m, PredictionType type, float missing,
                              HostDeviceVector<float>** out_preds, bst_layer_t layer_begin,
                              bst_layer_t layer_end) = 0;

  /*!
   * \brief Calculate feature score.  See doc in C API for outputs.
   */
  virtual void CalcFeatureScore(std::string const& importance_type,
                                common::Span<int32_t const> trees,
                                std::vector<bst_feature_t>* features,
                                std::vector<float>* scores) = 0;

  /*
   * \brief Get number of boosted rounds from gradient booster.
   */
  virtual int32_t BoostedRounds() const = 0;
  /**
   * \brief Get the number of output groups from the model.
   */
  virtual std::uint32_t Groups() const = 0;

  void LoadModel(Json const& in) override = 0;
  void SaveModel(Json* out) const override = 0;

  /**
   * @brief Get the number of features of the booster.
   * @return The number of features
   */
  virtual bst_feature_t GetNumFeature() const = 0;

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
   * \brief Set the feature names for current booster.
   * \param fn Input feature names
   */
  virtual void SetFeatureNames(std::vector<std::string> const& fn) = 0;
  /*!
   * \brief Get the feature names for current booster.
   * \param fn Output feature names
   */
  virtual void GetFeatureNames(std::vector<std::string>* fn) const = 0;
  /*!
   * \brief Set the feature types for current booster.
   * \param ft Input feature types.
   */
  virtual void SetFeatureTypes(std::vector<std::string> const& ft) = 0;
  /*!
   * \brief Get the feature types for current booster.
   * \param fn Output feature types
   */
  virtual void GetFeatureTypes(std::vector<std::string>* ft) const = 0;
  /**
   * @brief Getter for categories.
   */
  [[nodiscard]] virtual CatContainer const* Cats() const = 0;
  /**
   * @brief Slice the model.
   *
   * See InplacePredict for layer parameters.
   *
   * @param step step size between slice.
   * @param out_of_bound Return true if end layer is out of bound.
   *
   * @return a sliced model.
   */
  virtual Learner* Slice(bst_layer_t begin, bst_layer_t end, bst_layer_t step,
                         bool* out_of_bound) = 0;
  /*!
   * \brief dump the model in the requested format
   * \param fmap feature map that may help give interpretations of feature
   * \param with_stats extra statistics while dumping model
   * \param format the format to dump the model in
   * \return a vector of dump for boosters.
   */
  virtual std::vector<std::string> DumpModel(const FeatureMap& fmap, bool with_stats,
                                             std::string format) = 0;

  virtual XGBAPIThreadLocalEntry& GetThreadLocal() const = 0;
  /**
   * @brief Reset the booster object to release data caches used for training.
   */
  virtual void Reset() = 0;
  /*!
   * \brief Create a new instance of learner.
   * \param cache_data The matrix to cache the prediction.
   * \return Created learner.
   */
  static Learner* Create(const std::vector<std::shared_ptr<DMatrix>>& cache_data);
  /**
   * \brief Return the context object of this Booster.
   */
  virtual Context const* Ctx() const = 0;

 protected:
  /*! \brief objective function */
  std::unique_ptr<ObjFunction> obj_;
  /*! \brief The gradient booster used by the model*/
  std::unique_ptr<GradientBooster> gbm_;
  /*! \brief The evaluation metrics used to evaluate the model. */
  std::vector<std::unique_ptr<Metric>> metrics_;
  /*! \brief Training parameter. */
  Context ctx_;
};

/**
 * @brief Strategy for building multi-target models.
 */
enum class MultiStrategy : std::int32_t {
  kOneOutputPerTree = 0,
  kMultiOutputTree = 1,
};

/**
 * @brief State shared by the learner and gradient booster.
 */
struct LearnerModelState {
 private:
  /**
   * @brief Intercept in margin space, used to initialize predictions.
   */
  linalg::Vector<float> base_score_;
  /** @brief Intercept in objective space, used for model serialization. */
  std::vector<float> base_score_value_;

 public:
  /** @brief The number of features. */
  bst_feature_t num_feature{0};
  /** @brief Number of classes for a multi-class model. */
  std::int32_t num_class{0};
  /** @brief Number of targets. */
  bst_target_t num_target{1};
  /** @brief Whether the intercept should be estimated from training data. */
  bool boost_from_average{true};
  /** @brief The number of classes or targets. */
  std::uint32_t num_output_group{0};
  /** @brief Current task, determined by objective. */
  ObjInfo task{ObjInfo::kRegression};
  /** @brief Strategy for building multi-target models. */
  MultiStrategy multi_strategy{MultiStrategy::kOneOutputPerTree};

  LearnerModelState() = default;
  LearnerModelState(Context const* ctx, bst_feature_t n_features, std::int32_t n_classes,
                    bst_target_t n_targets, bool boost_from_average,
                    std::vector<float> base_score_value, linalg::Vector<float> base_score,
                    ObjInfo task, MultiStrategy multi_strategy);
  // This ctor is only used by tests.
  LearnerModelState(bst_feature_t n_features, linalg::Vector<float> base_score,
                    std::uint32_t n_groups, bst_target_t n_targets, MultiStrategy multi_strategy)
      : base_score_{std::move(base_score)},
        num_feature{n_features},
        num_class{n_groups > 1 ? static_cast<std::int32_t>(n_groups) : 0},
        num_target{n_targets},
        num_output_group{std::max(n_groups, n_targets)},
        multi_strategy{multi_strategy} {}

  linalg::VectorView<float const> BaseScore(Context const* ctx) const;
  [[nodiscard]] linalg::VectorView<float const> BaseScore(DeviceOrd device) const;
  [[nodiscard]] std::vector<float> const& BaseScoreValue() const { return base_score_value_; }
  void SetBaseScore(Context const* ctx, std::vector<float> value, linalg::Vector<float> margin);
  void ConfigureDevice(Context const* ctx);

  void Copy(LearnerModelState const& that);
  [[nodiscard]] bool IsVectorLeaf() const noexcept {
    return multi_strategy == MultiStrategy::kMultiOutputTree;
  }
  [[nodiscard]] bst_target_t OutputLength() const noexcept { return this->num_output_group; }
  [[nodiscard]] bst_target_t LeafLength() const noexcept {
    return this->IsVectorLeaf() ? this->OutputLength() : 1;
  }

  [[nodiscard]] bool NeedsInitialization() const {
    return num_feature == 0 || num_output_group == 0 || base_score_.Size() == 0;
  }
  [[nodiscard]] bool Initialized() const { return !this->NeedsInitialization(); }
};

}  // namespace xgboost
#endif  // XGBOOST_LEARNER_H_
