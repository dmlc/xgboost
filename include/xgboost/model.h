/*!
 * Copyright (c) 2019 by Contributors
 * \file model.h
 * \brief Defines the abstract interface for different components in XGBoost.
 */
#ifndef XGBOOST_MODEL_H_
#define XGBOOST_MODEL_H_

#include <cstdint>
#include <iostream>

namespace dmlc {
class Stream;
}  // namespace dmlc

namespace xgboost {

class Json;

struct Model {
  virtual ~Model() = default;
  /*!
   * \brief load the model from a json object
   * \param in json object where to load the model from
   */
  virtual void LoadModel(Json const& in) = 0;
  /*!
   * \brief saves the model config to a json object
   * \param out json container where to save the model to
   */
  virtual void SaveModel(Json* out) const = 0;
};

struct Configurable {
  virtual ~Configurable() = default;
  /*!
   * \brief Load configuration from JSON object
   * \param in JSON object containing the configuration
   */
  virtual void LoadConfig(Json const& in) = 0;
  /*!
   * \brief Save configuration to JSON object
   * \param out pointer to output JSON object
   */
  virtual void SaveConfig(Json* out) const = 0;
};

struct LearnerModelParamLegacy;

enum class OutputType : int32_t {
  kSingle,
  kMulti
};

inline std::ostream& operator<<(std::ostream& os, OutputType t) {
  os << static_cast<int32_t>(t);
  return os;
}

/*
 * \brief Basic Model Parameters, used to describe the booster.
 */
struct LearnerModelParam {
  /* \brief global bias */
  float base_score { 0.5 };
  /* \brief number of features  */
  uint32_t num_feature { 0 };
  /* \brief number of classes, if it is multi-class classification  */
  uint32_t num_output_group { 0 };
  /* \brief Output type of a tree, either single or multi. */
  OutputType output_type { OutputType::kSingle };

  LearnerModelParam() = default;
  // As the old `LearnerModelParamLegacy` is still used by binary IO, we keep
  // this one as an immutable copy.
  LearnerModelParam(LearnerModelParamLegacy const& user_param, float base_margin);
  /* \brief Whether this parameter is initialized with LearnerModelParamLegacy. */
  bool Initialized() const { return num_feature != 0; }
};
}  // namespace xgboost

#endif  // XGBOOST_MODEL_H_
