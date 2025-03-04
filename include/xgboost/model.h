/**
 * Copyright 2019-2025, XGBoost Contributors
 *
 * @file model.h
 * @brief Defines the abstract interface for different components in XGBoost.
 */
#ifndef XGBOOST_MODEL_H_
#define XGBOOST_MODEL_H_

namespace xgboost {

class Json;

struct Model {
  virtual ~Model() = default;
  /*!
   * \brief load the model from a JSON object
   * \param in JSON object where to load the model from
   */
  virtual void LoadModel(Json const& in) = 0;
  /*!
   * \brief saves the model config to a JSON object
   * \param out JSON container where to save the model to
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
}  // namespace xgboost

#endif  // XGBOOST_MODEL_H_
