/*!
 * Copyright (c) 2019 by Contributors
 * \file model.h
 * \brief Defines the abstract interface for different components in XGBoost.
 */
#ifndef XGBOOST_MODEL_H_
#define XGBOOST_MODEL_H_

#include <limits>
#include <cinttypes>

namespace dmlc {
class Stream;
}  // namespace dmlc

namespace xgboost {

namespace experimental {
template <typename D, std::size_t kElementEnd>
class ValueImpl;
class Document;

using Json = ValueImpl<Document, std::numeric_limits<std::size_t>::max()>;
}  // namespace experimental

struct Model {
  /*!
   * \brief Save the model to stream.
   * \param fo output write stream
   */
  virtual void SaveModel(dmlc::Stream* fo) const = 0;
  /*!
   * \brief Load the model from stream.
   * \param fi input read stream
   */
  virtual void LoadModel(dmlc::Stream* fi) = 0;
};

struct Configurable {
  /*!
   * \brief Load configuration from JSON object
   * \param in JSON object containing the configuration
   */
  virtual void LoadConfig(experimental::Json const& in) = 0;
  /*!
   * \brief Save configuration to JSON object
   * \param out pointer to output JSON object
   */
  virtual void SaveConfig(experimental::Json* out) const = 0;
};
}  // namespace xgboost

#endif  // XGBOOST_MODEL_H_
