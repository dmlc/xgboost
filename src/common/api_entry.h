/**
 * Copyright 2016-2023 by XGBoost contributors
 */
#ifndef XGBOOST_COMMON_API_ENTRY_H_
#define XGBOOST_COMMON_API_ENTRY_H_
#include <string>               // std::string
#include <vector>               // std::vector

#include "xgboost/base.h"       // GradientPair,bst_ulong
#include "xgboost/predictor.h"  // PredictionCacheEntry

namespace xgboost {
/**
 * \brief entry to to easily hold returning information
 */
struct XGBAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning raw buffer */
  std::vector<char> ret_char_vec;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief returning float vector. */
  std::vector<float> ret_vec_float;
  /*! \brief temp variable of gradient pairs. */
  std::vector<GradientPair> tmp_gpair;
  /*! \brief Temp variable for returning prediction result. */
  PredictionCacheEntry prediction_entry;
  /*! \brief Temp variable for returning prediction shape. */
  std::vector<bst_ulong> prediction_shape;
};
}  // namespace xgboost
#endif  // XGBOOST_COMMON_API_ENTRY_H_
