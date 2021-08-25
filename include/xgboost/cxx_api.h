/*!
 * Copyright (c) 2015-2021 by Contributors
 * \file cxx_api.h
 * \brief C++ API of XGBoost, which is designed to make it easy for other C++ applications to use
 * XGBoost.
 * \author Hyunsu Cho
 */
#ifndef XGBOOST_CXX_API_H_
#define XGBOOST_CXX_API_H_

#include <string>
#include <utility>
#include <memory>
#include <limits>
#include <exception>
#include <cstdint>

namespace xgboost {
namespace cxx_api {

class DMatrixInternal;
class BoosterInternal;

static_assert(std::numeric_limits<float>::has_quiet_NaN,
              "Must have a non-signaling NaN");

class XGBoostException : public std::exception {
 public:
  XGBoostException(const std::string& msg) : msg_(msg) {}
	const char* what() const noexcept {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

class DMatrix {
 public:
  DMatrix();
  DMatrix(const DMatrix&) = delete;
  DMatrix(DMatrix&&);
  ~DMatrix();
  static DMatrix CreateFromMat(
      const float* data,
      uint64_t nrow,
      uint64_t ncol,
      float missing = std::numeric_limits<float>::quiet_NaN(),
      int nthread = -1);

 private:
  std::unique_ptr<DMatrixInternal> pimpl_;

  friend class Booster;
};

struct PredictOutput {
 public:
  const float* preds;
  uint64_t ndim;
  const uint64_t* shape;
};

class Booster {
 public:
  Booster();
  ~Booster(); 
  void LoadModel(const std::string& fname);

  PredictOutput Predict(
      const DMatrix& data,
      bool output_margin = false,
      bool pred_leaf = false,
      bool pred_contribs = false,
      bool approx_contribs = false,
      bool pred_interactions = false,
      bool strict_shape = false,
      std::pair<int, int> iteration_range = {0, 0},
      bool training = false
  );

 private:
  std::unique_ptr<BoosterInternal> pimpl_;
};

}  // namespace cxx_api
}  // namespace xgboost

#endif  // XGBOOST_CXX_API_H_
