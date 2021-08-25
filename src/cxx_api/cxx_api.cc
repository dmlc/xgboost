/*!
 * Copyright (c) 2021 by Contributors
 * \file cxx_api.cc
 * \brief C++ API of XGBoost, which is designed to make it easy for other C++ applications to use
 * XGBoost.
 * \author Hyunsu Cho
 */
#include "xgboost/c_api.h"
#include "xgboost/cxx_api.h"
#include <sstream>
#include <memory>
#include <vector>
#include <type_traits>
#include <cstdint>

namespace xgboost {
namespace cxx_api {

static_assert(std::is_same<bst_ulong, uint64_t>::value,
              "Assumed bst_ulong == uint64_t but failed");

void CAPIGuard(int retval) {
  if (retval != 0) {
    throw XGBoostException(std::string(XGBGetLastError()));
  }
}

using DMatrixHandle = void*;
using BoosterHandle = void*;

class DMatrixInternal {
 private:
  DMatrixHandle handle_;
  friend class DMatrix;
  friend class Booster;
};

class BoosterInternal {
 private:
   BoosterHandle handle_;
   std::vector<DMatrixHandle> cache_mats;
   friend class Booster;
};

DMatrix::DMatrix() : pimpl_(std::make_unique<DMatrixInternal>()) {}
DMatrix::DMatrix(DMatrix&&) = default;
DMatrix::~DMatrix() {}

DMatrix DMatrix::CreateFromMat(
      const float* data,
      bst_ulong nrow,
      bst_ulong ncol,
      float missing,
      int nthread) {
  std::string array_interface;
  {
    std::ostringstream oss;
    oss << "{\"data\": [" << reinterpret_cast<std::size_t>(data) << ", true], "
        << "\"shape\": [" << nrow << ", " << ncol << "], "
        << "\"typestr\": \"<f4\", \"version\": 3}";
    array_interface = oss.str();
  }
  std::string config = "{\"nthread\": 16, \"missing\": NaN}";

  DMatrixHandle handle;
  CAPIGuard(XGDMatrixCreateFromDense(array_interface.c_str(), config.c_str(), &handle));

  DMatrix mat;
  mat.pimpl_->handle_ = handle;
  return mat;
}

Booster::Booster() : pimpl_(std::make_unique<BoosterInternal>()) {
  CAPIGuard(XGBoosterCreate(nullptr, 0, &pimpl_->handle_));
}

Booster::~Booster() {}

void Booster::LoadModel(const std::string& fname) {
  CAPIGuard(XGBoosterLoadModel(pimpl_->handle_, fname.c_str()));
}

PredictOutput Booster::Predict(
      const DMatrix& data,
      bool output_margin,
      bool pred_leaf,
      bool pred_contribs,
      bool approx_contribs,
      bool pred_interactions,
      bool strict_shape,
      std::pair<int, int> iteration_range,
      bool training) {

  int pred_type = 0;
  auto assign_type = [&pred_type](int t) {
    if (pred_type != 0) {
      throw XGBoostException("One type of prediction at a time.");
    }
    pred_type = t;
  };
  if (output_margin) {
    assign_type(1);
  }
  if (pred_contribs) {
    assign_type((!approx_contribs ? 2 : 3));
  }
  if (pred_interactions) {
    assign_type((!approx_contribs ? 4 : 5));
  }
  if (pred_leaf) {
    assign_type(6);
  }
  
  std::string args;
  {
    std::ostringstream oss;
    oss << "{\"type\": " << pred_type
        << ", \"training\": " << (training ? "true" : "false")
        << ", \"iteration_begin\": " << iteration_range.first
        << ", \"iteration_end\": " << iteration_range.second
        << ", \"strict_shape\": " << (strict_shape ? "true" : "false")
        << "}";
    args = oss.str();
  }

  PredictOutput out;
  CAPIGuard(XGBoosterPredictFromDMatrix(
        pimpl_->handle_,
        data.pimpl_->handle_,
        args.c_str(),
        &out.shape,
        &out.ndim,
        &out.preds));
  return out;
}

}  // namespace cxx_api
}  // namespace xgboost

