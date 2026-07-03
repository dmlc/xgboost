/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cross_validate.h"

#include <memory>  // for unique_ptr
#include <string>  // for string
#include <vector>  // for vector

#include "./c_api/c_api_error.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

namespace xgboost {
// The model part of the cross validation result, containing the trees and objectives.
//
// Tree updaters should not be part of it as they are considered "optimizers" and not part
// of the model.
class CvFolds {
  std::vector<std::unique_ptr<ObjFunction>> objs_;
  Context ctx_;

 public:
  explicit CvFolds(std::size_t k_folds) {
    CHECK_GT(k_folds, 0);
    std::string obj_name = "reg:squarederror";  // FIXME(jiamingy): Support more objs.
    ctx_.Init({{"device", "cuda"}});
    for (std::size_t i = 0; i < k_folds; ++i) {
      objs_.emplace_back(ObjFunction::Create(obj_name, &ctx_));
      objs_.back()->Configure(Args{});
    }
  }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->objs_.size(); }

  [[nodiscard]] Context const* Ctx() const { return &this->ctx_; }
  [[nodiscard]] ObjFunction* Objective(std::size_t fold_idx) const {
    CHECK_LT(fold_idx, this->objs_.size());
    return this->objs_[fold_idx].get();
  }
};

using CvFoldsHandle = void*;
}  // namespace xgboost

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldsCreate(size_t k_folds, CvFoldsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new CvFolds{k_folds};
  API_END();
}

XGB_DLL int XGBCvFoldsFree(CvFoldsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<CvFolds*>(hdl);
  API_END();
}

XGB_DLL int XGBCvFoldGpairsCreate(FoldGpairsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new FoldGpairs{};
  API_END();
}

XGB_DLL int XGBCvFoldGpairsFree(FoldGpairsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<FoldGpairs*>(hdl);
  API_END();
}
