#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"  // for CastDMatrixHandle
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

namespace xgboost {
struct FoldInfo {
  std::vector<HostDeviceVector<std::size_t>> ridxs;

 public:
  [[nodiscard]] auto TrainingFold(std::size_t k) const { return ridxs.at(k).ConstDeviceSpan(); }
  [[nodiscard]] auto KFolds() const noexcept(true) { return this->ridxs.size(); }
};

void GetGradient(Context const* ctx, MetaInfo const& info, FoldInfo const& finfo,
                 std::int32_t iter) {
  auto k_folds = finfo.KFolds();
  CHECK(info.num_nonzero_) << "Missing data is not yet supported.";
  std::string obj_name = "reg:squarederror";  // fixme
  std::vector<std::unique_ptr<ObjFunction>> objs;

  std::vector<linalg::Matrix<GradientPair>> gpairs;

  for (std::size_t k = 0; k < k_folds; ++k) {
    objs.emplace_back(ObjFunction::Create(obj_name, ctx));
    objs.back()->Configure(Args{});

    auto ridxs = finfo.TrainingFold(k);
    constexpr std::size_t kNnz = 0;  // fixme
    auto fold_info = info.Slice(ctx, ridxs, kNnz);

    gpairs.emplace_back();
    HostDeviceVector<float> preds(ridxs.size(), 0.0f, ctx->Device());
    objs.back()->GetGradient(preds, fold_info, iter, &gpairs.back());
  }
}

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
using FoldInfosHandle = void*;
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

XGB_DLL int XGBCvGetGradient(DMatrixHandle dtrain, FoldInfosHandle c_fold_info, int iter) {
  API_BEGIN();

  auto p_fmat = CastDMatrixHandle(dtrain);
  auto fold_info = static_cast<FoldInfo*>(c_fold_info);
  auto const& info = p_fmat->Info();
  CHECK(!fold_info->ridxs.empty());
  GetGradient(p_fmat->Ctx(), info, *fold_info, iter);
  API_END();
}
