#include "./c_api/c_api_error.h"
#include "./c_api/c_api_utils.h"  // for CastDMatrixHandle
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"

namespace xgboost {
void GetGradient(Context const* ctx, MetaInfo const& info,
                 std::vector<common::Span<std::size_t const>> ridxs_folds, std::int32_t iter) {
  auto k_folds = ridxs_folds.size();
  CHECK(info.num_nonzero_) << "Missing data is not yet supported.";
  std::string obj_name = "reg:squarederror";  // fixme
  std::vector<std::unique_ptr<ObjFunction>> objs;

  std::vector<linalg::Matrix<GradientPair>> gpairs;

  for (std::size_t k = 0; k < k_folds; ++k) {
    objs.emplace_back(ObjFunction::Create(obj_name, ctx));
    objs.back()->Configure(Args{});

    auto ridxs = ridxs_folds[k];
    constexpr std::size_t kNnz = 0;  // fixme
    auto fold_info = info.Slice(ctx, ridxs, kNnz);

    gpairs.emplace_back();
    HostDeviceVector<float> preds(ridxs.size(), 0.0f, ctx->Device());
    objs.back()->GetGradient(preds, fold_info, iter, &gpairs.back());
  }
}

struct FoldInfos {
  std::vector<HostDeviceVector<std::size_t>> ridxs;
};

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

  void SetCUDADevice(bst_d_ordinal_t ordinal) {
    this->ctx_.Init({{"device", "cuda:" + std::to_string(ordinal)}});
  }
  [[nodiscard]] Context const* Ctx() const { return &this->ctx_; }
  [[nodiscard]] ObjFunction* Objective(std::size_t fold_idx) const {
    CHECK_LT(fold_idx, this->objs_.size());
    return this->objs_[fold_idx].get();
  }
};

using CvFoldsHandle = void*;
}  // namespace xgboost

XGB_DLL int XGBCvFoldsCreate(size_t k_folds, xgboost::CvFoldsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new xgboost::CvFolds{k_folds};
  API_END();
}

XGB_DLL int XGBCvFoldsFree(xgboost::CvFoldsHandle hdl) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(hdl);
  delete static_cast<xgboost::CvFolds*>(hdl);
  API_END();
}

XGB_DLL int XGBCvGetGradient(DMatrixHandle dtrain, xgboost::FoldInfos* fold_info, int iter) {
  API_BEGIN();
  using namespace xgboost;  // NOLINT

  auto p_fmat = CastDMatrixHandle(dtrain);
  auto const& info = p_fmat->Info();
  auto const& ridxs_folds = fold_info->ridxs;
  CHECK(!ridxs_folds.empty());
  std::vector<common::Span<std::size_t const>> ridxs_view;
  for (auto const& v : ridxs_folds) {
    ridxs_view.emplace_back(v.ConstDeviceSpan());
  }
  GetGradient(p_fmat->Ctx(), info, ridxs_view, iter);
  API_END();
}
