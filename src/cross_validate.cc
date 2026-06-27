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
}  // namespace xgboost

int XGBCvGetGradient(DMatrixHandle dtrain, xgboost::FoldInfos* fold_info, int iter) {
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
