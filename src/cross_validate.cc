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
}  // namespace xgboost
