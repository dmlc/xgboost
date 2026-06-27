#include "updater_gpu_hist.cuh"

namespace xgboost::tree::cuda_impl {
void CalcRootSumFolds(Context const* ctx,
                      std::vector<linalg::MatrixView<GradientPairInt64>> d_gpair,
                      std::vector<common::Span<GradientPairInt64>> root_sum) {
  auto k_folds = d_gpair.size();
  CHECK_EQ(k_folds, root_sum.size());
  for (std::size_t k = 0; k < k_folds; ++k) {
    CalcRootSum(ctx, d_gpair[k], root_sum[k]);
  }
}

class FusedCvHistTreeMaker {
  void BuildHist();
  void EvaluateRoot();

 public:
  void InitRoots();
};
}  // namespace xgboost::tree::cuda_impl
