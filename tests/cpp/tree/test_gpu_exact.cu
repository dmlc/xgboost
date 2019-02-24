#include <gtest/gtest.h>
#include <xgboost/tree_updater.h>

#include <vector>
#include <string>
#include <utility>

#include "../helpers.h"

namespace xgboost {
namespace tree {

TEST(GPUExact, Update) {
  using Arg = std::pair<std::string, std::string>;
  std::vector<Arg> args{
    {"n_gpus", "1"},
    {"gpu_id", "0"},
    {"max_depth", "1"}};

  auto* p_gpuexact_maker = TreeUpdater::Create("grow_gpu");
  p_gpuexact_maker->Init(args);

  size_t constexpr n_rows = 4;
  size_t constexpr n_cols = 8;
  bst_float constexpr sparsity = 0.0f;

  auto dmat = CreateDMatrix(n_rows, n_cols, sparsity, 3);
  std::vector<GradientPair> h_gpair(n_rows);
  for (size_t i = 0; i < n_rows; ++i) {
      h_gpair[i] = GradientPair(i % 2, 1);
  }
  HostDeviceVector<GradientPair> gpair (h_gpair);
  RegTree tree;

  p_gpuexact_maker->Update(&gpair, (*dmat).get(), {&tree});
  auto const& nodes = tree.GetNodes();
  ASSERT_EQ(nodes.size(), 3);

  float constexpr kRtEps = 1e-6;
  ASSERT_NEAR(tree.Stat(0).sum_hess, 4, kRtEps);
  ASSERT_NEAR(tree.Stat(1).sum_hess, 2, kRtEps);
  ASSERT_NEAR(tree.Stat(2).sum_hess, 2, kRtEps);

  ASSERT_NEAR(tree.Stat(0).loss_chg, 0.8f, kRtEps);

  delete dmat;
  delete p_gpuexact_maker;
}

}  // namespace tree
}  // namespace xgboost