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
  auto lparam = CreateEmptyGenericParam(0, 1);
  std::vector<Arg> args{{"max_depth", "1"}};

  auto* p_gpuexact_maker = TreeUpdater::Create("grow_gpu", &lparam);
  p_gpuexact_maker->Init(args);

  size_t constexpr kNRows = 4;
  size_t constexpr kNCols = 8;
  bst_float constexpr kSparsity = 0.0f;

  auto dmat = CreateDMatrix(kNRows, kNCols, kSparsity, 3);
  std::vector<GradientPair> h_gpair(kNRows);
  for (size_t i = 0; i < kNRows; ++i) {
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