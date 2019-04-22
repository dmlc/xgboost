#include <gtest/gtest.h>
#include "../../../tests/cpp/helpers.h"

#include "../../../src/common/random.h"
#include "../src/rv.h"
#include "../src/gibbs_updater.h"

namespace xgboost {

TEST(GibbsUpdater, PersistentPool) {
  constexpr size_t kValues = 16;
  PersistentPool<int32_t> pool(kValues);
  ASSERT_EQ(pool.size(), kValues);

  auto& data = pool.data();
  ASSERT_EQ(data.size(), kValues);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  pool.erase(1);
  ASSERT_EQ(pool[1], 15);
  ASSERT_EQ(pool.size(), 15);

  pool.push(17);
  ASSERT_EQ(pool.size(), kValues);
  ASSERT_EQ(pool[kValues-1], 17);

  pool.push(57);
  ASSERT_EQ(pool.size(), kValues + 1);
  ASSERT_EQ(pool[pool.size() - 1], 57);
}

TEST(GibbsUpdater, Grow) {
  // RegTree tree;
  GibbsParam param;
  std::vector<std::pair<std::string, std::string>> args {};
  param.InitAllowUnknown(args);

  constexpr size_t kRows = 16;
  constexpr size_t kCols = 7;
  constexpr float kSparsity = 0.3;

  HostDeviceVector<float> labels;
  labels.Resize(kRows);
  std::vector<float>& h_labels = labels.HostVector();
  for (size_t i = 0; i < kRows; ++i) {
    h_labels[i] = i;
  }

  GibbsUpdater::TreesChain chain;
  chain.resize(1);     // one generation
  chain[0].resize(1);  // one tree

  auto dmat = CreateDMatrix(kRows, kCols, kSparsity, 3);

  TreeMutation mutate(&chain, 0, dmat->get());
  mutate.grow(0, param, dmat->get());

  ASSERT_EQ(mutate.NumExtraNodes(), 2);
  delete dmat;
}

TEST(GibbsUpdater, Prune) {
  GibbsParam param;
  std::vector<std::pair<std::string, std::string>> args {};
  param.InitAllowUnknown(args);
  constexpr size_t kRows = 16;
  constexpr size_t kCols = 7;
  constexpr float kSparsity = 0.3;
  auto dmat = CreateDMatrix(kRows, kCols, kSparsity, 3);

  GibbsUpdater::TreesChain chain;
  chain.resize(1);     // one generation
  chain[0].resize(1);  // one tree
  TreeMutation mutate(&chain, 0, dmat->get());
  mutate.grow(0, param, dmat->get());
  ASSERT_EQ(mutate.NumExtraNodes(), 2);

  mutate.prune(0, param, dmat->get());
  ASSERT_EQ(mutate.NumExtraNodes(), 0);
  delete dmat;
}

TEST(GibbsUpdater, Update) {
  std::vector<std::pair<std::string, std::string>> args {};

  constexpr size_t kRows = 16;
  constexpr size_t kCols = 7;
  constexpr float kSparsity = 0.3;
  auto dmat = CreateDMatrix(kRows, kCols, kSparsity, 3);
  auto& info = (*dmat)->Info();
  info.labels_.Resize(kRows);
  auto h_labels = info.labels_.HostVector();
  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = i;
  }

  std::vector<std::shared_ptr<RegTree>> trees {std::make_shared<RegTree>()};
  GibbsUpdater updater;
  updater.configure(args);
  updater.update((*(dmat)).get(), trees);

  delete dmat;
}

}  // namespace xgboost
