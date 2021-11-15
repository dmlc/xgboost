#include <gtest/gtest.h>

#include <algorithm>  // std::is_sorted

#include "../../../../src/tree/driver.h"
#include "../../../../src/tree/gpu_hist/expand_entry.cuh"

namespace xgboost {
namespace tree {

TEST(Driver, DepthWise) {
  TrainParam p;
  p.UpdateAllowUnknown(Args{{"grow_policy", "depthwise"}});
  Driver<GPUExpandEntry> driver(p);
  EXPECT_TRUE(driver.Pop().empty());
  DeviceSplitCandidate split;
  split.loss_chg = 1.0f;
  GPUExpandEntry root(0, 0, split, .0f, .0f, .0f);
  driver.Push({root});
  EXPECT_EQ(driver.Pop().front().nid, 0);
  driver.Push({GPUExpandEntry{1, 1, split, .0f, .0f, .0f}});
  driver.Push({GPUExpandEntry{2, 1, split, .0f, .0f, .0f}});
  driver.Push({GPUExpandEntry{3, 2, split, .0f, .0f, .0f}});
  // Should return entries from level 1
  auto res = driver.Pop();
  EXPECT_EQ(res.size(), 2);
  for (auto &e : res) {
    EXPECT_EQ(e.depth, 1);
  }
  res = driver.Pop();
  EXPECT_EQ(res[0].depth, 2);
  EXPECT_TRUE(driver.Pop().empty());
}

TEST(Driver, LossGuided) {
  DeviceSplitCandidate high_gain;
  high_gain.loss_chg = 5.0f;
  DeviceSplitCandidate low_gain;
  low_gain.loss_chg = 1.0f;

  TrainParam p;
  p.UpdateAllowUnknown(Args{{"grow_policy", "lossguide"}});
  Driver<GPUExpandEntry> driver(p);

  EXPECT_TRUE(driver.Pop().empty());
  GPUExpandEntry root(0, 0, high_gain, .0f, .0f, .0f);
  driver.Push({root});
  EXPECT_EQ(driver.Pop().front().nid, 0);
  // Select high gain first
  driver.Push({GPUExpandEntry{1, 1, low_gain, .0f, .0f, .0f}});
  driver.Push({GPUExpandEntry{2, 2, high_gain, .0f, .0f, .0f}});
  auto res = driver.Pop();
  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].nid, 2);
  res = driver.Pop();
  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].nid, 1);

  // If equal gain, use nid
  driver.Push({GPUExpandEntry{2, 1, low_gain, .0f, .0f, .0f}});
  driver.Push({GPUExpandEntry{1, 1, low_gain, .0f, .0f, .0f}});
  res = driver.Pop();
  EXPECT_EQ(res[0].nid, 1);
  res = driver.Pop();
  EXPECT_EQ(res[0].nid, 2);
}

TEST(Driver, MaxGreedyNodes) {
  TrainParam p;
  p.UpdateAllowUnknown(Args{{"grow_policy", "lossguide"}, {"max_greedy_nodes", "5"}});
  Driver<GPUExpandEntry> d_vec{p};     // push a vector
  Driver<GPUExpandEntry> d_single{p};  // push one at a time

  std::vector<GPUExpandEntry> candidates;
  for (size_t i = 0; i < 5; ++i) {
    DeviceSplitCandidate split;
    split.loss_chg = (i + 1);  // avoid 0 loss gain
    GPUExpandEntry e{0, 0, split, .0f, .0f, .0f};
    candidates.emplace_back(e);
    d_single.Push(e);
  }

  d_vec.Push(candidates);

  auto results = d_vec.Pop();
  ASSERT_EQ(results.size(), 5);
  auto comp = [](auto const& l, auto const& r) { return l.split.loss_chg > r.split.loss_chg; };
  ASSERT_TRUE(std::is_sorted(results.cbegin(), results.cend(), comp));

  results = d_single.Pop();
  ASSERT_TRUE(std::is_sorted(results.cbegin(), results.cend(), comp));
}
}  // namespace tree
}  // namespace xgboost
