#include <gtest/gtest.h>
#include "../../../../src/tree/gpu_hist/driver.cuh"

namespace xgboost {
namespace tree {

TEST(GpuHist, DriverDepthWise) {
  Driver driver(TrainParam::kDepthWise);
  EXPECT_TRUE(driver.Pop().empty());
  DeviceSplitCandidate split;
  split.loss_chg = 1.0f;
  ExpandEntry root(0, 0, split);
  driver.Push({root});
  EXPECT_EQ(driver.Pop().front().nid, 0);
  driver.Push({ExpandEntry{1, 1, split}});
  driver.Push({ExpandEntry{2, 1, split}});
  driver.Push({ExpandEntry{3, 2, split}});
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

TEST(GpuHist, DriverLossGuided) {
  DeviceSplitCandidate high_gain;
  high_gain.loss_chg = 5.0f;
  DeviceSplitCandidate low_gain;
  low_gain.loss_chg = 1.0f;

  Driver driver(TrainParam::kLossGuide);
  EXPECT_TRUE(driver.Pop().empty());
  ExpandEntry root(0, 0, high_gain);
  driver.Push({root});
  EXPECT_EQ(driver.Pop().front().nid, 0);
  // Select high gain first
  driver.Push({ExpandEntry{1, 1, low_gain}});
  driver.Push({ExpandEntry{2, 2, high_gain}});
  auto res = driver.Pop();
  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].nid, 2);
  res = driver.Pop();
  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].nid, 1);

  // If equal gain, use nid
  driver.Push({ExpandEntry{2, 1, low_gain}});
  driver.Push({ExpandEntry{1, 1, low_gain}});
  res = driver.Pop();
  EXPECT_EQ(res[0].nid, 1);
  res = driver.Pop();
  EXPECT_EQ(res[0].nid, 2);
}
}  // namespace tree
}  // namespace xgboost
