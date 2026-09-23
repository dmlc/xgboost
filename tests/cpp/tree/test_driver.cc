/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "../../../src/tree/driver.h"
#include "../../../src/tree/hist/expand_entry.h"
#include "../../../src/tree/param.h"
#include "test_empty_pop.h"

namespace xgboost::tree {
namespace {
CPUExpandEntry MakeCPUExpandEntry(bst_node_t nidx, bst_node_t depth, float loss_chg) {
  SplitEntry split;
  split.loss_chg = loss_chg;
  return CPUExpandEntry{nidx, depth, split};
}

CPUExpandEntry ValidCPU(bst_node_t nidx, bst_node_t depth) {
  return MakeCPUExpandEntry(nidx, depth, 2.0f);
}
CPUExpandEntry InvalidCPU(bst_node_t nidx, bst_node_t depth) {
  return MakeCPUExpandEntry(nidx, depth, 0.5f);
}

void QueueBoundaryCase(Driver<CPUExpandEntry>* driver, std::size_t batch_size) {
  std::vector<CPUExpandEntry> entries;
  entries.reserve(batch_size + 3);
  for (std::size_t i = 0; i < batch_size; ++i) {
    entries.push_back(ValidCPU(static_cast<bst_node_t>(i + 1), 1));
  }
  entries.push_back(InvalidCPU(static_cast<bst_node_t>(batch_size + 1), 1));
  entries.push_back(ValidCPU(static_cast<bst_node_t>(batch_size + 2), 2));
  entries.push_back(ValidCPU(static_cast<bst_node_t>(batch_size + 3), 2));
  driver->Push(entries);
}
}  // namespace

TEST(QuantileHist, DriverEmptyPopSkipsInvalidGroup) {
  constexpr std::size_t kBatch = 2;
  Driver<CPUExpandEntry> driver{test_empty_pop::DepthwiseGammaParam(), kBatch};
  QueueBoundaryCase(&driver, kBatch);
  test_empty_pop::ExpectPopReturnsDeeperWork(&driver, kBatch);
}

TEST(QuantileHist, DriverEmptyPopAtDefaultBatchSize) {
  constexpr std::size_t kBatch = 256;
  Driver<CPUExpandEntry> driver{test_empty_pop::DepthwiseGammaParam()};
  QueueBoundaryCase(&driver, kBatch);
  test_empty_pop::ExpectPopReturnsDeeperWork(&driver, kBatch);
}

TEST(QuantileHist, DriverGrowthLoopExpandsChildren) {
  constexpr std::size_t kBatch = 256;
  Driver<CPUExpandEntry> driver{test_empty_pop::DepthwiseGammaParam()};
  std::vector<CPUExpandEntry> level;
  level.reserve(kBatch + 1);
  for (std::size_t i = 0; i < kBatch; ++i) {
    level.push_back(ValidCPU(static_cast<bst_node_t>(i + 1), 1));
  }
  level.push_back(InvalidCPU(static_cast<bst_node_t>(kBatch + 1), 1));
  driver.Push(level);

  test_empty_pop::ExpectGrowthLoopExpandsChildren<CPUExpandEntry>(
      &driver, [](bst_node_t nidx, bst_node_t depth) { return ValidCPU(nidx, depth); }, kBatch);
}
}  // namespace xgboost::tree
