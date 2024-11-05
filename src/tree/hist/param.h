/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits

#include "xgboost/parameter.h"   // for XGBoostParameter
#include "xgboost/tree_model.h"  // for RegTree
#include "xgboost/context.h"     // for DeviceOrd

namespace xgboost::tree {
struct HistMakerTrainParam : public XGBoostParameter<HistMakerTrainParam> {
 private:
  constexpr static std::size_t NotSet() { return std::numeric_limits<std::size_t>::max(); }

  std::size_t max_cached_hist_node{NotSet()};  // NOLINT

 public:
  // Smaller for GPU due to memory limitation.
  constexpr static std::size_t CpuDefaultNodes() { return static_cast<std::size_t>(1) << 16; }
  constexpr static std::size_t CudaDefaultNodes() { return static_cast<std::size_t>(1) << 12; }

  bool debug_synchronize{false};
  bool extmem_single_page{false};

  void CheckTreesSynchronized(Context const* ctx, RegTree const* local_tree) const;

  std::size_t MaxCachedHistNodes(DeviceOrd device) const {
    if (max_cached_hist_node != NotSet()) {
      return max_cached_hist_node;
    }
    return device.IsCPU() ? CpuDefaultNodes() : CudaDefaultNodes();
  }

  // declare parameters
  DMLC_DECLARE_PARAMETER(HistMakerTrainParam) {
    DMLC_DECLARE_FIELD(debug_synchronize)
        .set_default(false)
        .describe("Check if all distributed tree are identical after tree construction.");
    DMLC_DECLARE_FIELD(max_cached_hist_node)
        .set_default(NotSet())
        .set_lower_bound(1)
        .describe("Maximum number of nodes in histogram cache.");
    DMLC_DECLARE_FIELD(extmem_single_page).set_default(false);
  }
};
}  // namespace xgboost::tree
