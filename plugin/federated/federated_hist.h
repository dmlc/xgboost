/**
 * Copyright 2024, XGBoost contributors
 */
#pragma once
#include <cstdint>  // for int32_t
#include <vector>   // for vector

#include "../../src/collective/comm_group.h"   // for GlobalCommGroup
#include "../../src/common/hist_util.h"        // for ParallelGHistBuilder
#include "../../src/common/row_set.h"          // for RowSetCollection
#include "../../src/common/threading_utils.h"  // for BlockedSpace2d
#include "../../src/data/gradient_index.h"     // for GHistIndexMatrix
#include "../../src/tree/hist/hist_cache.h"    // for BoundedHistCollection
#include "federated_comm.h"                    // for FederatedComm
#include "xgboost/base.h"                      // for GradientPair
#include "xgboost/context.h"                   // for Context
#include "xgboost/span.h"                      // for Span
#include "xgboost/tree_model.h"                // for RegTree

namespace xgboost::tree {
/**
 * @brief Federated histogram build policy
 */
class FederataedHistPolicy {
  // fixme: duplicated code
  bool is_col_split_{false};
  bool is_distributed_{false};
  decltype(std::declval<collective::FederatedComm>().EncryptionPlugin()) plugin_;
  xgboost::common::Span<std::uint8_t> hist_data_;
  // Only initialize the aggregation context once
  bool is_gidx_initialized_{false};
  Context const* ctx_;

 public:
  void DoReset(Context const *ctx, bool is_distributed, bool is_col_split) {
    this->is_distributed_ = is_distributed;
    CHECK(is_distributed);
    this->ctx_ = ctx;
    this->is_col_split_ = is_col_split;
    auto const &comm = collective::GlobalCommGroup()->Ctx(ctx, DeviceOrd::CPU());
    auto const &fed = dynamic_cast<collective::FederatedComm const &>(comm);
    plugin_ = fed.EncryptionPlugin();
    CHECK(is_distributed_) << "Unreachable. Single node training can not be federated.";
  }

  template <bool any_missing>
  void DoBuildLocalHistograms(common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
                              std::vector<bst_node_t> const &nodes_to_build,
                              common::RowSetCollection const &row_set_collection,
                              common::Span<GradientPair const> gpair_h, bool force_read_by_column,
                              common::ParallelGHistBuilder *p_buffer);

  void DoSyncHistogram(common::BlockedSpace2d const &space,
                       std::vector<bst_node_t> const &nodes_to_build,
                       std::vector<bst_node_t> const &nodes_to_trick,
                       common::ParallelGHistBuilder *p_buffer, tree::BoundedHistCollection *p_hist);
};
}  // namespace xgboost::tree
