/**
 * Copyright 2024, XGBoost contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <vector>   // for vector-

#include "../../src/collective/allgather.h"         // for AllgatherV
#include "../../src/collective/comm_group.h"        // for GlobalCommGroup
#include "../../src/collective/communicator-inl.h"  // for GetRank
#include "../../src/common/hist_util.h"             // for ParallelGHistBuilder, SubtractionHist
#include "../../src/common/row_set.h"               // for RowSetCollection
#include "../../src/common/threading_utils.h"       // for ParallelFor2d, Range1d, BlockedSpace2d
#include "../../src/data/gradient_index.h"          // for GHistIndexMatrix
#include "../../src/tree/hist/hist_cache.h"         // for BoundedHistCollection
#include "federated_comm.h"                         // for FederatedComm
#include "xgboost/base.h"                           // for GradientPair
#include "xgboost/context.h"                        // for Context
#include "xgboost/span.h"                           // for Span
#include "xgboost/tree_model.h"                     // for RegTree

namespace xgboost::tree {
/**
 * @brief Federated histogram build policy
 */
class FederataedHistPolicy {
  // fixme: duplicated code
  bool is_col_split_{false};
  bool is_distributed_{false};
  std::int32_t n_threads_{false};
  decltype(std::declval<collective::FederatedComm>().EncryptionPlugin()) plugin_;
  xgboost::common::Span<std::uint8_t> hist_data_;
  // only initialize the aggregation context once
  bool is_aggr_context_initialized_ = false;  // fixme

 public:
  void Reset(Context const *ctx, bool is_distributed, DMatrix const *p_fmat) {
    this->is_distributed_ = is_distributed;
    CHECK(is_distributed);
    this->n_threads_ = ctx->Threads();
    this->is_col_split_ = p_fmat->Info().IsColumnSplit();
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
                              common::ParallelGHistBuilder *buffer) {
    if (is_col_split_) {
      // Call the interface to transmit gidx information to the secure worker for encrypted
      // histogram computation
      auto cuts = gidx.Cuts().Ptrs();
      // fixme: this can be done during reset.
      if (!is_aggr_context_initialized_) {
        auto slots = std::vector<int>();
        auto num_rows = gidx.Size();
        for (std::size_t row = 0; row < num_rows; row++) {
          for (std::size_t f = 0; f < cuts.size() - 1; f++) {
            auto slot = gidx.GetGindex(row, f);
            slots.push_back(slot);
          }
        }
        plugin_->Reset(cuts, slots);
        is_aggr_context_initialized_ = true;
      }

      // Further use the row set collection info to
      // get the encrypted histogram from the secure worker
      std::vector<std::uint64_t const *> ptrs(nodes_to_build.size());
      std::vector<std::size_t> sizes(nodes_to_build.size());
      std::vector<bst_node_t> nodes(nodes_to_build.size());
      for (std::size_t i = 0; i < nodes_to_build.size(); ++i) {
        auto nidx = nodes_to_build[i];
        ptrs[i] = row_set_collection[nidx].begin;
        sizes[i] = row_set_collection[nidx].Size();
        nodes[i] = nidx;
      }
      hist_data_ = this->plugin_->BuildEncryptedHistVert(ptrs, sizes, nodes);
    } else {
      common::BuildSampleHistograms<any_missing>(this->n_threads_, space, gidx, nodes_to_build,
                                                 row_set_collection, gpair_h, force_read_by_column,
                                                 buffer);
    }
  }

  void DoSyncHistogram(Context const *ctx, RegTree const *p_tree,
                       std::vector<bst_node_t> const &nodes_to_build,
                       std::vector<bst_node_t> const &nodes_to_trick,
                       common::ParallelGHistBuilder *buffer, tree::BoundedHistCollection *p_hist) {
    auto n_total_bins = buffer->TotalBins();
    common::BlockedSpace2d space(
        nodes_to_build.size(), [&](std::size_t) { return n_total_bins; }, 1024);

    auto &hist = *p_hist;
    if (is_col_split_) {
      // Under secure vertical mode, we perform allgather to get the global histogram.
      // note that only Label Owner needs the global histogram
      CHECK(!nodes_to_build.empty());
      // Front item of nodes_to_build
      auto first_nidx = nodes_to_build.front();
      // *2 because we have a pair of g and h for each histogram item
      std::size_t n = n_total_bins * nodes_to_build.size() * 2;

      // Perform AllGather
      HostDeviceVector<std::int8_t> hist_entries;
      std::vector<std::int64_t> recv_segments;
      collective::SafeColl(
          collective::AllgatherV(ctx, linalg::MakeVec(hist_data_), &recv_segments, &hist_entries));

      // Call interface here to post-process the messages
      auto hist_aggr = plugin_->SyncEncryptedHistVert(
          common::RestoreType<std::uint8_t>(hist_entries.HostSpan()));

      // Update histogram for label owner
      if (collective::GetRank() == 0) {
        // iterator of the beginning of the vector
        auto it = reinterpret_cast<double *>(hist[first_nidx].data());
        // iterate through the hist vector of the label owner
        for (std::size_t i = 0; i < n; i++) {
          // get the sum of the entries from all ranks
          double hist_sum = 0.0;
          for (std::size_t rank_idx = 0; rank_idx < hist_aggr.size() / n; rank_idx++) {
            int flat_idx = rank_idx * n + i;
            hist_sum += hist_aggr[flat_idx];
          }
          // update rank 0's record with the global histogram
          *it = hist_sum;
          it++;
        }
      }
    } else {
      common::ParallelFor2d(space, this->n_threads_, [&](std::size_t node, common::Range1d r) {
        // Merging histograms from each thread.
        buffer->ReduceHist(node, r.begin(), r.end());
      });
      // Secure mode, we need to call interface to perform encryption and decryption
      // note that the actual aggregation will be performed at server side
      auto first_nidx = nodes_to_build.front();
      std::size_t n = n_total_bins * nodes_to_build.size() * 2;
      auto hist_to_aggr = std::vector<double>();
      for (std::size_t hist_idx = 0; hist_idx < n; hist_idx++) {
        double hist_item = reinterpret_cast<double *>(hist[first_nidx].data())[hist_idx];
        hist_to_aggr.push_back(hist_item);
      }
      // ProcessHistograms
      auto hist_buf = plugin_->BuildEncryptedHistHori(hist_to_aggr);

      // allgather
      HostDeviceVector<std::int8_t> hist_entries;
      std::vector<std::int64_t> recv_segments;
      auto rc =
          collective::AllgatherV(ctx, linalg::MakeVec(hist_buf), &recv_segments, &hist_entries);
      collective::SafeColl(rc);

      auto hist_aggr = plugin_->SyncEncryptedHistHori(
          common::RestoreType<std::uint8_t>(hist_entries.HostSpan()));
      // Assign the aggregated histogram back to the local histogram
      for (std::size_t hist_idx = 0; hist_idx < n; hist_idx++) {
        reinterpret_cast<double *>(hist[first_nidx].data())[hist_idx] = hist_aggr[hist_idx];
      }
    }

    common::BlockedSpace2d const &subspace =
        nodes_to_trick.size() == nodes_to_build.size()
            ? space
            : common::BlockedSpace2d{nodes_to_trick.size(),
                                     [&](std::size_t) { return n_total_bins; }, 1024};
    common::ParallelFor2d(
        subspace, this->n_threads_, [&](std::size_t nidx_in_set, common::Range1d r) {
          auto subtraction_nidx = nodes_to_trick[nidx_in_set];
          auto parent_id = p_tree->Parent(subtraction_nidx);
          auto sibling_nidx = p_tree->IsLeftChild(subtraction_nidx) ? p_tree->RightChild(parent_id)
                                                                    : p_tree->LeftChild(parent_id);
          auto sibling_hist = hist[sibling_nidx];
          auto parent_hist = hist[parent_id];
          auto subtract_hist = hist[subtraction_nidx];
          common::SubtractionHist(subtract_hist, parent_hist, sibling_hist, r.begin(), r.end());
        });
  }
};
}  // namespace xgboost::tree
