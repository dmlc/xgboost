/**
 * Copyright 2024, XGBoost contributors
 */
#include "federated_hist.h"

#include "../../src/collective/allgather.h"         // for AllgatherV
#include "../../src/collective/communicator-inl.h"  // for GetRank
#include "../../src/tree/hist/histogram.h"

namespace xgboost::tree {
template <bool any_missing>
void FederataedHistPolicy::DoBuildLocalHistograms(
    common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
    std::vector<bst_node_t> const &nodes_to_build,
    common::RowSetCollection const &row_set_collection, common::Span<GradientPair const> gpair_h,
    bool force_read_by_column, common::ParallelGHistBuilder *buffer) {
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
    BuildSampleHistograms<any_missing>(this->n_threads_, space, gidx, nodes_to_build,
                                       row_set_collection, gpair_h, force_read_by_column, buffer);
  }
}

template void FederataedHistPolicy::DoBuildLocalHistograms<true>(
    common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
    std::vector<bst_node_t> const &nodes_to_build,
    common::RowSetCollection const &row_set_collection, common::Span<GradientPair const> gpair_h,
    bool force_read_by_column, common::ParallelGHistBuilder *buffer);
template void FederataedHistPolicy::DoBuildLocalHistograms<false>(
    common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
    std::vector<bst_node_t> const &nodes_to_build,
    common::RowSetCollection const &row_set_collection, common::Span<GradientPair const> gpair_h,
    bool force_read_by_column, common::ParallelGHistBuilder *buffer);

void FederataedHistPolicy::DoSyncHistogram(Context const *ctx, RegTree const *p_tree,
                                           std::vector<bst_node_t> const &nodes_to_build,
                                           std::vector<bst_node_t> const &nodes_to_trick,
                                           common::ParallelGHistBuilder *buffer,
                                           tree::BoundedHistCollection *p_hist) {
  auto n_total_bins = buffer->TotalBins();
  common::BlockedSpace2d space(
      nodes_to_build.size(), [&](std::size_t) { return n_total_bins; }, 1024);
  CHECK(!nodes_to_build.empty());

  auto &hist = *p_hist;
  if (is_col_split_) {
    // Under secure vertical mode, we perform allgather to get the global histogram. Note
    // that only the label owner (rank == 0) needs the global histogram
    auto first_nidx = nodes_to_build.front();
    // *2 because we have a pair of g and h for each histogram item
    std::size_t n = n_total_bins * nodes_to_build.size() * 2;

    // Perform AllGather
    HostDeviceVector<std::int8_t> hist_entries;
    std::vector<std::int64_t> recv_segments;
    collective::SafeColl(
        collective::AllgatherV(ctx, linalg::MakeVec(hist_data_), &recv_segments, &hist_entries));

    // Call interface here to post-process the messages
    auto hist_aggr =
        plugin_->SyncEncryptedHistVert(common::RestoreType<std::uint8_t>(hist_entries.HostSpan()));

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
    auto rc = collective::AllgatherV(ctx, linalg::MakeVec(hist_buf), &recv_segments, &hist_entries);
    collective::SafeColl(rc);

    auto hist_aggr =
        plugin_->SyncEncryptedHistHori(common::RestoreType<std::uint8_t>(hist_entries.HostSpan()));
    // Assign the aggregated histogram back to the local histogram
    for (std::size_t hist_idx = 0; hist_idx < n; hist_idx++) {
      reinterpret_cast<double *>(hist[first_nidx].data())[hist_idx] = hist_aggr[hist_idx];
    }
  }

  SubtractHistParallel(ctx, space, p_tree, nodes_to_build, nodes_to_trick, buffer, p_hist);
}
}  // namespace xgboost::tree
