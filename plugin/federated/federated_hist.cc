/**
 * Copyright 2024, XGBoost contributors
 */
#include "federated_hist.h"

#include "../../src/collective/allgather.h"         // for AllgatherV
#include "../../src/collective/communicator-inl.h"  // for GetRank
#include "../../src/tree/hist/histogram.h"  // for SubtractHistParallel, BuildSampleHistograms

namespace xgboost::tree {
namespace {
// Copy the bins into a dense matrix.
auto CopyBinsToDense(Context const *ctx, GHistIndexMatrix const &gidx) {
  auto n_samples = gidx.Size();
  auto n_features = gidx.Features();
  std::vector<bst_bin_t> bins(n_samples * n_features);
  auto bins_view = linalg::MakeTensorView(ctx, bins, n_samples, n_features);
  common::ParallelFor(n_samples, ctx->Threads(), [&](auto ridx) {
    for (bst_feature_t fidx = 0; fidx < n_features; fidx++) {
      bins_view(ridx, fidx) = gidx.GetGindex(ridx, fidx);
    }
  });
  return bins;
}
}  // namespace

template <bool any_missing>
void FederataedHistPolicy::DoBuildLocalHistograms(
    common::BlockedSpace2d const &space, GHistIndexMatrix const &gidx,
    std::vector<bst_node_t> const &nodes_to_build,
    common::RowSetCollection const &row_set_collection, common::Span<GradientPair const> gpair_h,
    bool force_read_by_column, common::ParallelGHistBuilder *buffer) {
  if (is_col_split_) {
    // Call the interface to transmit gidx information to the secure worker for encrypted
    // histogram computation

    // FIXME: this can be done during reset.
    if (!is_gidx_initialized_) {
      auto bins = CopyBinsToDense(ctx_, gidx);
      auto cuts = gidx.Cuts().Ptrs();
      plugin_->Reset(cuts, bins);
      is_gidx_initialized_ = true;
    }

    // Further use the row set collection info to
    // get the encrypted histogram from the secure worker
    std::vector<std::uint64_t const *> ptrs(nodes_to_build.size());
    std::vector<std::size_t> sizes(nodes_to_build.size());
    std::vector<bst_node_t> nodes(nodes_to_build.size());
    for (std::size_t i = 0; i < nodes_to_build.size(); ++i) {
      auto nidx = nodes_to_build[i];
      ptrs[i] = row_set_collection[nidx].begin();
      sizes[i] = row_set_collection[nidx].Size();
      nodes[i] = nidx;
    }
    hist_data_ = this->plugin_->BuildEncryptedHistVert(ptrs, sizes, nodes);
  } else {
    BuildSampleHistograms<any_missing>(this->ctx_->Threads(), space, gidx, nodes_to_build,
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

    // Perform AllGather
    HostDeviceVector<std::int8_t> hist_entries;
    std::vector<std::int64_t> recv_segments;
    collective::SafeColl(
        collective::AllgatherV(ctx, linalg::MakeVec(hist_data_), &recv_segments, &hist_entries));

    // Call interface here to post-process the messages
    common::Span<double> hist_aggr =
        plugin_->SyncEncryptedHistVert(common::RestoreType<std::uint8_t>(hist_entries.HostSpan()));

    // Update histogram for the label owner
    if (collective::GetRank() == 0) {
      // iterator of the beginning of the vector
      bst_node_t n_nodes = nodes_to_build.size();
      std::int32_t n_workers = collective::GetWorldSize();
      bst_idx_t worker_size = hist_aggr.size() / n_workers;
      CHECK_EQ(hist_aggr.size() % n_workers, 0);
      // Initialize histogram. For the normal case, this is done by the parallel hist
      // buffer. We should try to unify the code paths.
      for (auto nidx : nodes_to_build) {
        auto hist_dst = hist[nidx];
        std::fill_n(hist_dst.data(), hist_dst.size(), GradientPairPrecise{});
      }

      // for each worker
      for (auto widx = 0; widx < n_workers; ++widx) {
        auto worker_hist = hist_aggr.subspan(widx * worker_size, worker_size);
        // for each node
        for (bst_node_t nidx_in_set = 0; nidx_in_set < n_nodes; ++nidx_in_set) {
          constexpr double kFactor = sizeof(GradientPairPrecise) / sizeof(double);
          static_assert(kFactor == 2);
          auto hist_size = n_total_bins * kFactor;  // Histogram size for one node.
          auto hist_src = worker_hist.subspan(hist_size * nidx_in_set, hist_size);
          auto hist_src_g = common::RestoreType<GradientPairPrecise>(hist_src);
          auto hist_dst = hist[nodes_to_build[nidx_in_set]];
          CHECK_EQ(hist_src_g.size(), hist_dst.size());
          common::IncrementHist(hist_dst, hist_src_g, 0, hist_dst.size());
        }
      }
    }
  } else {
    common::ParallelFor2d(space, this->ctx_->Threads(), [&](std::size_t node, common::Range1d r) {
      // Merging histograms from each thread.
      buffer->ReduceHist(node, r.begin(), r.end());
    });
    // Secure mode, we need to call the plugin to perform encryption and decryption. Note
    // that the actual aggregation is performed at server side
    auto first_nidx = nodes_to_build.front();
    std::size_t n = n_total_bins * nodes_to_build.size() * 2;
    auto hist_to_aggr = std::vector<double>();
    for (std::size_t hist_idx = 0; hist_idx < n; hist_idx++) {
      double hist_item = reinterpret_cast<double *>(hist[first_nidx].data())[hist_idx];
      hist_to_aggr.push_back(hist_item);
    }
    auto hist_buf = plugin_->BuildEncryptedHistHori(hist_to_aggr);

    // allgather
    HostDeviceVector<std::int8_t> hist_entries;
    std::vector<std::int64_t> recv_segments;
    auto rc = collective::AllgatherV(ctx, linalg::MakeVec(hist_buf), &recv_segments, &hist_entries);
    collective::SafeColl(rc);

    auto hist_aggr =
        plugin_->SyncEncryptedHistHori(common::RestoreType<std::uint8_t>(hist_entries.HostSpan()));
    // Assign the aggregated histogram back to the local histogram
    auto hist_dst = reinterpret_cast<double *>(hist[first_nidx].data());
    std::copy_n(hist_aggr.data(), hist_aggr.size(), hist_dst);
  }

  SubtractHistParallel(ctx, space, p_tree, nodes_to_build, nodes_to_trick, buffer, p_hist);
}
}  // namespace xgboost::tree
