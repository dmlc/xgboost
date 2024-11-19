/**
 * Copyright 2017-2024, XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <cmath>
#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr, make_unique
#include <utility>  // for move
#include <vector>

#include "../collective/aggregator.h"
#include "../collective/broadcast.h"
#include "../common/bitfield.h"
#include "../common/categorical.h"
#include "../common/cuda_context.cuh"  // CUDAContext
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/random.h"  // for ColumnSampler, GlobalRandom
#include "../common/timer.h"
#include "../data/ellpack_page.cuh"
#include "../data/ellpack_page.h"
#include "constraints.cuh"
#include "driver.h"
#include "gpu_hist/evaluate_splits.cuh"
#include "gpu_hist/expand_entry.cuh"
#include "gpu_hist/feature_groups.cuh"
#include "gpu_hist/gradient_based_sampler.cuh"
#include "gpu_hist/histogram.cuh"
#include "gpu_hist/row_partitioner.cuh"
#include "hist/param.h"
#include "param.h"
#include "updater_gpu_common.cuh"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/span.h"
#include "xgboost/task.h"  // for ObjInfo
#include "xgboost/tree_model.h"

namespace xgboost::tree {
#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);
#endif  // !defined(GTEST_TEST)

/**
 * \struct  DeviceHistogramStorage
 *
 * \summary Data storage for node histograms on device. Automatically expands.
 *
 * \tparam GradientSumT      histogram entry type.
 * \tparam kStopGrowingSize  Do not grow beyond this size
 *
 * \author  Rory
 * \date    28/07/2018
 */
template <size_t kStopGrowingSize = 1 << 28>
class DeviceHistogramStorage {
 private:
  using GradientSumT = GradientPairInt64;
  /*! \brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  // Large buffer of zeroed memory, caches histograms
  dh::device_vector<typename GradientSumT::ValueT> data_;
  // If we run out of storage allocate one histogram at a time
  // in overflow. Not cached, overwritten when a new histogram
  // is requested
  dh::device_vector<typename GradientSumT::ValueT> overflow_;
  std::map<int, size_t> overflow_nidx_map_;
  int n_bins_;
  DeviceOrd device_id_;
  static constexpr size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2, "Number of items in gradient type should be 2.");

 public:
  // Start with about 16mb
  DeviceHistogramStorage() { data_.reserve(1 << 22); }
  void Init(DeviceOrd device_id, int n_bins) {
    this->n_bins_ = n_bins;
    this->device_id_ = device_id;
  }

  void Reset() {
    auto d_data = data_.data().get();
    dh::LaunchN(data_.size(), [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    nidx_map_.clear();
    overflow_nidx_map_.clear();
  }
  [[nodiscard]] bool HistogramExists(int nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend() ||
           overflow_nidx_map_.find(nidx) != overflow_nidx_map_.cend();
  }
  [[nodiscard]] int Bins() const { return n_bins_; }
  [[nodiscard]] size_t HistogramSize() const { return n_bins_ * kNumItemsInGradientSum; }
  dh::device_vector<typename GradientSumT::ValueT>& Data() { return data_; }

  void AllocateHistograms(const std::vector<int>& new_nidxs) {
    for (int nidx : new_nidxs) {
      CHECK(!HistogramExists(nidx));
    }
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize() * new_nidxs.size();
    if (used_size >= kStopGrowingSize) {
      // Use overflow
      // Delete previous entries
      overflow_nidx_map_.clear();
      overflow_.resize(HistogramSize() * new_nidxs.size());
      // Zero memory
      auto d_data = overflow_.data().get();
      dh::LaunchN(overflow_.size(),
                  [=] __device__(size_t idx) { d_data[idx] = 0.0; });
      // Append new histograms
      for (int nidx : new_nidxs) {
        overflow_nidx_map_[nidx] = overflow_nidx_map_.size() * HistogramSize();
      }
    } else {
      CHECK_GE(data_.size(), used_size);
      // Expand if necessary
      if (data_.size() < new_used_size) {
        data_.resize(std::max(data_.size() * 2, new_used_size));
      }
      // Append new histograms
      for (int nidx : new_nidxs) {
        nidx_map_[nidx] = nidx_map_.size() * HistogramSize();
      }
    }

    CHECK_GE(data_.size(), nidx_map_.size() * HistogramSize());
  }

  /**
   * \summary   Return pointer to histogram memory for a given node.
   * \param nidx    Tree node index.
   * \return    hist pointer.
   */
  common::Span<GradientSumT> GetNodeHistogram(int nidx) {
    CHECK(this->HistogramExists(nidx));

    if (nidx_map_.find(nidx) != nidx_map_.cend()) {
      // Fetch from normal cache
      auto ptr = data_.data().get() + nidx_map_.at(nidx);
      return {reinterpret_cast<GradientSumT*>(ptr), static_cast<std::size_t>(n_bins_)};
    } else {
      // Fetch from overflow
      auto ptr = overflow_.data().get() + overflow_nidx_map_.at(nidx);
      return {reinterpret_cast<GradientSumT*>(ptr), static_cast<std::size_t>(n_bins_)};
    }
  }
};

// Manage memory for a single GPU
struct GPUHistMakerDevice {
 private:
  GPUHistEvaluator evaluator_;
  Context const* ctx_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  MetaInfo const& info_;

 public:
  EllpackPageImpl const* page{nullptr};
  common::Span<FeatureType const> feature_types;

  std::unique_ptr<RowPartitioner> row_partitioner;
  DeviceHistogramStorage<> hist{};

  dh::device_vector<GradientPair> d_gpair;  // storage for gpair;
  common::Span<GradientPair> gpair;

  dh::device_vector<int> monotone_constraints;
  // node idx for each sample
  dh::device_vector<bst_node_t> positions;

  TrainParam param;

  std::unique_ptr<GradientQuantiser> quantiser;

  dh::PinnedMemory pinned;
  dh::PinnedMemory pinned2;

  common::Monitor monitor;
  FeatureInteractionConstraintDevice interaction_constraints;

  std::unique_ptr<GradientBasedSampler> sampler;

  std::unique_ptr<FeatureGroups> feature_groups;

  GPUHistMakerDevice(Context const* ctx, bool is_external_memory,
                     common::Span<FeatureType const> _feature_types, bst_idx_t _n_rows,
                     TrainParam _param, std::shared_ptr<common::ColumnSampler> column_sampler,
                     uint32_t n_features, BatchParam batch_param, MetaInfo const& info)
      : evaluator_{_param, n_features, ctx->Device()},
        ctx_(ctx),
        feature_types{_feature_types},
        param(std::move(_param)),
        column_sampler_(std::move(column_sampler)),
        interaction_constraints(param, n_features),
        info_{info} {
    sampler = std::make_unique<GradientBasedSampler>(ctx, _n_rows, batch_param, param.subsample,
                                                     param.sampling_method, is_external_memory);
    if (!param.monotone_constraints.empty()) {
      // Copy assigning an empty vector causes an exception in MSVC debug builds
      monotone_constraints = param.monotone_constraints;
    }

    CHECK(column_sampler_);
    monitor.Init(std::string("GPUHistMakerDevice") + ctx_->Device().Name());
  }

  ~GPUHistMakerDevice() = default;

  void InitFeatureGroupsOnce() {
    if (!feature_groups) {
      CHECK(page);
      feature_groups = std::make_unique<FeatureGroups>(page->Cuts(), page->is_dense,
                                                       dh::MaxSharedMemoryOptin(ctx_->Ordinal()),
                                                       sizeof(GradientPairPrecise));
    }
  }

  // Reset values for each update iteration
  void Reset(HostDeviceVector<GradientPair>* dh_gpair, DMatrix* dmat, int64_t num_columns) {
    auto const& info = dmat->Info();
    this->column_sampler_->Init(ctx_, num_columns, info.feature_weights.HostVector(),
                                param.colsample_bynode, param.colsample_bylevel,
                                param.colsample_bytree);
    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));

    this->interaction_constraints.Reset();

    if (d_gpair.size() != dh_gpair->Size()) {
      d_gpair.resize(dh_gpair->Size());
    }
    dh::safe_cuda(cudaMemcpyAsync(d_gpair.data().get(), dh_gpair->ConstDevicePointer(),
                                  dh_gpair->Size() * sizeof(GradientPair),
                                  cudaMemcpyDeviceToDevice));
    auto sample = sampler->Sample(ctx_, dh::ToSpan(d_gpair), dmat);
    page = sample.page;
    gpair = sample.gpair;

    this->evaluator_.Reset(page->Cuts(), feature_types, dmat->Info().num_col_, param,
                           dmat->Info().IsColumnSplit(), ctx_->Device());

    quantiser = std::make_unique<GradientQuantiser>(ctx_, this->gpair, dmat->Info());

    row_partitioner.reset();  // Release the device memory first before reallocating
    row_partitioner = std::make_unique<RowPartitioner>(ctx_->Device(), sample.sample_rows);

    // Init histogram
    hist.Init(ctx_->Device(), page->Cuts().TotalBins());
    hist.Reset();

    this->InitFeatureGroupsOnce();
  }

  GPUExpandEntry EvaluateRootSplit(GradientPairInt64 root_sum) {
    int nidx = RegTree::kRoot;
    GPUTrainingParam gpu_param(param);
    auto sampled_features = column_sampler_->GetFeatureSet(0);
    sampled_features->SetDevice(ctx_->Device());
    common::Span<bst_feature_t> feature_set =
        interaction_constraints.Query(sampled_features->DeviceSpan(), nidx);
    auto matrix = page->GetDeviceAccessor(ctx_->Device());
    EvaluateSplitInputs inputs{nidx, 0, root_sum, feature_set, hist.GetNodeHistogram(nidx)};
    EvaluateSplitSharedInputs shared_inputs{
        gpu_param,
        *quantiser,
        feature_types,
        matrix.feature_segments,
        matrix.gidx_fvalue_map,
        matrix.min_fvalue,
        matrix.is_dense && !collective::IsDistributed()
    };
    auto split = this->evaluator_.EvaluateSingleSplit(ctx_, inputs, shared_inputs);
    return split;
  }

  void EvaluateSplits(const std::vector<GPUExpandEntry>& candidates, const RegTree& tree,
                               common::Span<GPUExpandEntry> pinned_candidates_out) {
    if (candidates.empty()) return;
    dh::TemporaryArray<EvaluateSplitInputs> d_node_inputs(2 * candidates.size());
    dh::TemporaryArray<DeviceSplitCandidate> splits_out(2 * candidates.size());
    std::vector<bst_node_t> nidx(2 * candidates.size());
    auto h_node_inputs = pinned2.GetSpan<EvaluateSplitInputs>(2 * candidates.size());
    auto matrix = page->GetDeviceAccessor(ctx_->Device());
    EvaluateSplitSharedInputs shared_inputs{GPUTrainingParam{param}, *quantiser, feature_types,
                                            matrix.feature_segments, matrix.gidx_fvalue_map,
                                            matrix.min_fvalue,
                                            // is_dense represents the local data
                                            matrix.is_dense && !collective::IsDistributed()};
    dh::TemporaryArray<GPUExpandEntry> entries(2 * candidates.size());
    // Store the feature set ptrs so they dont go out of scope before the kernel is called
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> feature_sets;
    for (std::size_t i = 0; i < candidates.size(); i++) {
      auto candidate = candidates.at(i);
      int left_nidx = tree[candidate.nid].LeftChild();
      int right_nidx = tree[candidate.nid].RightChild();
      nidx[i * 2] = left_nidx;
      nidx[i * 2 + 1] = right_nidx;
      auto left_sampled_features = column_sampler_->GetFeatureSet(tree.GetDepth(left_nidx));
      left_sampled_features->SetDevice(ctx_->Device());
      feature_sets.emplace_back(left_sampled_features);
      common::Span<bst_feature_t> left_feature_set =
          interaction_constraints.Query(left_sampled_features->DeviceSpan(), left_nidx);
      auto right_sampled_features = column_sampler_->GetFeatureSet(tree.GetDepth(right_nidx));
      right_sampled_features->SetDevice(ctx_->Device());
      feature_sets.emplace_back(right_sampled_features);
      common::Span<bst_feature_t> right_feature_set =
          interaction_constraints.Query(right_sampled_features->DeviceSpan(),
                                        right_nidx);
      h_node_inputs[i * 2] = {left_nidx, candidate.depth + 1,
                              candidate.split.left_sum, left_feature_set,
                              hist.GetNodeHistogram(left_nidx)};
      h_node_inputs[i * 2 + 1] = {right_nidx, candidate.depth + 1,
                                  candidate.split.right_sum, right_feature_set,
                                  hist.GetNodeHistogram(right_nidx)};
    }
    bst_feature_t max_active_features = 0;
    for (auto input : h_node_inputs) {
      max_active_features =
          std::max(max_active_features, static_cast<bst_feature_t>(input.feature_set.size()));
    }
    dh::safe_cuda(cudaMemcpyAsync(
        d_node_inputs.data().get(), h_node_inputs.data(),
        h_node_inputs.size() * sizeof(EvaluateSplitInputs), cudaMemcpyDefault));

    this->evaluator_.EvaluateSplits(ctx_, nidx, max_active_features, dh::ToSpan(d_node_inputs),
                                    shared_inputs, dh::ToSpan(entries));
    dh::safe_cuda(cudaMemcpyAsync(pinned_candidates_out.data(),
                                  entries.data().get(), sizeof(GPUExpandEntry) * entries.size(),
                                  cudaMemcpyDeviceToHost));
    dh::DefaultStream().Sync();
  }

  void BuildHist(int nidx) {
    auto d_node_hist = hist.GetNodeHistogram(nidx);
    auto d_ridx = row_partitioner->GetRows(nidx);
    BuildGradientHistogram(ctx_->CUDACtx(), page->GetDeviceAccessor(ctx_->Device()),
                           feature_groups->DeviceAccessor(ctx_->Device()), gpair, d_ridx,
                           d_node_hist, *quantiser);
  }

  // Attempt to do subtraction trick
  // return true if succeeded
  bool SubtractionTrick(int nidx_parent, int nidx_histogram, int nidx_subtraction) {
    if (!hist.HistogramExists(nidx_histogram) || !hist.HistogramExists(nidx_parent)) {
      return false;
    }
    auto d_node_hist_parent = hist.GetNodeHistogram(nidx_parent);
    auto d_node_hist_histogram = hist.GetNodeHistogram(nidx_histogram);
    auto d_node_hist_subtraction = hist.GetNodeHistogram(nidx_subtraction);

    dh::LaunchN(page->Cuts().TotalBins(), [=] __device__(size_t idx) {
      d_node_hist_subtraction[idx] =
          d_node_hist_parent[idx] - d_node_hist_histogram[idx];
    });
    return true;
  }

  // Extra data for each node that is passed
  // to the update position function
  struct NodeSplitData {
    RegTree::Node split_node;
    FeatureType split_type;
    common::KCatBitField node_cats;
  };

  void UpdatePositionColumnSplit(EllpackDeviceAccessor d_matrix,
                                 std::vector<NodeSplitData> const& split_data,
                                 std::vector<bst_node_t> const& nidx,
                                 std::vector<bst_node_t> const& left_nidx,
                                 std::vector<bst_node_t> const& right_nidx) {
    auto const num_candidates = split_data.size();

    using BitVector = LBitField64;
    using BitType = BitVector::value_type;
    auto const size = BitVector::ComputeStorageSize(d_matrix.n_rows * num_candidates);
    dh::TemporaryArray<BitType> decision_storage(size, 0);
    dh::TemporaryArray<BitType> missing_storage(size, 0);
    BitVector decision_bits{dh::ToSpan(decision_storage)};
    BitVector missing_bits{dh::ToSpan(missing_storage)};

    dh::TemporaryArray<NodeSplitData> split_data_storage(num_candidates);
    dh::safe_cuda(cudaMemcpyAsync(split_data_storage.data().get(), split_data.data(),
                                  num_candidates * sizeof(NodeSplitData), cudaMemcpyDefault));
    auto d_split_data = dh::ToSpan(split_data_storage);

    dh::LaunchN(d_matrix.n_rows, [=] __device__(std::size_t ridx) mutable {
      for (auto i = 0; i < num_candidates; i++) {
        auto const& data = d_split_data[i];
        auto const cut_value = d_matrix.GetFvalue(ridx, data.split_node.SplitIndex());
        if (isnan(cut_value)) {
          missing_bits.Set(ridx * num_candidates + i);
        } else {
          bool go_left;
          if (data.split_type == FeatureType::kCategorical) {
            go_left = common::Decision(data.node_cats.Bits(), cut_value);
          } else {
            go_left = cut_value <= data.split_node.SplitCond();
          }
          if (go_left) {
            decision_bits.Set(ridx * num_candidates + i);
          }
        }
      }
    });

    auto rc = collective::Success() << [&] {
      return collective::Allreduce(
          ctx_, linalg::MakeTensorView(ctx_, dh::ToSpan(decision_storage), decision_storage.size()),
          collective::Op::kBitwiseOR);
    } << [&] {
      return collective::Allreduce(
          ctx_, linalg::MakeTensorView(ctx_, dh::ToSpan(missing_storage), missing_storage.size()),
          collective::Op::kBitwiseAND);
    };
    collective::SafeColl(rc);

    row_partitioner->UpdatePositionBatch(
        nidx, left_nidx, right_nidx, split_data,
        [=] __device__(bst_uint ridx, int split_index, NodeSplitData const& data) {
          auto const index = ridx * num_candidates + split_index;
          bool go_left;
          if (missing_bits.Check(index)) {
            go_left = data.split_node.DefaultLeft();
          } else {
            go_left = decision_bits.Check(index);
          }
          return go_left;
        });
  }

  void UpdatePosition(std::vector<GPUExpandEntry> const& candidates, RegTree* p_tree) {
    if (candidates.empty()) {
      return;
    }

    std::vector<bst_node_t> nidx(candidates.size());
    std::vector<bst_node_t> left_nidx(candidates.size());
    std::vector<bst_node_t> right_nidx(candidates.size());
    std::vector<NodeSplitData> split_data(candidates.size());

    for (size_t i = 0; i < candidates.size(); i++) {
      auto const& e = candidates[i];
      RegTree::Node split_node = (*p_tree)[e.nid];
      auto split_type = p_tree->NodeSplitType(e.nid);
      nidx.at(i) = e.nid;
      left_nidx.at(i) = split_node.LeftChild();
      right_nidx.at(i) = split_node.RightChild();
      split_data.at(i) = NodeSplitData{split_node, split_type, evaluator_.GetDeviceNodeCats(e.nid)};

      CHECK_EQ(split_type == FeatureType::kCategorical, e.split.is_cat);
    }

    auto d_matrix = page->GetDeviceAccessor(ctx_->Device());

    if (info_.IsColumnSplit()) {
      UpdatePositionColumnSplit(d_matrix, split_data, nidx, left_nidx, right_nidx);
      return;
    }

    row_partitioner->UpdatePositionBatch(
        nidx, left_nidx, right_nidx, split_data,
        [=] __device__(bst_uint ridx, int split_index, const NodeSplitData& data) {
          // given a row index, returns the node id it belongs to
          float cut_value = d_matrix.GetFvalue(ridx, data.split_node.SplitIndex());
          // Missing value
          bool go_left = true;
          if (isnan(cut_value)) {
            go_left = data.split_node.DefaultLeft();
          } else {
            if (data.split_type == FeatureType::kCategorical) {
              go_left = common::Decision(data.node_cats.Bits(), cut_value);
            } else {
              go_left = cut_value <= data.split_node.SplitCond();
            }
          }
          return go_left;
        });
  }

  // After tree update is finished, update the position of all training
  // instances to their final leaf. This information is used later to update the
  // prediction cache
  void FinalisePosition(RegTree const* p_tree, DMatrix* p_fmat, ObjInfo task,
                        HostDeviceVector<bst_node_t>* p_out_position) {
    // Prediction cache will not be used with external memory
    if (!p_fmat->SingleColBlock()) {
      if (task.UpdateTreeLeaf()) {
        LOG(FATAL) << "Current objective function can not be used with external memory.";
      }
      p_out_position->Resize(0);
      positions.clear();
      return;
    }

    dh::TemporaryArray<RegTree::Node> d_nodes(p_tree->GetNodes().size());
    dh::safe_cuda(cudaMemcpyAsync(d_nodes.data().get(), p_tree->GetNodes().data(),
                                  d_nodes.size() * sizeof(RegTree::Node),
                                  cudaMemcpyHostToDevice));
    auto const& h_split_types = p_tree->GetSplitTypes();
    auto const& categories = p_tree->GetSplitCategories();
    auto const& categories_segments = p_tree->GetSplitCategoriesPtr();

    dh::caching_device_vector<FeatureType> d_split_types;
    dh::caching_device_vector<uint32_t> d_categories;
    dh::caching_device_vector<RegTree::CategoricalSplitMatrix::Segment> d_categories_segments;

    if (!categories.empty()) {
      dh::CopyToD(h_split_types, &d_split_types);
      dh::CopyToD(categories, &d_categories);
      dh::CopyToD(categories_segments, &d_categories_segments);
    }

    FinalisePositionInPage(page, dh::ToSpan(d_nodes), dh::ToSpan(d_split_types),
                           dh::ToSpan(d_categories), dh::ToSpan(d_categories_segments),
                           p_out_position);
  }

  void FinalisePositionInPage(
      EllpackPageImpl const* page, const common::Span<RegTree::Node> d_nodes,
      common::Span<FeatureType const> d_feature_types, common::Span<uint32_t const> categories,
      common::Span<RegTree::CategoricalSplitMatrix::Segment> categories_segments,
      HostDeviceVector<bst_node_t>* p_out_position) {
    auto d_matrix = page->GetDeviceAccessor(ctx_->Device());
    auto d_gpair = this->gpair;
    p_out_position->SetDevice(ctx_->Device());
    p_out_position->Resize(row_partitioner->GetRows().size());

    auto new_position_op = [=] __device__(size_t row_id, int position) {
      // What happens if user prune the tree?
      if (!d_matrix.IsInRange(row_id)) {
        return RowPartitioner::kIgnoredTreePosition;
      }
      auto node = d_nodes[position];

      while (!node.IsLeaf()) {
        bst_float element = d_matrix.GetFvalue(row_id, node.SplitIndex());
        // Missing value
        if (isnan(element)) {
          position = node.DefaultChild();
        } else {
          bool go_left = true;
          if (common::IsCat(d_feature_types, position)) {
            auto node_cats = categories.subspan(categories_segments[position].beg,
                                                categories_segments[position].size);
            go_left = common::Decision(node_cats, element);
          } else {
            go_left = element <= node.SplitCond();
          }
          if (go_left) {
            position = node.LeftChild();
          } else {
            position = node.RightChild();
          }
        }

        node = d_nodes[position];
      }

      return position;
    };  // NOLINT

    auto d_out_position = p_out_position->DeviceSpan();
    row_partitioner->FinalisePosition(d_out_position, new_position_op);

    auto s_position = p_out_position->ConstDeviceSpan();
    positions.resize(s_position.size());
    dh::safe_cuda(cudaMemcpyAsync(positions.data().get(), s_position.data(),
                                  s_position.size_bytes(), cudaMemcpyDeviceToDevice,
                                  ctx_->CUDACtx()->Stream()));

    dh::LaunchN(row_partitioner->GetRows().size(), [=] __device__(size_t idx) {
      bst_node_t position = d_out_position[idx];
      bool is_row_sampled = d_gpair[idx].GetHess() - .0f == 0.f;
      d_out_position[idx] = is_row_sampled ? ~position : position;
    });
  }

  bool UpdatePredictionCache(linalg::MatrixView<float> out_preds_d, RegTree const* p_tree) {
    if (positions.empty()) {
      return false;
    }

    CHECK(p_tree);
    CHECK(out_preds_d.Device().IsCUDA());
    CHECK_EQ(out_preds_d.Device().ordinal, ctx_->Ordinal());

    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    auto d_position = dh::ToSpan(positions);
    CHECK_EQ(out_preds_d.Size(), d_position.size());

    auto const& h_nodes = p_tree->GetNodes();
    dh::caching_device_vector<RegTree::Node> nodes(h_nodes.size());
    dh::safe_cuda(cudaMemcpyAsync(nodes.data().get(), h_nodes.data(),
                                  h_nodes.size() * sizeof(RegTree::Node), cudaMemcpyHostToDevice,
                                  ctx_->CUDACtx()->Stream()));
    auto d_nodes = dh::ToSpan(nodes);
    CHECK_EQ(out_preds_d.Shape(1), 1);
    dh::LaunchN(d_position.size(), ctx_->CUDACtx()->Stream(),
                [=] XGBOOST_DEVICE(std::size_t idx) mutable {
                  bst_node_t nidx = d_position[idx];
                  auto weight = d_nodes[nidx].LeafValue();
                  out_preds_d(idx, 0) += weight;
                });
    return true;
  }

  // num histograms is the number of contiguous histograms in memory to reduce over
  void AllReduceHist(int nidx, int num_histograms) {
    monitor.Start("AllReduce");
    auto d_node_hist = hist.GetNodeHistogram(nidx).data();
    using ReduceT = typename std::remove_pointer<decltype(d_node_hist)>::type::ValueT;
    auto rc = collective::GlobalSum(
        ctx_, info_,
        linalg::MakeVec(reinterpret_cast<ReduceT*>(d_node_hist),
                        page->Cuts().TotalBins() * 2 * num_histograms, ctx_->Device()));
    SafeColl(rc);

    monitor.Stop("AllReduce");
  }

  /**
   * \brief Build GPU local histograms for the left and right child of some parent node
   */
  void BuildHistLeftRight(std::vector<GPUExpandEntry> const& candidates, const RegTree& tree) {
    if (candidates.empty()) return;
    // Some nodes we will manually compute histograms
    // others we will do by subtraction
    std::vector<int> hist_nidx;
    std::vector<int> subtraction_nidx;
    for (auto& e : candidates) {
      // Decide whether to build the left histogram or right histogram
      // Use sum of Hessian as a heuristic to select node with fewest training instances
      bool fewer_right = e.split.right_sum.GetQuantisedHess() < e.split.left_sum.GetQuantisedHess();
      if (fewer_right) {
        hist_nidx.emplace_back(tree[e.nid].RightChild());
        subtraction_nidx.emplace_back(tree[e.nid].LeftChild());
      } else {
        hist_nidx.emplace_back(tree[e.nid].LeftChild());
        subtraction_nidx.emplace_back(tree[e.nid].RightChild());
      }
    }
    std::vector<int> all_new = hist_nidx;
    all_new.insert(all_new.end(), subtraction_nidx.begin(), subtraction_nidx.end());
    // Allocate the histograms
    // Guaranteed contiguous memory
    hist.AllocateHistograms(all_new);

    for (auto nidx : hist_nidx) {
      this->BuildHist(nidx);
    }

    // Reduce all in one go
    // This gives much better latency in a distributed setting
    // when processing a large batch
    this->AllReduceHist(hist_nidx.at(0), hist_nidx.size());

    for (size_t i = 0; i < subtraction_nidx.size(); i++) {
      auto build_hist_nidx = hist_nidx.at(i);
      auto subtraction_trick_nidx = subtraction_nidx.at(i);
      auto parent_nidx = candidates.at(i).nid;

      if (!this->SubtractionTrick(parent_nidx, build_hist_nidx, subtraction_trick_nidx)) {
        // Calculate other histogram manually
        this->BuildHist(subtraction_trick_nidx);
        this->AllReduceHist(subtraction_trick_nidx, 1);
      }
    }
  }

  void ApplySplit(const GPUExpandEntry& candidate, RegTree* p_tree) {
    RegTree& tree = *p_tree;

    // Sanity check - have we created a leaf with no training instances?
    if (!collective::IsDistributed() && row_partitioner) {
      CHECK(row_partitioner->GetRows(candidate.nid).size() > 0)
          << "No training instances in this leaf!";
    }

    auto base_weight = candidate.base_weight;
    auto left_weight = candidate.left_weight * param.learning_rate;
    auto right_weight = candidate.right_weight * param.learning_rate;
    auto parent_hess = quantiser
                           ->ToFloatingPoint(candidate.split.left_sum +
                                             candidate.split.right_sum)
                           .GetHess();
    auto left_hess =
        quantiser->ToFloatingPoint(candidate.split.left_sum).GetHess();
    auto right_hess =
        quantiser->ToFloatingPoint(candidate.split.right_sum).GetHess();

    auto is_cat = candidate.split.is_cat;
    if (is_cat) {
      // should be set to nan in evaluation split.
      CHECK(common::CheckNAN(candidate.split.fvalue));
      std::vector<common::CatBitField::value_type> split_cats;

      auto h_cats = this->evaluator_.GetHostNodeCats(candidate.nid);
      auto n_bins_feature = page->Cuts().FeatureBins(candidate.split.findex);
      split_cats.resize(common::CatBitField::ComputeStorageSize(n_bins_feature), 0);
      CHECK_LE(split_cats.size(), h_cats.size());
      std::copy(h_cats.data(), h_cats.data() + split_cats.size(), split_cats.data());

      tree.ExpandCategorical(
          candidate.nid, candidate.split.findex, split_cats, candidate.split.dir == kLeftDir,
          base_weight, left_weight, right_weight, candidate.split.loss_chg, parent_hess,
          left_hess, right_hess);
    } else {
      CHECK(!common::CheckNAN(candidate.split.fvalue));
      tree.ExpandNode(candidate.nid, candidate.split.findex, candidate.split.fvalue,
                      candidate.split.dir == kLeftDir, base_weight, left_weight, right_weight,
                      candidate.split.loss_chg, parent_hess,
          left_hess, right_hess);
    }
    evaluator_.ApplyTreeSplit(candidate, p_tree);

    const auto& parent = tree[candidate.nid];
    interaction_constraints.Split(candidate.nid, parent.SplitIndex(), parent.LeftChild(),
                                  parent.RightChild());
  }

  GPUExpandEntry InitRoot(RegTree* p_tree) {
    constexpr bst_node_t kRootNIdx = 0;
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto quantiser = *this->quantiser;
    auto gpair_it = dh::MakeTransformIterator<GradientPairInt64>(
        dh::tbegin(gpair), [=] __device__(auto const &gpair) {
          return quantiser.ToFixedPoint(gpair);
        });
    GradientPairInt64 root_sum_quantised =
        dh::Reduce(ctx_->CUDACtx()->CTP(), gpair_it, gpair_it + gpair.size(),
                   GradientPairInt64{}, thrust::plus<GradientPairInt64>{});
    using ReduceT = typename decltype(root_sum_quantised)::ValueT;
    auto rc = collective::GlobalSum(
        ctx_, info_, linalg::MakeVec(reinterpret_cast<ReduceT*>(&root_sum_quantised), 2));
    collective::SafeColl(rc);

    hist.AllocateHistograms({kRootNIdx});
    this->BuildHist(kRootNIdx);
    this->AllReduceHist(kRootNIdx, 1);

    // Remember root stats
    auto root_sum = quantiser.ToFloatingPoint(root_sum_quantised);
    p_tree->Stat(kRootNIdx).sum_hess = root_sum.GetHess();
    auto weight = CalcWeight(param, root_sum);
    p_tree->Stat(kRootNIdx).base_weight = weight;
    (*p_tree)[kRootNIdx].SetLeaf(param.learning_rate * weight);

    // Generate first split
    auto root_entry = this->EvaluateRootSplit(root_sum_quantised);
    return root_entry;
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat, ObjInfo const* task,
                  RegTree* p_tree, HostDeviceVector<bst_node_t>* p_out_position) {
    auto& tree = *p_tree;
    // Process maximum 32 nodes at a time
    Driver<GPUExpandEntry> driver(param, 32);

    monitor.Start("Reset");
    this->Reset(gpair_all, p_fmat, p_fmat->Info().num_col_);
    monitor.Stop("Reset");

    monitor.Start("InitRoot");
    driver.Push({this->InitRoot(p_tree)});
    monitor.Stop("InitRoot");

    // The set of leaves that can be expanded asynchronously
    auto expand_set = driver.Pop();
    while (!expand_set.empty()) {
      for (auto& candidate : expand_set) {
        this->ApplySplit(candidate, p_tree);
      }
      // Get the candidates we are allowed to expand further
      // e.g. We do not bother further processing nodes whose children are beyond max depth
      std::vector<GPUExpandEntry> filtered_expand_set;
      std::copy_if(expand_set.begin(), expand_set.end(), std::back_inserter(filtered_expand_set),
                   [&](const auto& e) { return driver.IsChildValid(e); });

      auto new_candidates =
          pinned.GetSpan<GPUExpandEntry>(filtered_expand_set.size() * 2, GPUExpandEntry());

      monitor.Start("UpdatePosition");
      // Update position is only run when child is valid, instead of right after apply
      // split (as in approx tree method).  Hense we have the finalise position call
      // in GPU Hist.
      this->UpdatePosition(filtered_expand_set, p_tree);
      monitor.Stop("UpdatePosition");

      monitor.Start("BuildHist");
      this->BuildHistLeftRight(filtered_expand_set, tree);
      monitor.Stop("BuildHist");

      monitor.Start("EvaluateSplits");
      this->EvaluateSplits(filtered_expand_set, *p_tree, new_candidates);
      monitor.Stop("EvaluateSplits");
      dh::DefaultStream().Sync();
      driver.Push(new_candidates.begin(), new_candidates.end());
      expand_set = driver.Pop();
    }

    monitor.Start("FinalisePosition");
    this->FinalisePosition(p_tree, p_fmat, *task, p_out_position);
    monitor.Stop("FinalisePosition");
  }
};

class GPUHistMaker : public TreeUpdater {
  using GradientSumT = GradientPairPrecise;

 public:
  explicit GPUHistMaker(Context const* ctx, ObjInfo const* task) : TreeUpdater(ctx), task_{task} {};
  void Configure(const Args& args) override {
    // Used in test to count how many configurations are performed
    LOG(DEBUG) << "[GPU Hist]: Configure";
    hist_maker_param_.UpdateAllowUnknown(args);
    dh::CheckComputeCapability();
    initialised_ = false;

    monitor_.Init("updater_gpu_hist");
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("hist_train_param"), &this->hist_maker_param_);
    initialised_ = false;
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["hist_train_param"] = ToJson(hist_maker_param_);
  }

  ~GPUHistMaker() {  // NOLINT
    dh::GlobalMemoryLogger().Log();
  }

  void Update(TrainParam const* param, linalg::Matrix<GradientPair>* gpair, DMatrix* dmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override {
    monitor_.Start("Update");

    CHECK_EQ(gpair->Shape(1), 1) << MTNotImplemented();
    auto gpair_hdv = gpair->Data();
    // build tree
    try {
      std::size_t t_idx{0};
      for (xgboost::RegTree* tree : trees) {
        this->UpdateTree(param, gpair_hdv, dmat, tree, &out_position[t_idx]);
        this->hist_maker_param_.CheckTreesSynchronized(ctx_, tree);
        ++t_idx;
      }
      dh::safe_cuda(cudaGetLastError());
    } catch (const std::exception& e) {
      LOG(FATAL) << "Exception in gpu_hist: " << e.what() << std::endl;
    }
    monitor_.Stop("Update");
  }

  void InitDataOnce(TrainParam const* param, DMatrix* dmat) {
    CHECK_GE(ctx_->Ordinal(), 0) << "Must have at least one device";
    info_ = &dmat->Info();

    this->column_sampler_ = common::MakeColumnSampler(ctx_);

    auto batch_param = BatchParam{param->max_bin, TrainParam::DftSparseThreshold()};
    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    info_->feature_types.SetDevice(ctx_->Device());
    maker = std::make_unique<GPUHistMakerDevice>(
        ctx_, !dmat->SingleColBlock(), info_->feature_types.ConstDeviceSpan(), info_->num_row_,
        *param, column_sampler_, info_->num_col_, batch_param, dmat->Info());

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(TrainParam const* param, DMatrix* dmat, RegTree const* p_tree) {
    if (!initialised_) {
      monitor_.Start("InitDataOnce");
      this->InitDataOnce(param, dmat);
      monitor_.Stop("InitDataOnce");
    }
    p_last_tree_ = p_tree;
    CHECK(hist_maker_param_.GetInitialised());
  }

  void UpdateTree(TrainParam const* param, HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat,
                  RegTree* p_tree, HostDeviceVector<bst_node_t>* p_out_position) {
    monitor_.Start("InitData");
    this->InitData(param, p_fmat, p_tree);
    monitor_.Stop("InitData");

    gpair->SetDevice(ctx_->Device());
    maker->UpdateTree(gpair, p_fmat, task_, p_tree, p_out_position);
  }

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::MatrixView<bst_float> p_out_preds) override {
    if (maker == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.Start("UpdatePredictionCache");
    bool result = maker->UpdatePredictionCache(p_out_preds, p_last_tree_);
    monitor_.Stop("UpdatePredictionCache");
    return result;
  }

  MetaInfo* info_{};  // NOLINT

  std::unique_ptr<GPUHistMakerDevice> maker;  // NOLINT

  [[nodiscard]] char const* Name() const override { return "grow_gpu_hist"; }
  [[nodiscard]] bool HasNodePosition() const override { return true; }

 private:
  bool initialised_{false};

  HistMakerTrainParam hist_maker_param_;

  DMatrix* p_last_fmat_{nullptr};
  RegTree const* p_last_tree_{nullptr};
  ObjInfo const* task_{nullptr};

  common::Monitor monitor_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
};

#if !defined(GTEST_TEST)
XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([](Context const* ctx, ObjInfo const* task) {
      return new GPUHistMaker(ctx, task);
    });
#endif  // !defined(GTEST_TEST)

class GPUGlobalApproxMaker : public TreeUpdater {
 public:
  explicit GPUGlobalApproxMaker(Context const* ctx, ObjInfo const* task)
      : TreeUpdater(ctx), task_{task} {};
  void Configure(Args const& args) override {
    // Used in test to count how many configurations are performed
    LOG(DEBUG) << "[GPU Approx]: Configure";
    hist_maker_param_.UpdateAllowUnknown(args);
    if (hist_maker_param_.max_cached_hist_node != HistMakerTrainParam::DefaultNodes()) {
      LOG(WARNING) << "The `max_cached_hist_node` is ignored in GPU.";
    }
    dh::CheckComputeCapability();
    initialised_ = false;

    monitor_.Init(this->Name());
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("hist_train_param"), &this->hist_maker_param_);
    initialised_ = false;
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["hist_train_param"] = ToJson(hist_maker_param_);
  }
  ~GPUGlobalApproxMaker() override { dh::GlobalMemoryLogger().Log(); }

  void Update(TrainParam const* param, linalg::Matrix<GradientPair>* gpair, DMatrix* p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override {
    monitor_.Start("Update");

    this->InitDataOnce(p_fmat);
    // build tree
    hess_.resize(gpair->Size());
    auto hess = dh::ToSpan(hess_);

    gpair->SetDevice(ctx_->Device());
    auto d_gpair = gpair->Data()->ConstDeviceSpan();
    auto cuctx = ctx_->CUDACtx();
    thrust::transform(cuctx->CTP(), dh::tcbegin(d_gpair), dh::tcend(d_gpair), dh::tbegin(hess),
                      [=] XGBOOST_DEVICE(GradientPair const& g) { return g.GetHess(); });

    auto const& info = p_fmat->Info();
    info.feature_types.SetDevice(ctx_->Device());
    auto batch = BatchParam{param->max_bin, hess, !task_->const_hess};
    maker_ = std::make_unique<GPUHistMakerDevice>(
        ctx_, !p_fmat->SingleColBlock(), info.feature_types.ConstDeviceSpan(), info.num_row_,
        *param, column_sampler_, info.num_col_, batch, p_fmat->Info());

    std::size_t t_idx{0};
    for (xgboost::RegTree* tree : trees) {
      this->UpdateTree(gpair->Data(), p_fmat, tree, &out_position[t_idx]);
      this->hist_maker_param_.CheckTreesSynchronized(ctx_, tree);
      ++t_idx;
    }

    monitor_.Stop("Update");
  }

  void InitDataOnce(DMatrix* p_fmat) {
    if (this->initialised_) {
      return;
    }

    monitor_.Start(__func__);
    CHECK(ctx_->IsCUDA()) << error::InvalidCUDAOrdinal();
    this->column_sampler_ = common::MakeColumnSampler(ctx_);

    p_last_fmat_ = p_fmat;
    initialised_ = true;
    monitor_.Stop(__func__);
  }

  void InitData(DMatrix* p_fmat, RegTree const* p_tree) {
    this->InitDataOnce(p_fmat);
    p_last_tree_ = p_tree;
    CHECK(hist_maker_param_.GetInitialised());
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat, RegTree* p_tree,
                  HostDeviceVector<bst_node_t>* p_out_position) {
    monitor_.Start("InitData");
    this->InitData(p_fmat, p_tree);
    monitor_.Stop("InitData");

    gpair->SetDevice(ctx_->Device());
    maker_->UpdateTree(gpair, p_fmat, task_, p_tree, p_out_position);
  }

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::MatrixView<bst_float> p_out_preds) override {
    if (maker_ == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.Start("UpdatePredictionCache");
    bool result = maker_->UpdatePredictionCache(p_out_preds, p_last_tree_);
    monitor_.Stop("UpdatePredictionCache");
    return result;
  }

  [[nodiscard]] char const* Name() const override { return "grow_gpu_approx"; }
  [[nodiscard]] bool HasNodePosition() const override { return true; }

 private:
  bool initialised_{false};

  HistMakerTrainParam hist_maker_param_;
  dh::device_vector<float> hess_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  std::unique_ptr<GPUHistMakerDevice> maker_;

  DMatrix* p_last_fmat_{nullptr};
  RegTree const* p_last_tree_{nullptr};
  ObjInfo const* task_{nullptr};

  common::Monitor monitor_;
};

#if !defined(GTEST_TEST)
XGBOOST_REGISTER_TREE_UPDATER(GPUApproxMaker, "grow_gpu_approx")
    .describe("Grow tree with GPU.")
    .set_body([](Context const* ctx, ObjInfo const* task) {
      return new GPUGlobalApproxMaker(ctx, task);
    });
#endif  // !defined(GTEST_TEST)
}  // namespace xgboost::tree
