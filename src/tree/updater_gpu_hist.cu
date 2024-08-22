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
#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../common/cuda_rt_utils.h"   // for CheckComputeCapability
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
#include "sample_position.h"       // for SamplePosition
#include "updater_gpu_common.cuh"  // for HistBatch
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/span.h"
#include "xgboost/task.h"  // for ObjInfo
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

namespace xgboost::tree {
DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);

using cuda_impl::ApproxBatch;
using cuda_impl::HistBatch;

// GPU tree updater implementation.
struct GPUHistMakerDevice {
 private:
  GPUHistEvaluator evaluator_;
  Context const* ctx_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  MetaInfo const& info_;

  DeviceHistogramBuilder histogram_;
  // node idx for each sample
  dh::device_vector<bst_node_t> positions_;
  std::unique_ptr<RowPartitioner> row_partitioner_;

 public:
  // Extra data for each node that is passed to the update position function
  struct NodeSplitData {
    RegTree::Node split_node;
    FeatureType split_type;
    common::KCatBitField node_cats;
  };
  static_assert(std::is_trivially_copyable_v<NodeSplitData>);

 public:
  EllpackPageImpl const* page{nullptr};
  common::Span<FeatureType const> feature_types;

  DeviceHistogramStorage<> hist{};

  dh::device_vector<GradientPair> d_gpair;  // storage for gpair;
  common::Span<GradientPair> gpair;

  dh::device_vector<int> monotone_constraints;

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

    if (!row_partitioner_) {
      row_partitioner_ = std::make_unique<RowPartitioner>();
    }
    row_partitioner_->Reset(ctx_, sample.sample_rows, page->base_rowid);
    CHECK_EQ(page->base_rowid, 0);

    // Init histogram
    hist.Init(ctx_->Device(), page->Cuts().TotalBins());
    hist.Reset(ctx_);

    this->InitFeatureGroupsOnce();

    this->histogram_.Reset(ctx_, feature_groups->DeviceAccessor(ctx_->Device()), false);
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
    if (candidates.empty()) {
      return;
    }
    this->monitor.Start(__func__);
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
    this->monitor.Stop(__func__);
  }

  void BuildHist(int nidx) {
    auto d_node_hist = hist.GetNodeHistogram(nidx);
    auto d_ridx = row_partitioner_->GetRows(nidx);
    this->histogram_.BuildHistogram(ctx_->CUDACtx(), page->GetDeviceAccessor(ctx_->Device()),
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

    row_partitioner_->UpdatePositionBatch(
        nidx, left_nidx, right_nidx, split_data,
        [=] __device__(bst_uint ridx, int nidx_in_batch, NodeSplitData const& data) {
          auto const index = ridx * num_candidates + nidx_in_batch;
          bool go_left;
          if (missing_bits.Check(index)) {
            go_left = data.split_node.DefaultLeft();
          } else {
            go_left = decision_bits.Check(index);
          }
          return go_left;
        });
  }

  struct GoLeftOp {
    EllpackDeviceAccessor d_matrix;

    __device__ bool operator()(cuda_impl::RowIndexT ridx, NodeSplitData const& data) const {
      RegTree::Node const& node = data.split_node;
      // given a row index, returns the node id it belongs to
      float cut_value = d_matrix.GetFvalue(ridx, node.SplitIndex());
      // Missing value
      bool go_left = true;
      if (isnan(cut_value)) {
        go_left = node.DefaultLeft();
      } else {
        if (data.split_type == FeatureType::kCategorical) {
          go_left = common::Decision(data.node_cats.Bits(), cut_value);
        } else {
          go_left = cut_value <= node.SplitCond();
        }
      }
      return go_left;
    }
  };

  void UpdatePosition(std::vector<GPUExpandEntry> const& candidates, RegTree* p_tree) {
    if (candidates.empty()) {
      return;
    }

    monitor.Start(__func__);

    std::vector<bst_node_t> nidx(candidates.size());
    std::vector<bst_node_t> left_nidx(candidates.size());
    std::vector<bst_node_t> right_nidx(candidates.size());
    std::vector<NodeSplitData> split_data(candidates.size());

    for (size_t i = 0; i < candidates.size(); i++) {
      auto const& e = candidates[i];
      RegTree::Node const& split_node = (*p_tree)[e.nid];
      auto split_type = p_tree->NodeSplitType(e.nid);
      nidx[i] = e.nid;
      left_nidx[i] = split_node.LeftChild();
      right_nidx[i] = split_node.RightChild();
      split_data[i] = NodeSplitData{split_node, split_type, evaluator_.GetDeviceNodeCats(e.nid)};

      CHECK_EQ(split_type == FeatureType::kCategorical, e.split.is_cat);
    }

    auto d_matrix = page->GetDeviceAccessor(ctx_->Device());

    if (info_.IsColumnSplit()) {
      UpdatePositionColumnSplit(d_matrix, split_data, nidx, left_nidx, right_nidx);
      monitor.Stop(__func__);
      return;
    }
    auto go_left = GoLeftOp{d_matrix};
    row_partitioner_->UpdatePositionBatch(
        nidx, left_nidx, right_nidx, split_data,
        [=] __device__(cuda_impl::RowIndexT ridx, int /*nidx_in_batch*/,
                       const NodeSplitData& data) { return go_left(ridx, data); });
    monitor.Stop(__func__);
  }

  // After tree update is finished, update the position of all training
  // instances to their final leaf. This information is used later to update the
  // prediction cache
  void FinalisePosition(RegTree const* p_tree, DMatrix* p_fmat, ObjInfo task,
                        HostDeviceVector<bst_node_t>* p_out_position) {
    if (!p_fmat->SingleColBlock() && task.UpdateTreeLeaf()) {
      LOG(FATAL) << "Current objective function can not be used with external memory.";
    }
    if (p_fmat->Info().num_row_ != row_partitioner_->GetRows().size()) {
      // Subsampling with external memory. Not supported.
      p_out_position->Resize(0);
      positions_.clear();
      return;
    }

    p_out_position->SetDevice(ctx_->Device());
    p_out_position->Resize(row_partitioner_->GetRows().size());
    auto d_out_position = p_out_position->DeviceSpan();

    auto d_gpair = this->gpair;
    auto encode_op = [=] __device__(bst_idx_t row_id, bst_node_t nidx) {
      bool is_invalid = d_gpair[row_id].GetHess() - .0f == 0.f;
      return SamplePosition::Encode(nidx, !is_invalid);
    };  // NOLINT

    if (!p_fmat->SingleColBlock()) {
      CHECK_EQ(row_partitioner_->GetNumNodes(), p_tree->NumNodes());
      row_partitioner_->FinalisePosition(d_out_position, encode_op);
      dh::CopyTo(d_out_position, &positions_);
      return;
    }

    dh::caching_device_vector<uint32_t> categories;
    dh::CopyToD(p_tree->GetSplitCategories(), &categories);
    auto const& cat_segments = p_tree->GetSplitCategoriesPtr();
    auto d_categories = dh::ToSpan(categories);

    auto d_matrix = page->GetDeviceAccessor(ctx_->Device());

    std::vector<NodeSplitData> split_data(p_tree->NumNodes());
    auto const& tree = *p_tree;
    for (std::size_t i = 0, n = split_data.size(); i < n; ++i) {
      RegTree::Node split_node = tree[i];
      auto split_type = p_tree->NodeSplitType(i);
      auto node_cats = common::GetNodeCats(d_categories, cat_segments[i]);
      split_data[i] = NodeSplitData{std::move(split_node), split_type, node_cats};
    }

    auto go_left_op = GoLeftOp{d_matrix};
    dh::caching_device_vector<NodeSplitData> d_split_data;
    dh::CopyToD(split_data, &d_split_data);
    auto s_split_data = dh::ToSpan(d_split_data);

    row_partitioner_->FinalisePosition(d_out_position,
                                      [=] __device__(bst_idx_t row_id, bst_node_t nidx) {
                                        auto split_data = s_split_data[nidx];
                                        auto node = split_data.split_node;
                                        while (!node.IsLeaf()) {
                                          auto go_left = go_left_op(row_id, split_data);
                                          nidx = go_left ? node.LeftChild() : node.RightChild();
                                          node = s_split_data[nidx].split_node;
                                        }
                                        return encode_op(row_id, nidx);
                                      });
    dh::CopyTo(d_out_position, &positions_);
  }

  bool UpdatePredictionCache(linalg::MatrixView<float> out_preds_d, RegTree const* p_tree) {
    if (positions_.empty()) {
      return false;
    }

    CHECK(p_tree);
    CHECK(out_preds_d.Device().IsCUDA());
    CHECK_EQ(out_preds_d.Device().ordinal, ctx_->Ordinal());

    auto d_position = dh::ToSpan(positions_);
    CHECK_EQ(out_preds_d.Size(), d_position.size());

    // Use the nodes from tree, the leaf value might be changed by the objective since the
    // last update tree call.
    dh::caching_device_vector<RegTree::Node> nodes;
    dh::CopyTo(p_tree->GetNodes(), &nodes);
    common::Span<RegTree::Node> d_nodes = dh::ToSpan(nodes);
    CHECK_EQ(out_preds_d.Shape(1), 1);
    dh::LaunchN(d_position.size(), ctx_->CUDACtx()->Stream(),
                [=] XGBOOST_DEVICE(std::size_t idx) mutable {
                  bst_node_t nidx = d_position[idx];
                  nidx = SamplePosition::Decode(nidx);
                  auto weight = d_nodes[nidx].LeafValue();
                  out_preds_d(idx, 0) += weight;
                });
    return true;
  }

  // num histograms is the number of contiguous histograms in memory to reduce over
  void AllReduceHist(bst_node_t nidx, int num_histograms) {
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
    if (candidates.empty()) {
      return;
    }
    this->monitor.Start(__func__);
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
    hist.AllocateHistograms(ctx_, all_new);

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
    this->monitor.Stop(__func__);
  }

  void ApplySplit(const GPUExpandEntry& candidate, RegTree* p_tree) {
    RegTree& tree = *p_tree;

    // Sanity check - have we created a leaf with no training instances?
    if (!collective::IsDistributed() && row_partitioner_) {
      CHECK(row_partitioner_->GetRows(candidate.nid).size() > 0)
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

    hist.AllocateHistograms(ctx_, {kRootNIdx});
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
    bool const is_single_block = p_fmat->SingleColBlock();

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
          pinned.GetSpan<GPUExpandEntry>(filtered_expand_set.size() * 2, GPUExpandEntry{});
      // Update all the nodes if working with external memory, this saves us from working
      // with the finalize position call, which adds an additional iteration and requires
      // special handling for row index.
      this->UpdatePosition(is_single_block ? filtered_expand_set : expand_set, p_tree);

      this->BuildHistLeftRight(filtered_expand_set, tree);

      this->EvaluateSplits(filtered_expand_set, *p_tree, new_candidates);
      dh::DefaultStream().Sync();

      driver.Push(new_candidates.begin(), new_candidates.end());
      expand_set = driver.Pop();
    }
    // Row partitioner can have lesser nodes than the tree since we skip some leaf
    // nodes. These nodes are handled in the `FinalisePosition` call. However, a leaf can
    // be spliable before evaluation but invalid after evaluation as we have more
    // restrictions like min loss change after evalaution. Therefore, the check condition
    // is greater than or equal to.
    if (is_single_block) {
      CHECK_GE(p_tree->NumNodes(), this->row_partitioner_->GetNumNodes());
    }
    this->FinalisePosition(p_tree, p_fmat, *task, p_out_position);
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
    common::CheckComputeCapability();
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

  ~GPUHistMaker() override { dh::GlobalMemoryLogger().Log(); }

  void Update(TrainParam const* param, linalg::Matrix<GradientPair>* gpair, DMatrix* dmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override {
    monitor_.Start(__func__);

    CHECK_EQ(gpair->Shape(1), 1) << MTNotImplemented();
    auto gpair_hdv = gpair->Data();
    // build tree
    std::size_t t_idx{0};
    for (xgboost::RegTree* tree : trees) {
      this->UpdateTree(param, gpair_hdv, dmat, tree, &out_position[t_idx]);
      this->hist_maker_param_.CheckTreesSynchronized(ctx_, tree);
      ++t_idx;
    }
    dh::safe_cuda(cudaGetLastError());
    monitor_.Stop(__func__);
  }

  void InitDataOnce(TrainParam const* param, DMatrix* dmat) {
    CHECK_GE(ctx_->Ordinal(), 0) << "Must have at least one device";
    info_ = &dmat->Info();

    // Synchronise the column sampling seed
    uint32_t column_sampling_seed = common::GlobalRandom()();
    auto rc = collective::Broadcast(
        ctx_, linalg::MakeVec(&column_sampling_seed, sizeof(column_sampling_seed)), 0);
    SafeColl(rc);
    this->column_sampler_ = std::make_shared<common::ColumnSampler>(column_sampling_seed);

    dh::safe_cuda(cudaSetDevice(ctx_->Ordinal()));
    info_->feature_types.SetDevice(ctx_->Device());
    maker = std::make_unique<GPUHistMakerDevice>(
        ctx_, !dmat->SingleColBlock(), info_->feature_types.ConstDeviceSpan(), info_->num_row_,
        *param, column_sampler_, info_->num_col_, HistBatch(*param), dmat->Info());

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

  bool UpdatePredictionCache(const DMatrix* data, linalg::MatrixView<float> p_out_preds) override {
    if (maker == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.Start(__func__);
    bool result = maker->UpdatePredictionCache(p_out_preds, p_last_tree_);
    monitor_.Stop(__func__);
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

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([](Context const* ctx, ObjInfo const* task) {
      return new GPUHistMaker(ctx, task);
    });

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
    common::CheckComputeCapability();
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
    monitor_.Start(__func__);

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
    maker_ = std::make_unique<GPUHistMakerDevice>(
        ctx_, !p_fmat->SingleColBlock(), info.feature_types.ConstDeviceSpan(), info.num_row_,
        *param, column_sampler_, info.num_col_, ApproxBatch(*param, hess, *task_), p_fmat->Info());

    std::size_t t_idx{0};
    for (xgboost::RegTree* tree : trees) {
      this->UpdateTree(gpair->Data(), p_fmat, tree, &out_position[t_idx]);
      this->hist_maker_param_.CheckTreesSynchronized(ctx_, tree);
      ++t_idx;
    }

    monitor_.Stop(__func__);
  }

  void InitDataOnce(DMatrix* p_fmat) {
    if (this->initialised_) {
      return;
    }

    monitor_.Start(__func__);
    CHECK(ctx_->IsCUDA()) << error::InvalidCUDAOrdinal();
    uint32_t column_sampling_seed = common::GlobalRandom()();
    this->column_sampler_ = std::make_shared<common::ColumnSampler>(column_sampling_seed);

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

  bool UpdatePredictionCache(const DMatrix* data, linalg::MatrixView<float> p_out_preds) override {
    if (maker_ == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.Start(__func__);
    bool result = maker_->UpdatePredictionCache(p_out_preds, p_last_tree_);
    monitor_.Stop(__func__);
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

XGBOOST_REGISTER_TREE_UPDATER(GPUApproxMaker, "grow_gpu_approx")
    .describe("Grow tree with GPU.")
    .set_body([](Context const* ctx, ObjInfo const* task) {
      return new GPUGlobalApproxMaker(ctx, task);
    });
}  // namespace xgboost::tree
