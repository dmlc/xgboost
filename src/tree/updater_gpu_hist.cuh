/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once
#include <thrust/reduce.h>   // for reduce_by_key
#include <thrust/version.h>  // for THRUST_MAJOR_VERSION

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../common/device_helpers.cuh"        // for MakeTransformIterator
#include "driver.h"                            // for Driver
#include "gpu_hist/feature_groups.cuh"         // for FeatureGroups
#include "gpu_hist/histogram.cuh"              // for DeviceHistogramBuilder
#include "gpu_hist/leaf_sum.cuh"               // for LeafGradSum
#include "gpu_hist/multi_evaluate_splits.cuh"  // for MultiHistEvaluator
#include "gpu_hist/row_partitioner.cuh"        // for RowPartitioner
#include "hist/hist_param.h"                   // for HistMakerTrainParam
#include "tree_view.h"                         // for MultiTargetTreeView
#include "xgboost/base.h"                      // for bst_idx_t
#include "xgboost/context.h"                   // for Context
#include "xgboost/gradient.h"                  // for GradientContainer
#include "xgboost/host_device_vector.h"        // for HostDeviceVector
#include "xgboost/tree_model.h"                // for RegTree

namespace xgboost::tree::cuda_impl {
// Use a large number to handle external memory with deep trees.
inline constexpr std::size_t kMaxNodeBatchSize = 1024;
using xgboost::cuda_impl::StaticBatch;

template <typename GoLeftOp>
struct GoLeftWrapperOp {
  GoLeftOp go_left;
  template <typename NodeSplitData>
  __device__ bool operator()(cuda_impl::RowIndexT ridx, int /*nidx_in_batch*/,
                             const NodeSplitData& data) const {
    return go_left(ridx, data);
  }
};

// Some nodes we will manually compute histograms, others we will do by subtraction
template <typename TreeView, typename ExpandEntry, typename HessComp>
void AssignNodes(TreeView const& tree, std::vector<ExpandEntry> const& candidates,
                 common::Span<bst_node_t> nodes_to_build, common::Span<bst_node_t> nodes_to_sub,
                 HessComp&& compare_hess) {
  std::size_t nidx_in_set{0};
  auto p_build_nidx = nodes_to_build.data();
  auto p_sub_nidx = nodes_to_sub.data();
  for (auto& e : candidates) {
    // Decide whether to build the left histogram or right histogram Use sum of Hessian as
    // a heuristic to select node with fewest training instances This optimization is for
    // distributed training to avoid an allreduce call for synchronizing the number of
    // instances for each node.
    bool fewer_right = compare_hess(e);
    if (fewer_right) {
      p_build_nidx[nidx_in_set] = tree.RightChild(e.nidx);
      p_sub_nidx[nidx_in_set] = tree.LeftChild(e.nidx);
    } else {
      p_build_nidx[nidx_in_set] = tree.LeftChild(e.nidx);
      p_sub_nidx[nidx_in_set] = tree.RightChild(e.nidx);
    }
    ++nidx_in_set;
  }
}

void CalcRootSum(Context const* ctx, linalg::MatrixView<GradientPair> d_gpair,
                 common::Span<GradientQuantiser const> roundings,
                 common::Span<GradientPairInt64> root_sum) {
  auto n_samples = d_gpair.Shape(0);
  auto n_targets = d_gpair.Shape(1);
  // Calculate the root sum
  CHECK_EQ(n_targets, root_sum.size());

  auto key_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    auto cidx = i / n_samples;
    return cidx;
  });
  auto val_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) -> GradientPairInt64 {
    auto cidx = i / n_samples;
    auto ridx = i % n_samples;
    auto g = d_gpair(ridx, cidx);
    return roundings[cidx].ToFixedPoint(g);
  });
  thrust::reduce_by_key(ctx->CUDACtx()->CTP(), key_it, key_it + d_gpair.Size(), val_it,
                        thrust::make_discard_iterator(), dh::tbegin(root_sum));
}

/**
 * @brief Implementation for vector leaf.
 */
class MultiTargetHistMaker {
 private:
  Context const* ctx_;

  TrainParam const param_;
  RowPartitionerBatches partitioners_;

  HistMakerTrainParam const* hist_param_;
  std::shared_ptr<common::HistogramCuts const> const cuts_;
  std::unique_ptr<FeatureGroups> feature_groups_;
  DeviceHistogramBuilder histogram_;
  std::unique_ptr<MultiGradientQuantiser> split_quantizer_;
  std::unique_ptr<MultiGradientQuantiser> value_quantizer_;

  MultiHistEvaluator evaluator_;

  // Gradient used for building the tree structure
  linalg::Matrix<GradientPair> split_gpair_;
  // Gradient used for calculating the leaf values
  linalg::Matrix<GradientPair> value_gpair_;
  std::vector<bst_idx_t> const batch_ptr_;

  dh::PinnedMemory pinned_;

  void BuildHist(EllpackPage const& page, std::int32_t k, bst_node_t nidx) {
    auto d_gpair = this->split_gpair_.View(this->ctx_->Device());
    CHECK(!this->partitioners_.Empty());
    auto d_ridx = this->partitioners_.At(k)->GetRows(nidx);
    auto hist = histogram_.GetNodeHistogram(nidx);
    auto roundings = this->split_quantizer_->Quantizers();
    auto acc = page.Impl()->GetDeviceEllpack(this->ctx_, {});
    histogram_.BuildHistogram(this->ctx_->CUDACtx(), acc,
                              this->feature_groups_->DeviceAccessor(this->ctx_->Device()), d_gpair,
                              d_ridx, hist, roundings);
  }

 public:
  void Reset(linalg::Matrix<GradientPair>* gpair_all, DMatrix* p_fmat) {
    bst_idx_t n_targets = gpair_all->Shape(1);
    auto in_gpair = gpair_all->View(ctx_->Device());

    /**
     * Initialize the partitioners
     */
    partitioners_.Reset(this->ctx_, batch_ptr_);

    /**
     * Initialize the histogram
     */
    std::size_t shape[2]{p_fmat->Info().num_row_, n_targets};
    split_gpair_ = linalg::Matrix<GradientPair>{shape, ctx_->Device(), linalg::kF};
    TransposeGradient(this->ctx_, in_gpair, split_gpair_.View(ctx_->Device()));

    this->split_quantizer_ = std::make_unique<MultiGradientQuantiser>(
        this->ctx_, split_gpair_.View(ctx_->Device()), p_fmat->Info());

    if (!this->value_gpair_.Empty()) {
      this->value_quantizer_ = std::make_unique<MultiGradientQuantiser>(
          this->ctx_, value_gpair_.View(ctx_->Device()), p_fmat->Info());
    }

    bool force_global = true;
    histogram_.Reset(this->ctx_, this->hist_param_->MaxCachedHistNodes(ctx_->Device()),
                     feature_groups_->DeviceAccessor(ctx_->Device()),
                     cuts_->TotalBins() * n_targets, force_global);
  }

  [[nodiscard]] MultiExpandEntry InitRoot(DMatrix* p_fmat, RegTree* p_tree) {
    auto d_gpair = split_gpair_.View(ctx_->Device());
    auto n_targets = d_gpair.Shape(1);

    // Calculate the root sum
    this->evaluator_.AllocNodeSum(RegTree::kRoot, n_targets);
    auto d_root_sum = this->evaluator_.GetNodeSum(RegTree::kRoot, n_targets);
    CalcRootSum(this->ctx_, d_gpair, this->split_quantizer_->Quantizers(), d_root_sum);

    // Build the root histogram.
    histogram_.AllocateHistograms(ctx_, {RegTree::kRoot});

    CHECK_EQ(p_fmat->NumBatches(), this->partitioners_.Size());
    std::int32_t k = 0;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
      this->BuildHist(page, k, RegTree::kRoot);
      ++k;
    }

    // Evaluate root split
    auto node_hist = this->histogram_.GetNodeHistogram(RegTree::kRoot);
    MultiEvaluateSplitInputs input{RegTree::kRoot, p_tree->GetDepth(RegTree::kRoot), d_root_sum,
                                   node_hist};
    auto d_roundings = split_quantizer_->Quantizers();
    GPUTrainingParam param{this->param_};
    MultiEvaluateSplitSharedInputs shared_inputs{d_roundings,
                                                 this->cuts_->cut_ptrs_.ConstDeviceSpan(),
                                                 this->cuts_->cut_values_.ConstDeviceSpan(),
                                                 this->cuts_->min_vals_.ConstDeviceSpan(),
                                                 this->param_.max_bin,
                                                 param};
    auto entry = this->evaluator_.EvaluateSingleSplit(ctx_, input, shared_inputs);
    p_tree->SetRoot(linalg::MakeVec(this->ctx_->Device(), entry.base_weight));

    return entry;
  }

  void ApplySplit(std::vector<MultiExpandEntry> const& h_candidates, RegTree* p_tree) {
    CHECK(!h_candidates.empty());
    auto n_targets = h_candidates.front().base_weight.size();

    // TODO(jiamingy): Avoid device to host copies.
    for (auto const& candidate : h_candidates) {
      std::vector<float> h_base_weight(candidate.base_weight.size());
      std::vector<float> h_left_weight(candidate.left_weight.size());
      std::vector<float> h_right_weight(candidate.right_weight.size());
      dh::CopyDeviceSpanToVector(&h_base_weight, candidate.base_weight);
      dh::CopyDeviceSpanToVector(&h_left_weight, candidate.left_weight);
      dh::CopyDeviceSpanToVector(&h_right_weight, candidate.right_weight);

      p_tree->ExpandNode(candidate.nidx, candidate.split.findex, candidate.split.fvalue,
                         candidate.split.dir == kLeftDir, linalg::MakeVec(h_base_weight),
                         linalg::MakeVec(h_left_weight), linalg::MakeVec(h_right_weight));
    }

    dh::device_vector<MultiExpandEntry> candidates{h_candidates};
    this->evaluator_.ApplyTreeSplit(this->ctx_, p_tree, dh::ToSpan(candidates), n_targets);
  }
  /**
   * @brief Calculate the leaf weight based on the node sum for each leaf.
   *
   * This method helps support reduced gradient. Weights in p_tree are calculated using
   * split gradient. This function replaces those weights with new weights calculated from
   * value gradient.
   */
  void ExpandTreeLeaf(linalg::Matrix<GradientPair> const& full_grad, RegTree* p_tree) const {
    auto n_leaves = static_cast<bst_target_t>(p_tree->GetNumLeaves());
    auto out_sum = linalg::Constant(ctx_, GradientPairInt64{}, n_leaves, p_tree->NumTargets());
    auto d_out_sum = out_sum.View(this->ctx_->Device());

    auto d_full_grad = full_grad.View(this->ctx_->Device());
    auto d_roundings = this->value_quantizer_->Quantizers();
    // Node indices for all leaves
    std::vector<bst_node_t> leaves_idx(n_leaves);

#if THRUST_MAJOR_VERSION >= 3
    // do nothing
#else
    CHECK_EQ(this->partitioners_.Size(), 1)
        << "External memory not implemented for old CCCL versions. (thrust < 3.0)";
#endif
    std::int32_t batch_idx = 0;
    for (auto const& p_part : this->partitioners_) {
      auto leaves = p_part->GetLeaves();
      CHECK_EQ(leaves.size(), n_leaves);
      LeafGradSum(this->ctx_, leaves, d_roundings, p_part->GetRows(), d_full_grad, d_out_sum);
      if (batch_idx == 0) {
        // Populate the node indices
        std::transform(leaves.begin(), leaves.end(), leaves_idx.begin(),
                       [](LeafInfo const& leaf) { return leaf.nidx; });
      }
      // sanity check
      if (this->hist_param_->debug_synchronize) {
        auto it = common::MakeIndexTransformIter([&](std::size_t i) { return leaves.at(i).nidx; });
        CHECK(std::equal(it, it + n_leaves, leaves_idx.cbegin()));
      }
      ++batch_idx;
    }

    auto param = GPUTrainingParam{this->param_};
    auto out_weight = linalg::Empty<float>(this->ctx_, n_leaves, p_tree->NumTargets());
    // Use full value gradient for leaf values.
    LeafWeight(this->ctx_, param, this->value_quantizer_->Quantizers(),
               out_sum.View(this->ctx_->Device()), out_weight.View(this->ctx_->Device()));

    p_tree->SetLeaves(leaves_idx, out_weight.Data()->ConstHostSpan());
  }

  struct NodeSplitData {
    bst_node_t nidx;
  };

  struct PartitionNodes {
    std::vector<bst_node_t> nidx;
    std::vector<bst_node_t> left_nidx;
    std::vector<bst_node_t> right_nidx;
    std::vector<NodeSplitData> split_data;

    explicit PartitionNodes(std::size_t n_candidates)
        : nidx(n_candidates),
          left_nidx(n_candidates),
          right_nidx(n_candidates),
          split_data(n_candidates) {}
  };

  PartitionNodes CreatePartitionNodes(RegTree const* p_tree,
                                      std::vector<MultiExpandEntry> const& candidates) {
    PartitionNodes nodes(candidates.size());
    auto tree = p_tree->HostMtView();
    for (std::size_t i = 0, n = candidates.size(); i < n; i++) {
      auto const& e = candidates[i];
      auto split_type = tree.SplitType(e.nidx);
      nodes.nidx.at(i) = e.nidx;
      nodes.left_nidx[i] = tree.LeftChild(e.nidx);
      nodes.right_nidx[i] = tree.RightChild(e.nidx);
      nodes.split_data[i] = NodeSplitData{e.nidx};

      CHECK_EQ(split_type == FeatureType::kCategorical, e.split.is_cat);
    }
    return nodes;
  }

  // TODO(jiamingy): Merge this with the single target version. Make sure copying tree
  // data doesn't block external memory execution.
  //
  // Pulling in the device view has negative performance impact as we need to resize the
  // tree internal buffers repeatedly, which invokes many small data copies and memory
  // allocations.
  template <typename Accessor>
  struct GoLeftOp {
    Accessor d_matrix;
    MultiTargetTreeView tree;
    __device__ bool operator()(cuda_impl::RowIndexT ridx, NodeSplitData const& data) const {
      // given a row index, returns the node id it belongs to
      float cut_value = d_matrix.GetFvalue(ridx, tree.SplitIndex(data.nidx));
      // Missing value
      bool go_left = true;
      if (isnan(cut_value)) {
        go_left = tree.DefaultLeft(data.nidx);
      } else {
        if (tree.SplitType(data.nidx) == FeatureType::kCategorical) {
          go_left = common::Decision(tree.NodeCats(data.nidx), cut_value);
        } else {
          go_left = cut_value <= tree.SplitCond(data.nidx);
        }
      }
      return go_left;
    }
  };

  void ReduceHist(DMatrix* p_fmat, std::vector<MultiExpandEntry> const& candidates,
                  std::vector<bst_node_t> const& build_nidx,
                  std::vector<bst_node_t> const& subtraction_nidx) {
    if (candidates.empty()) {
      return;
    }

    // Perform subtraction for sibling nodes
    auto need_build = this->histogram_.SubtractHist(ctx_, candidates, build_nidx, subtraction_nidx);
    if (need_build.empty()) {
      return;
    }

    // Build the nodes that can not obtain the histogram using subtraction. This is the slow path.
    std::int32_t k = 0;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
      for (auto nidx : need_build) {
        this->BuildHist(page, k, nidx);
      }
      ++k;
    }
  }

  void PartitionAndBuildHist(DMatrix* p_fmat, std::vector<MultiExpandEntry> const& expand_set,
                             std::vector<MultiExpandEntry> const& candidates,
                             RegTree const* p_tree) {
    if (expand_set.empty()) {
      return;
    }
    CHECK_LE(candidates.size(), expand_set.size());
    // TODO(jiamingy): Implement finalize partition. Avoid using the expand_set.

    // Prepare for update partition
    auto nodes = this->CreatePartitionNodes(p_tree, expand_set);

    std::vector<bst_node_t> build_nidx(expand_set.size());
    std::vector<bst_node_t> subtraction_nidx(expand_set.size());
    auto mt_tree = p_tree->HostMtView();
    AssignNodes(mt_tree, expand_set, build_nidx, subtraction_nidx, [](MultiExpandEntry const& e) {
      bool fewer_right = e.right_fst_hess < e.left_fst_hess;
      return fewer_right;
    });

    histogram_.AllocateHistograms(this->ctx_, build_nidx, subtraction_nidx);

    // Pull to device
    mt_tree = MultiTargetTreeView{this->ctx_->Device(), p_tree};
    std::int32_t k{0};
    // TODO(jiamingy): Support external memory.
    bool prefetch_copy = true;
    for (auto const& page :
         p_fmat->GetBatches<EllpackPage>(this->ctx_, StaticBatch(prefetch_copy))) {
      page.Impl()->Visit(this->ctx_, {}, [&](auto&& d_acc) {
        using Acc = std::remove_reference_t<decltype(d_acc)>;
        using GoLeft = GoLeftOp<Acc>;
        auto go_left = GoLeft{d_acc, mt_tree};
        partitioners_.UpdatePositionBatch(this->ctx_, k, nodes.nidx, nodes.left_nidx,
                                          nodes.right_nidx, nodes.split_data,
                                          GoLeftWrapperOp<GoLeft>{go_left});

        for (auto nidx : build_nidx) {
          this->BuildHist(page, k, nidx);
        }
      });
      ++k;
    }

    this->ReduceHist(p_fmat, expand_set, build_nidx, subtraction_nidx);
  }

  void EvaluateSplits(std::vector<MultiExpandEntry> const& candidates, RegTree const& tree,
                      common::Span<MultiExpandEntry> pinned_candidates_out) {
    if (candidates.empty()) {
      return;
    }
    GPUTrainingParam param{this->param_};
    MultiEvaluateSplitSharedInputs shared_inputs{
        this->split_quantizer_->Quantizers(),
        this->cuts_->cut_ptrs_.ConstDeviceSpan(),
        this->cuts_->cut_values_.ConstDeviceSpan(),
        this->cuts_->min_vals_.ConstDeviceSpan(),
        this->param_.max_bin,
        param,
    };

    dh::device_vector<MultiEvaluateSplitInputs> inputs(2 * candidates.size());
    dh::device_vector<MultiExpandEntry> outputs(2 * candidates.size());

    auto mt_tree = tree.HostMtView();
    std::vector<MultiEvaluateSplitInputs> h_node_inputs(candidates.size() * 2);
    bst_node_t max_nidx = 0;
    for (auto const& candidate : candidates) {
      bst_node_t left_nidx = mt_tree.LeftChild(candidate.nidx);
      bst_node_t right_nidx = mt_tree.RightChild(candidate.nidx);
      max_nidx = std::max({max_nidx, left_nidx, right_nidx});
    }
    auto n_targets = this->split_gpair_.Shape(1);
    for (std::size_t i = 0; i < candidates.size(); i++) {
      auto candidate = candidates.at(i);
      bst_node_t left_nidx = mt_tree.LeftChild(candidate.nidx);
      bst_node_t right_nidx = mt_tree.RightChild(candidate.nidx);
      // Make sure no allocation is happening.
      // The parent sum is calculated in the last apply tree split.
      auto left = MultiEvaluateSplitInputs{left_nidx, candidate.depth + 1,
                                           this->evaluator_.GetNodeSum(left_nidx, n_targets),
                                           histogram_.GetNodeHistogram(left_nidx)};
      auto right = MultiEvaluateSplitInputs{right_nidx, candidate.depth + 1,
                                            this->evaluator_.GetNodeSum(right_nidx, n_targets),
                                            histogram_.GetNodeHistogram(right_nidx)};
      h_node_inputs[i * 2] = left;
      h_node_inputs[i * 2 + 1] = right;
    }
    dh::safe_cuda(cudaMemcpyAsync(inputs.data().get(), h_node_inputs.data(),
                                  common::SizeBytes<MultiEvaluateSplitInputs>(h_node_inputs.size()),
                                  cudaMemcpyDefault, ctx_->CUDACtx()->Stream()));
    this->evaluator_.EvaluateSplits(this->ctx_, dh::ToSpan(inputs), shared_inputs,
                                    dh::ToSpan(outputs));
    dh::safe_cuda(cudaMemcpyAsync(pinned_candidates_out.data(), outputs.data().get(),
                                  pinned_candidates_out.size_bytes(), cudaMemcpyDefault,
                                  ctx_->CUDACtx()->Stream()));
  }

  void UpdateTree(GradientContainer* gpair, DMatrix* p_fmat, ObjInfo const* task, RegTree* p_tree) {
    auto* split_grad = gpair->Grad();
    if (gpair->HasValueGrad()) {
      this->value_gpair_ =
          linalg::Matrix<GradientPair>{gpair->value_gpair.Shape(), ctx_->Device(), linalg::kF};
      TransposeGradient(this->ctx_, gpair->value_gpair.View(this->ctx_->Device()),
                        value_gpair_.View(this->ctx_->Device()));
    }

    this->GrowTree(split_grad, p_fmat, task, p_tree);

    if (gpair->HasValueGrad()) {
      this->ExpandTreeLeaf(gpair->value_gpair, p_tree);
    } else {
      p_tree->GetMultiTargetTree()->SetLeaves();
    }
  }

  void GrowTree(linalg::Matrix<GradientPair>* split_gpair, DMatrix* p_fmat, ObjInfo const*,
                RegTree* p_tree) {
    if (this->param_.learning_rate - 1.0 != 0.0) {
      LOG(FATAL) << "GPU" << MTNotImplemented();
    }
    Driver<MultiExpandEntry> driver{param_, kMaxNodeBatchSize};

    this->Reset(split_gpair, p_fmat);
    driver.Push({this->InitRoot(p_fmat, p_tree)});

    // The set of leaves that can be expanded asynchronously
    auto expand_set = driver.Pop();
    while (!expand_set.empty()) {
      this->ApplySplit(expand_set, p_tree);
      // Get the candidates we are allowed to expand further
      // e.g. We do not bother further processing nodes whose children are beyond max depth
      std::vector<MultiExpandEntry> valid_candidates;
      std::copy_if(expand_set.begin(), expand_set.end(), std::back_inserter(valid_candidates),
                   [&](auto const& e) { return driver.IsChildValid(e); });

      // Allocate children nodes.
      auto new_candidates = pinned_.GetSpan(valid_candidates.size() * 2, MultiExpandEntry{});

      this->PartitionAndBuildHist(p_fmat, expand_set, valid_candidates, p_tree);

      this->EvaluateSplits(valid_candidates, *p_tree, new_candidates);
      this->ctx_->CUDACtx()->Stream().Sync();

      driver.Push(new_candidates.begin(), new_candidates.end());

      expand_set = driver.Pop();
    }
  }

  explicit MultiTargetHistMaker(Context const* ctx, TrainParam param,
                                HistMakerTrainParam const* hist_param,
                                std::vector<bst_idx_t> batch_ptr,
                                std::shared_ptr<common::HistogramCuts const> cuts,
                                bool dense_compressed)
      : ctx_{ctx},
        param_{std::move(param)},
        hist_param_{hist_param},
        cuts_{std::move(cuts)},
        feature_groups_{std::make_unique<FeatureGroups>(*cuts_, dense_compressed,
                                                        dh::MaxSharedMemoryOptin(ctx_->Ordinal()))},
        batch_ptr_{std::move(batch_ptr)} {}
};
}  // namespace xgboost::tree::cuda_impl
