/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <algorithm>                 // for max, min
#include <cstdint>                   // for int32_t
#include <cub/util_type.cuh>         // for KeyValuePair, Uninitialized
#include <cub/warp/warp_reduce.cuh>  // for WarpReduce
#include <cub/warp/warp_scan.cuh>    // for WarpScan
#include <cuda/ptx>                  // for get_sreg_laneid
#include <cuda/std/functional>       // for plus
#include <limits>                    // for numeric_limits
#include <type_traits>               // for integral_constant

#include "../../common/categorical.h"     // for CatBitField, UseOneHot
#include "../../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/nvtx_utils.h"      // for xgboost_NVTX_FN_RANGE
#include "evaluate_splits.cuh"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
namespace {
using ScanT = cub::WarpScan<GradientPairInt64>;
using ArgMaxT = cub::KeyValuePair<std::int32_t, float>;
using MaxT = cub::WarpReduce<ArgMaxT>;
using SumT = cub::WarpReduce<GradientPairInt64>;

enum class SplitKind { kNumerical, kOneHot, kPartition };

// Reversed bin index for the backward pass. All inputs are global across features.
XGBOOST_DEV_INLINE bst_bin_t RevBinIdx(bst_bin_t begin, bst_bin_t end, bst_bin_t bin) {
  return begin + (end - bin - 1);
}

// Load using efficient 128 vector load instruction
XGBOOST_DEV_INLINE GradientPairInt64 LoadGpair(GradientPairInt64 const *ptr) {
  static_assert(sizeof(GradientPairInt64) == sizeof(float4));
  float4 tmp = *reinterpret_cast<float4 const *>(ptr);
  return *reinterpret_cast<GradientPairInt64 const *>(&tmp);
}

/**
 * @brief Find the best split of one feature for one node, using a warp.
 *
 * A pass scans the histogram in one direction and scores every split point with the gain
 * shared with the CPU implementation. Only the running prefix of each target is retained
 * during the scan. With multiple targets, the child sums of the best split are restored
 * after all passes.
 *
 * @tparam kTargets Number of targets if known at compile time.
 */
template <std::size_t kTargets>
struct FeatureSplitEnumerator {
  struct TempStorage {
    ScanT::TempStorage scan;
    MaxT::TempStorage max;
    SumT::TempStorage sum;
  };

  EvaluateSplitInputs const &node;
  EvaluateSplitSharedInputs const &shared;
  TreeEvaluator::SplitEvaluator<EvalParam> const &evaluator;
  bst_feature_t fidx;
  bst_bin_t begin;
  bst_bin_t end;
  bst_bin_t const *sorted_idx;
  // Running prefix sum for each target.
  common::Span<GradientPairInt64> carry;
  // Child sum of the best split for each target.
  common::Span<GradientPairInt64> best_sum;
  TempStorage *storage;
  DeviceSplitCandidate *best;
  bst_bin_t *best_bin;
  double parent_gain{0};

  [[nodiscard]] __device__ bst_target_t Targets() const {
    if constexpr (kTargets == common::dynamic_extent) {
      return shared.Targets();
    } else {
      return kTargets;
    }
  }

  __device__ FeatureSplitEnumerator(EvaluateSplitInputs const &node,
                                    EvaluateSplitSharedInputs const &shared,
                                    TreeEvaluator::SplitEvaluator<EvalParam> const &evaluator,
                                    bst_feature_t fidx, bst_bin_t const *sorted_idx,
                                    common::Span<GradientPairInt64> carry,
                                    common::Span<GradientPairInt64> best_sum,
                                    TempStorage *storage, DeviceSplitCandidate *best,
                                    bst_bin_t *best_bin)
      : node{node},
        shared{shared},
        evaluator{evaluator},
        fidx{fidx},
        begin{static_cast<bst_bin_t>(shared.feature_segments[fidx])},
        end{static_cast<bst_bin_t>(shared.feature_segments[fidx + 1])},
        sorted_idx{sorted_idx},
        carry{carry},
        best_sum{best_sum},
        storage{storage},
        best{best},
        best_bin{best_bin} {
    common::Span<GradientQuantiser const, kTargets> roundings{shared.roundings.data(),
                                                              this->Targets()};
    QuantizedGradientSum<kTargets> parent{node.parent_sum.data(), roundings};
    parent_gain = evaluator.CalcGain(node.nidx, shared.param, parent);
  }

  template <SplitKind kKind, DefaultDirection kDir>
  [[nodiscard]] __device__ bst_bin_t HistogramIndex(bst_bin_t bin) const {
    if constexpr (kKind == SplitKind::kPartition) {
      return sorted_idx[kDir == kLeftDir ? bin : RevBinIdx(begin, end, bin)];
    } else if constexpr (kKind == SplitKind::kNumerical) {
      return kDir == kRightDir ? bin : RevBinIdx(begin, end, bin);
    } else {
      return bin;
    }
  }
  [[nodiscard]] __device__ GradientPairInt64 const *Histogram(bst_target_t t) const {
    return node.histogram.data() + static_cast<std::size_t>(t) * shared.n_total_bins_per_tar;
  }

  // Whether the sum of the observed bins differs from the node sum. Valid after the
  // forward pass.
  [[nodiscard]] __device__ bool HasMissing() const {
    for (bst_target_t t = 0; t < this->Targets(); ++t) {
      if (!(carry[t] == node.parent_sum[t])) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Enumerate split points in one direction.
   *
   * @tparam kDir The default direction for missing values. The scanned child is the left
   *              child for kRightDir and the right child for kLeftDir.
   */
  template <SplitKind kKind, DefaultDirection kDir>
  __device__ void Enumerate() {
    auto lane = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
    auto nt = this->Targets();
    auto limit = end;
    if constexpr (kKind == SplitKind::kPartition) {
      // A partition must leave at least one category on each side.
      limit = begin + std::min(shared.param.max_cat_threshold, end - begin) - 1;
    }
    for (bst_target_t t = lane; t < nt; t += dh::WarpThreads()) {
      carry[t] = GradientPairInt64{};
    }
    __syncwarp();
    if constexpr (kKind == SplitKind::kOneHot && kDir == kRightDir) {
      // Missing values follow the matching category to the right. The left child is the
      // feature sum excluding the category.
      for (bst_target_t t = 0; t < nt; ++t) {
        GradientPairInt64 local;
        for (auto i = begin + lane; i < end; i += dh::WarpThreads()) {
          local += LoadGpair(this->Histogram(t) + i);
        }
        auto total = SumT(storage->sum).Sum(local);
        if (lane == 0) {
          carry[t] = total;
        }
        __syncwarp();
      }
    }

    for (bst_bin_t tile = begin; tile < limit; tile += dh::WarpThreads()) {
      auto bin_idx = tile + lane;
      auto hidx = this->HistogramIndex<kKind, kDir>(std::min(bin_idx, end - 1));
      double gain = 0, left_hess = 0, right_hess = 0;
      GradientPairInt64 bin;
      for (bst_target_t t = 0; t < nt; ++t) {
        bin = bin_idx < limit ? LoadGpair(this->Histogram(t) + hidx) : GradientPairInt64{};
        if constexpr (kKind == SplitKind::kOneHot) {
          if constexpr (kDir == kRightDir) {
            bin = carry[t] - bin;
          }
        } else {
          if (lane == 0) {
            bin += carry[t];
          }
          GradientPairInt64 aggregate;
          ScanT(storage->scan).InclusiveScan(bin, bin, cuda::std::plus{}, aggregate);
          __syncwarp();
          if (lane == 0) {
            carry[t] = aggregate;
          }
        }
        auto rounding = shared.roundings[t];
        auto child = rounding.ToFloatingPoint(bin);
        auto sibling = rounding.ToFloatingPoint(node.parent_sum[t] - bin);
        auto const &left = kDir == kRightDir ? child : sibling;
        auto const &right = kDir == kRightDir ? sibling : child;
        left_hess += left.GetHess();
        right_hess += right.GetHess();
        gain += evaluator.CalcTargetSplitGain(shared.param, node.nidx, fidx, t, nt == 1, left,
                                              right);
      }
      auto k = static_cast<double>(nt);
      float loss_chg = -std::numeric_limits<float>::infinity();
      if (bin_idx < limit && IsValidSplit(shared.param, left_hess / k, right_hess / k)) {
        // Subtract the parent gain in double before narrowing, as the CPU implementation does.
        loss_chg = static_cast<float>(gain - parent_gain);
        loss_chg = isnan(loss_chg) ? -std::numeric_limits<float>::infinity() : loss_chg;
      }

      // Like the CPU scan, retain the first split on equal gain.
      auto max = MaxT(storage->max).Reduce({lane, loss_chg}, cub::ArgMax{});
      int updated_lane = -1;
      // Lane 0 owns the candidate throughout both passes.
      if (lane == 0 && max.value > best->loss_chg) {
        auto idx = tile + max.key;
        *best_bin = idx;
        updated_lane = max.key;
        best->loss_chg = max.value;
        best->dir = kDir;
        best->findex = fidx;
        best->child_sum = best_sum;
        best->is_cat = kKind != SplitKind::kNumerical;
        if constexpr (kKind == SplitKind::kNumerical) {
          if constexpr (kDir == kLeftDir) {
            auto split_bin = RevBinIdx(begin, end, idx);
            best->fvalue = split_bin == begin ? -std::numeric_limits<float>::infinity()
                                              : shared.feature_values[split_bin - 1];
          } else {
            best->fvalue = shared.feature_values[idx];
          }
        } else {
          best->fvalue = std::numeric_limits<float>::quiet_NaN();
          if constexpr (kKind == SplitKind::kPartition) {
            auto offset = idx - begin;
            best->thresh = kDir == kLeftDir ? offset : end - begin - offset - 1;
          } else {
            best->thresh = common::AsCat(shared.feature_values[idx]);
          }
        }
      }
      if constexpr (kTargets == 1) {
        // The winning lane already holds the complete child sum.
        if (lane == __shfl_sync(dh::WarpFullMask(), updated_lane, 0)) {
          best_sum[0] = bin;
        }
      }
      __syncwarp();
    }
  }

  // Restore the child sums of the best split for multiple targets.
  template <SplitKind kKind>
  __device__ void RestoreBest() {
    if constexpr (kTargets != 1) {
      auto best_idx = __shfl_sync(dh::WarpFullMask(), *best_bin, 0);
      if (best_idx < 0) {
        return;
      }
      auto lane = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
      auto dir = __shfl_sync(dh::WarpFullMask(), static_cast<int>(best->dir), 0);
      for (bst_target_t t = 0; t < this->Targets(); ++t) {
        auto hist = this->Histogram(t);
        GradientPairInt64 local;
        if constexpr (kKind == SplitKind::kOneHot) {
          for (auto i = begin + lane; i < end; i += dh::WarpThreads()) {
            // Only the category for kLeftDir, all other categories for kRightDir.
            if ((i == best_idx) == (dir == kLeftDir)) {
              local += hist[i];
            }
          }
        } else {
          for (auto i = begin + lane; i <= best_idx; i += dh::WarpThreads()) {
            local += hist[dir == kRightDir ? this->HistogramIndex<kKind, kRightDir>(i)
                                           : this->HistogramIndex<kKind, kLeftDir>(i)];
          }
        }
        auto sum = SumT(storage->sum).Sum(local);
        if (lane == 0) {
          best_sum[t] = sum;
        }
        __syncwarp();
      }
    }
  }

  template <SplitKind kKind>
  __device__ void Run() {
    using Kind = std::integral_constant<SplitKind, kKind>;
    if constexpr (kKind == SplitKind::kPartition) {
      this->Enumerate<kKind, kLeftDir>();
      this->Enumerate<kKind, kRightDir>();
    } else {
      this->Enumerate<kKind, kRightDir>();
      // Without missing values, the backward pass enumerates the same splits.
      auto has_missing = this->HasMissing();
      // Finish reading the forward sums before the backward pass clears carry.
      __syncwarp();
      if (has_missing) {
        this->Enumerate<kKind, kLeftDir>();
      }
    }
    this->RestoreBest<Kind::value>();
  }
};

// This kernel uses block_size == warp_size. Each thread has a large amount of work, and
// with a single warp, synchronisation is performed with warp intrinsics.
template <std::size_t kTargets, bool kHasCategorical>
__global__ __launch_bounds__(dh::WarpThreads()) void EvaluateSplitsKernel(
    common::Span<EvaluateSplitInputs const> nodes, EvaluateSplitSharedInputs shared,
    common::Span<bst_bin_t const> sorted_idx, TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
    common::Span<GradientPairInt64> scratch, common::Span<DeviceSplitCandidate> out) {
  using Enumerator = FeatureSplitEnumerator<kTargets>;
  __shared__ typename Enumerator::TempStorage storage;
  __shared__ cub::Uninitialized<GradientPairInt64> carry_storage;
  DeviceSplitCandidate best;
  bst_bin_t best_bin = -1;

  auto lane = cuda::ptx::get_sreg_laneid();
  auto n_features = shared.max_active_feature;
  auto nidx_in_set = blockIdx.x / n_features;
  auto const &node = nodes[nidx_in_set];
  // Features are sampled, some blocks may not have any feature to work on.
  auto fi = blockIdx.x % n_features;
  if (fi < node.feature_set.size()) {
    auto fidx = node.feature_set[fi];
    auto nt = kTargets == common::dynamic_extent ? shared.Targets() : kTargets;
    auto best_sum = scratch.subspan(static_cast<std::size_t>(blockIdx.x) * nt, nt);
    // The best sum is restored after the scan with multiple targets, share the storage.
    auto carry = kTargets == 1 ? common::Span{&carry_storage.Alias(), 1} : best_sum;
    auto node_sorted_idx = sorted_idx.empty() ? nullptr
                                              : sorted_idx.data() +
                                                    nidx_in_set * shared.n_total_bins_per_tar;
    Enumerator enumerator{node,  shared,   evaluator, fidx,  node_sorted_idx,
                          carry, best_sum, &storage,  &best, &best_bin};

    if (kHasCategorical && shared.IsCategorical(fidx)) {
      if (shared.IsPartition(fidx)) {
        enumerator.template Run<SplitKind::kPartition>();
      } else {
        enumerator.template Run<SplitKind::kOneHot>();
      }
    } else {
      enumerator.template Run<SplitKind::kNumerical>();
    }
  }
  if (lane == 0) {
    out[blockIdx.x] = best;
  }
}

__global__ __launch_bounds__(dh::WarpThreads()) void FinalizeSplitsKernel(
    common::Span<EvaluateSplitInputs const> d_inputs, EvaluateSplitSharedInputs shared,
    common::Span<bst_bin_t const> d_sorted_idx, TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
    common::Span<DeviceSplitCandidate const> d_splits, GPUHistEvaluator::NodeWeightBuffer d_weights,
    common::Span<GradientPairInt64> d_split_sums, common::Span<CatWordT> d_split_cats,
    std::size_t cat_storage_size, common::Span<GPUExpandEntry> out_splits) {
  using SplitMaxT = cub::WarpReduce<DeviceSplitCandidate>;
  __shared__ SplitMaxT::TempStorage storage;

  auto nidx_in_set = blockIdx.x;
  auto n_features = shared.max_active_feature;
  auto lane = cuda::ptx::get_sreg_laneid();
  // Higher gain wins, ties are broken by the lower feature index.
  auto op = [](DeviceSplitCandidate const &lhs, DeviceSplitCandidate const &rhs) {
    if (lhs.loss_chg != rhs.loss_chg) {
      return lhs.loss_chg > rhs.loss_chg ? lhs : rhs;
    }
    return lhs.findex <= rhs.findex ? lhs : rhs;
  };
  DeviceSplitCandidate split;
  for (auto fi = lane; fi < n_features; fi += dh::WarpThreads()) {
    split = op(split, d_splits[nidx_in_set * n_features + fi]);
  }
  split = SplitMaxT{storage}.Reduce(split, op);
  if (lane != 0) {
    return;
  }

  auto n_targets = shared.Targets();
  auto const &input = d_inputs[nidx_in_set];
  auto nidx = input.nidx;
  auto const &param = shared.param;
  auto roundings = shared.roundings;

  // The base weight is required even when no valid split is found.
  auto base_weight = d_weights.Base(nidx);
  double parent_hess = 0;
  for (bst_target_t t = 0; t < n_targets; ++t) {
    auto g = roundings[t].ToFloatingPoint(input.parent_sum[t]);
    base_weight[t] = evaluator.CalcWeight(nidx, t, param, g);
    parent_hess += g.GetHess();
  }
  auto &out = out_splits[nidx_in_set];
  out = GPUExpandEntry{nidx, input.depth, split, parent_hess, 0.0};

  auto node_cats = d_split_cats.subspan(nidx * cat_storage_size, cat_storage_size);
  for (auto &word : node_cats) {
    word = 0;
  }
  if (split.child_sum.empty()) {
    return;
  }
  if (split.is_cat) {
    common::CatBitField cats{node_cats};
    auto fidx = split.findex;
    if (!shared.IsPartition(fidx)) {
      cats.Set(split.thresh);
    } else {
      auto f_begin = shared.feature_segments[fidx];
      auto f_sorted_idx = d_sorted_idx.subspan(nidx_in_set * shared.n_total_bins_per_tar + f_begin,
                                               shared.FeatureBins(fidx));
      // Categories at the beginning of the sorted order go to the right.
      auto partition = split.dir == kLeftDir ? split.thresh + 1 : split.thresh;
      for (bst_bin_t i = 0; i < partition; ++i) {
        cats.Set(common::AsCat(shared.feature_values[f_sorted_idx[i]]));
      }
    }
  }

  auto left_weight = d_weights.Left(nidx);
  auto right_weight = d_weights.Right(nidx);
  // The child sum points to the scratch buffer, persist it for the loss-guide policy.
  auto split_sum = d_split_sums.subspan(nidx * n_targets, n_targets);
  double left_hess = 0, right_hess = 0;
  for (bst_target_t t = 0; t < n_targets; ++t) {
    split_sum[t] = split.child_sum[t];
    auto child = roundings[t].ToFloatingPoint(split.child_sum[t]);
    auto sibling = roundings[t].ToFloatingPoint(input.parent_sum[t] - split.child_sum[t]);
    auto const &left = split.dir == kRightDir ? child : sibling;
    auto const &right = split.dir == kRightDir ? sibling : child;
    auto [lw, rw] = evaluator.CalcSplitWeights(param, nidx, split.findex, t, left, right);
    left_weight[t] = lw;
    right_weight[t] = rw;
    left_hess += left.GetHess();
    right_hess += right.GetHess();
  }
  out.left_hess = left_hess;
  out.right_hess = right_hess;
}
}  // anonymous namespace

GPUExpandEntry GPUHistEvaluator::EvaluateSingleSplit(
    Context const *ctx, EvaluateSplitInputs const &input,
    EvaluateSplitSharedInputs const &shared_inputs) {
  dh::CachingDeviceUVector<EvaluateSplitInputs> inputs(1);
  dh::safe_cuda(cudaMemcpyAsync(inputs.data(), &input, sizeof(input), cudaMemcpyDefault,
                                ctx->CUDACtx()->Stream()));
  dh::CachingDeviceUVector<GPUExpandEntry> outputs(1);
  this->EvaluateSplits(ctx, dh::ToSpan(inputs), shared_inputs, input.nidx, dh::ToSpan(outputs));
  GPUExpandEntry entry;
  dh::safe_cuda(cudaMemcpyAsync(&entry, outputs.data(), sizeof(entry), cudaMemcpyDefault,
                                ctx->CUDACtx()->Stream()));
  ctx->CUDACtx()->Stream().Sync();
  return entry;
}

void GPUHistEvaluator::EvaluateSplits(Context const *ctx,
                                      common::Span<EvaluateSplitInputs const> d_inputs,
                                      EvaluateSplitSharedInputs const &shared_inputs,
                                      bst_node_t max_nidx,
                                      common::Span<GPUExpandEntry> out_splits) {
  xgboost_NVTX_FN_RANGE();
  auto n_targets = shared_inputs.Targets();
  auto evaluator = this->GetEvaluator();
  CHECK_EQ(n_targets, this->n_targets_);
  CHECK_EQ(n_targets, evaluator.n_targets);
  CHECK_GE(shared_inputs.n_total_bins_per_tar, 1);
  auto n_features = shared_inputs.max_active_feature;
  CHECK_LT(n_features, shared_inputs.feature_segments.size());

  std::uint32_t n_nodes = d_inputs.size();
  CHECK_EQ(n_nodes, out_splits.size());
  if (n_nodes == 0) {
    return;
  }

  // Allocate weight and split sum storage on demand for the maximum node ID being evaluated.
  this->node_cat_storage_size_ = shared_inputs.cat_storage_size;
  Grow(&this->node_weights_, max_nidx, n_targets * NodeWeightBuffer::kWeightsPerNode);
  Grow(&this->split_sums_, max_nidx, n_targets);
  Grow(&this->split_cats_, max_nidx, this->node_cat_storage_size_);

  common::Span<bst_bin_t const> d_sorted_idx;
  if (this->need_sort_histogram_) {
    d_sorted_idx = this->SortHistogram(ctx, d_inputs, shared_inputs);
  }

  // Find the best split for each feature. Empty workers in distributed training have no
  // feature and produce no valid split.
  std::size_t n_blocks = static_cast<std::size_t>(n_nodes) * n_features;
  dh::CachingDeviceUVector<DeviceSplitCandidate> d_splits(n_blocks);
  if (n_blocks != 0) {
    this->scratch_.resize(n_blocks * n_targets);
    auto launch = [&](auto targets, auto has_cat) {
      auto kernel = EvaluateSplitsKernel<decltype(targets)::value, decltype(has_cat)::value>;
      dh::LaunchKernel{static_cast<std::uint32_t>(n_blocks), dh::WarpThreads(), 0,
                       ctx->CUDACtx()->Stream()}(kernel, d_inputs, shared_inputs, d_sorted_idx,
                                                 evaluator, dh::ToSpan(this->scratch_),
                                                 dh::ToSpan(d_splits));
    };
    auto dispatch = [&](auto targets) {
      if (shared_inputs.cat_storage_size > 0) {
        launch(targets, std::true_type{});
      } else {
        launch(targets, std::false_type{});
      }
    };
    if (n_targets == 1) {
      dispatch(std::integral_constant<std::size_t, 1>{});
    } else {
      dispatch(std::integral_constant<std::size_t, common::dynamic_extent>{});
    }
  }

  // Find the best split for each node, then calculate and persist weights and categories.
  dh::LaunchKernel{n_nodes, dh::WarpThreads(), 0, ctx->CUDACtx()->Stream()}(
      FinalizeSplitsKernel, d_inputs, shared_inputs, d_sorted_idx, evaluator,
      dh::ToSpan(d_splits), this->GetNodeWeights(), dh::ToSpan(this->split_sums_),
      dh::ToSpan(this->split_cats_), this->node_cat_storage_size_, out_splits);
}

void GPUHistEvaluator::CopyToHost(Context const *ctx, bst_node_t first_nidx,
                                  bst_node_t last_nidx) {
  CHECK_LE(first_nidx, last_nidx);
  auto copy = [&](auto const &d_vec, auto *h_vec, std::size_t n_per_node) {
    if (n_per_node == 0) {
      return;
    }
    auto required = static_cast<std::size_t>(last_nidx + 1) * n_per_node;
    CHECK_GE(d_vec.size(), required);
    if (h_vec->size() < required) {
      h_vec->resize(required);
    }
    auto offset = static_cast<std::size_t>(first_nidx) * n_per_node;
    auto n_bytes = (required - offset) * sizeof(*h_vec->data());
    dh::safe_cuda(cudaMemcpyAsync(h_vec->data() + offset, d_vec.data() + offset, n_bytes,
                                  cudaMemcpyDeviceToHost, ctx->CUDACtx()->Stream()));
  };
  copy(this->node_weights_, &this->h_node_weights_,
       this->n_targets_ * NodeWeightBuffer::kWeightsPerNode);
  copy(this->split_cats_, &this->h_split_cats_, this->node_cat_storage_size_);
}

void GPUHistEvaluator::ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                                      common::Span<GPUExpandEntry const> h_candidates) {
  xgboost_NVTX_FN_RANGE();
  CHECK(!h_candidates.empty());
  auto h_splits = this->h_splits_.GetSpan<NodeSplit>(h_candidates.size());
  bst_node_t max_nidx{0};
  for (std::size_t i = 0; i < h_candidates.size(); ++i) {
    auto const &c = h_candidates[i];
    h_splits[i] = {c.nidx, p_tree->LeftChild(c.nidx), p_tree->RightChild(c.nidx),
                   static_cast<bst_feature_t>(c.split.findex), c.split.dir};
    max_nidx = std::max({max_nidx, h_splits[i].left_nidx, h_splits[i].right_nidx});
  }
  this->d_splits_.resize(h_splits.size());
  dh::safe_cuda(cudaMemcpyAsync(this->d_splits_.data(), h_splits.data(), h_splits.size_bytes(),
                                cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
  auto d_splits = dh::ToSpan(this->d_splits_);

  auto weights = this->GetNodeWeights();
  this->tree_evaluator_->AddSplits(
      d_splits.size(), max_nidx, [=] XGBOOST_DEVICE(std::size_t i, bst_target_t t) {
        auto const &s = d_splits[i];
        return TreeEvaluator::SplitBounds{s.nidx,
                                          s.left_nidx,
                                          s.right_nidx,
                                          s.fidx,
                                          weights.Left(s.nidx)[t],
                                          weights.Right(s.nidx)[t]};
      });

  // Assign the node sums for the next evaluation. Use the persistent split sums instead of
  // the candidate's child sum, which can be stale in the loss-guide policy.
  this->AllocNodeSum(max_nidx);
  auto node_sums = dh::ToSpan(this->node_sums_);
  auto split_sums = dh::ToSpan(this->split_sums_);
  auto n_targets = this->n_targets_;
  dh::LaunchN(n_targets * d_splits.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) {
                auto const &s = d_splits[i / n_targets];
                auto t = i % n_targets;
                auto parent = node_sums[s.nidx * n_targets + t];
                auto child = split_sums[s.nidx * n_targets + t];
                auto non_missing = s.dir == kRightDir ? s.left_nidx : s.right_nidx;
                auto missing = s.dir == kRightDir ? s.right_nidx : s.left_nidx;
                node_sums[non_missing * n_targets + t] = child;
                node_sums[missing * n_targets + t] = parent - child;
              });
}
}  // namespace xgboost::tree
