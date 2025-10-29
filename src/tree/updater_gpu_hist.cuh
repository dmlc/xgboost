/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once
#include <thrust/reduce.h>  // for reduce_by_key

#include <memory>  // for unique_ptr
#include <vector>  // for vector

#include "../common/device_helpers.cuh"        // for MakeTransformIterator
#include "driver.h"                            // for Driver
#include "gpu_hist/feature_groups.cuh"         // for FeatureGroups
#include "gpu_hist/histogram.cuh"              // for DeviceHistogramBuilder
#include "gpu_hist/multi_evaluate_splits.cuh"  // for MultiHistEvaluator
#include "gpu_hist/row_partitioner.cuh"        // for RowPartitioner
#include "hist/hist_param.h"                   // for HistMakerTrainParam
#include "xgboost/base.h"                      // for bst_idx_t
#include "xgboost/context.h"                   // for Context
#include "xgboost/host_device_vector.h"        // for HostDeviceVector
#include "xgboost/tree_model.h"                // for RegTree

namespace xgboost::tree::cuda_impl {
// Use a large number to handle external memory with deep trees.
inline constexpr std::size_t kMaxNodeBatchSize = 1024;
using xgboost::cuda_impl::StaticBatch;

/**
 * @brief Implementation for vector leaf.
 */
class MultiTargetHistMaker {
 private:
  Context const* ctx_;

  TrainParam const param_;
  std::vector<std::unique_ptr<RowPartitioner>> partitioners_;

  HistMakerTrainParam const* hist_param_;
  std::shared_ptr<common::HistogramCuts const> const cuts_;
  std::unique_ptr<FeatureGroups> feature_groups_;
  DeviceHistogramBuilder histogram_;
  std::unique_ptr<MultiGradientQuantiser> quantiser_;

  MultiHistEvaluator evaluator_;

  linalg::Matrix<GradientPair> dh_gpair_;
  std::vector<bst_idx_t> const batch_ptr_;

  void BuildHist(EllpackPage const& page, bst_node_t nidx) {
    auto d_gpair = this->dh_gpair_.View(this->ctx_->Device());
    CHECK(!this->partitioners_.empty());
    auto d_ridx = this->partitioners_.front()->GetRows(nidx);
    auto hist = histogram_.GetNodeHistogram(nidx);
    auto roundings = this->quantiser_->Quantizers();
    auto acc = page.Impl()->GetDeviceEllpack(this->ctx_, {});
    histogram_.BuildHistogram(this->ctx_->CUDACtx(), acc,
                              this->feature_groups_->DeviceAccessor(this->ctx_->Device()), d_gpair,
                              d_ridx, hist, roundings);
  }

 public:
  void Reset(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat, RegTree* p_tree) {
    bst_idx_t n_targets = p_tree->NumTargets();
    auto in_gpair = linalg::MakeTensorView(ctx_, gpair_all, p_fmat->Info().num_row_, n_targets);

    /**
     * Initialize the partitioners
     */
    std::size_t n_batches = p_fmat->NumBatches();
    if (!partitioners_.empty()) {
      CHECK_EQ(partitioners_.size(), n_batches);
    }
    for (std::size_t k = 0; k < n_batches; ++k) {
      if (partitioners_.size() != n_batches) {
        // First run.
        partitioners_.emplace_back(std::make_unique<RowPartitioner>());
      }
      auto base_ridx = this->batch_ptr_[k];
      auto n_samples = this->batch_ptr_.at(k + 1) - base_ridx;
      partitioners_[k]->Reset(ctx_, n_samples, base_ridx);
    }
    this->partitioners_.resize(n_batches);

    /**
     * Initialize the histogram
     */
    std::size_t shape[2]{p_fmat->Info().num_row_, n_targets};
    dh_gpair_ = linalg::Matrix<GradientPair>{shape, ctx_->Device(), linalg::kF};
    TransposeGradient(this->ctx_, in_gpair, dh_gpair_.View(ctx_->Device()));

    this->quantiser_ = std::make_unique<MultiGradientQuantiser>(
        this->ctx_, dh_gpair_.View(ctx_->Device()), p_fmat->Info());

    bool force_global = true;
    histogram_.Reset(this->ctx_, this->hist_param_->MaxCachedHistNodes(ctx_->Device()),
                     feature_groups_->DeviceAccessor(ctx_->Device()),
                     cuts_->TotalBins() * n_targets, force_global);
  }

  [[nodiscard]] MultiExpandEntry InitRoot(DMatrix* p_fmat, RegTree* p_tree) {
    auto d_gpair = dh_gpair_.View(ctx_->Device());
    auto n_samples = d_gpair.Shape(0);
    auto n_targets = d_gpair.Shape(1);

    dh::device_vector<GradientPairInt64> root_sum(n_targets);

    auto key_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
      auto cidx = i / n_samples;
      return cidx;
    });
    auto d_roundings = quantiser_->Quantizers();
    auto val_it =
        dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) -> GradientPairInt64 {
          auto cidx = i / n_samples;
          auto ridx = i % n_samples;
          auto g = d_gpair(ridx, cidx);
          return d_roundings[cidx].ToFixedPoint(g);
        });
    thrust::reduce_by_key(ctx_->CUDACtx()->CTP(), key_it, key_it + d_gpair.Size(), val_it,
                          thrust::make_discard_iterator(), root_sum.begin());

    histogram_.AllocateHistograms(ctx_, {RegTree::kRoot});

    CHECK_EQ(p_fmat->NumBatches(), this->partitioners_.size());
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
      this->BuildHist(page, RegTree::kRoot);
    }

    auto node_hist = this->histogram_.GetNodeHistogram(RegTree::kRoot);
    MultiEvaluateSplitInputs input{RegTree::kRoot, p_tree->GetDepth(RegTree::kRoot),
                                   dh::ToSpan(root_sum), node_hist};
    GPUTrainingParam param{this->param_};
    MultiEvaluateSplitSharedInputs shared_inputs{d_roundings,
                                                 this->cuts_->cut_ptrs_.ConstDeviceSpan(),
                                                 this->cuts_->cut_values_.ConstDeviceSpan(),
                                                 this->cuts_->min_vals_.ConstDeviceSpan(),
                                                 this->param_.max_bin,
                                                 param};
    auto entry = this->evaluator_.EvaluateSingleSplit(ctx_, input, shared_inputs);

    // TODO(jiamingy): Support learning rate.
    std::vector<float> h_base_weight(entry.base_weight.size());
    dh::CopyDeviceSpanToVector(&h_base_weight, entry.base_weight);
    p_tree->SetLeaf(RegTree::kRoot, linalg::MakeVec(h_base_weight));
    return entry;
  }

  void ApplySplit(MultiExpandEntry const& candidate, RegTree* p_tree) {
    // TODO(jiamingy): Support learning rate.
    // TODO(jiamingy): Avoid device to host copies.
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

  void UpdateTree(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat, ObjInfo const*,
                  RegTree* p_tree, HostDeviceVector<bst_node_t>*) {
    bst_idx_t n_targets = p_tree->NumTargets();
    Driver<MultiExpandEntry> driver{param_, kMaxNodeBatchSize};

    this->Reset(gpair_all, p_fmat, p_tree);
    driver.Push({this->InitRoot(p_fmat, p_tree)});
    // The set of leaves that can be expanded asynchronously
    auto expand_set = driver.Pop();
    while (!expand_set.empty()) {
      for (auto& candidate : expand_set) {
        this->ApplySplit(candidate, p_tree);
      }
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
