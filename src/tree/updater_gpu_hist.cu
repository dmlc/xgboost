/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <xgboost/tree_updater.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "xgboost/host_device_vector.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"
#include "xgboost/json.h"

#include "../common/io.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/timer.h"
#include "../data/ellpack_page.cuh"

#include "param.h"
#include "updater_gpu_common.cuh"
#include "constraints.cuh"
#include "gpu_hist/gradient_based_sampler.cuh"
#include "gpu_hist/row_partitioner.cuh"
#include "gpu_hist/histogram.cuh"
#include "gpu_hist/evaluate_splits.cuh"
#include "gpu_hist/driver.cuh"

namespace xgboost {
namespace tree {
#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);
#endif  // !defined(GTEST_TEST)

// training parameters specific to this algorithm
struct GPUHistMakerTrainParam
    : public XGBoostParameter<GPUHistMakerTrainParam> {
  bool single_precision_histogram;
  bool deterministic_histogram;
  bool debug_synchronize;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPUHistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
    DMLC_DECLARE_FIELD(deterministic_histogram).set_default(true).describe(
        "Pre-round the gradient for obtaining deterministic gradient histogram.");
    DMLC_DECLARE_FIELD(debug_synchronize).set_default(false).describe(
        "Check if all distributed tree are identical after tree construction.");
  }
};
#if !defined(GTEST_TEST)
DMLC_REGISTER_PARAMETER(GPUHistMakerTrainParam);
#endif  // !defined(GTEST_TEST)

/**
 * \struct  DeviceHistogram
 *
 * \summary Data storage for node histograms on device. Automatically expands.
 *
 * \tparam GradientSumT      histogram entry type.
 * \tparam kStopGrowingSize  Do not grow beyond this size
 *
 * \author  Rory
 * \date    28/07/2018
 */
template <typename GradientSumT, size_t kStopGrowingSize = 1 << 26>
class DeviceHistogram {
 private:
  /*! \brief Map nidx to starting index of its histogram. */
  std::map<int, size_t> nidx_map_;
  dh::device_vector<typename GradientSumT::ValueT> data_;
  int n_bins_;
  int device_id_;
  static constexpr size_t kNumItemsInGradientSum =
      sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT);
  static_assert(kNumItemsInGradientSum == 2,
                "Number of items in gradient type should be 2.");

 public:
  void Init(int device_id, int n_bins) {
    this->n_bins_ = n_bins;
    this->device_id_ = device_id;
  }

  void Reset() {
    auto d_data = data_.data().get();
      dh::LaunchN(device_id_, data_.size(),
                  [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    nidx_map_.clear();
  }
  bool HistogramExists(int nidx) const {
    return nidx_map_.find(nidx) != nidx_map_.cend();
  }
  int Bins() const {
    return n_bins_;
  }
  size_t HistogramSize() const {
    return n_bins_ * kNumItemsInGradientSum;
  }

  dh::device_vector<typename GradientSumT::ValueT>& Data() {
    return data_;
  }

  void AllocateHistogram(int nidx) {
    if (HistogramExists(nidx)) return;
    // Number of items currently used in data
    const size_t used_size = nidx_map_.size() * HistogramSize();
    const size_t new_used_size = used_size + HistogramSize();
    if (data_.size() >= kStopGrowingSize) {
      // Recycle histogram memory
      if (new_used_size <= data_.size()) {
        // no need to remove old node, just insert the new one.
        nidx_map_[nidx] = used_size;
        // memset histogram size in bytes
      } else {
        std::pair<int, size_t> old_entry = *nidx_map_.begin();
        nidx_map_.erase(old_entry.first);
        nidx_map_[nidx] = old_entry.second;
      }
      // Zero recycled memory
      auto d_data = data_.data().get() + nidx_map_[nidx];
      dh::LaunchN(device_id_, n_bins_ * 2,
                  [=] __device__(size_t idx) { d_data[idx] = 0.0f; });
    } else {
      // Append new node histogram
      nidx_map_[nidx] = used_size;
      // Check there is enough memory for another histogram node
      if (data_.size() < new_used_size + HistogramSize()) {
        size_t new_required_memory =
            std::max(data_.size() * 2, HistogramSize());
        data_.resize(new_required_memory);
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
    auto ptr = data_.data().get() + nidx_map_[nidx];
    return common::Span<GradientSumT>(
        reinterpret_cast<GradientSumT*>(ptr), n_bins_);
  }
};

struct CalcWeightTrainParam {
  float min_child_weight;
  float reg_alpha;
  float reg_lambda;
  float max_delta_step;
  float learning_rate;
  XGBOOST_DEVICE explicit CalcWeightTrainParam(const TrainParam& p)
      : min_child_weight(p.min_child_weight),
        reg_alpha(p.reg_alpha),
        reg_lambda(p.reg_lambda),
        max_delta_step(p.max_delta_step),
        learning_rate(p.learning_rate) {}
};

// Manage memory for a single GPU
template <typename GradientSumT>
struct GPUHistMakerDevice {
  int device_id;
  EllpackPageImpl* page;
  BatchParam batch_param;

  std::unique_ptr<RowPartitioner> row_partitioner;
  DeviceHistogram<GradientSumT> hist{};

  common::Span<GradientPair> gpair;

  dh::caching_device_vector<int> monotone_constraints;
  dh::caching_device_vector<bst_float> prediction_cache;

  /*! \brief Sum gradient for each node. */
  std::vector<GradientPair> node_sum_gradients;

  TrainParam param;
  bool deterministic_histogram;

  GradientSumT histogram_rounding;

  dh::PinnedMemory pinned;

  std::vector<cudaStream_t> streams{};

  common::Monitor monitor;
  std::vector<ValueConstraint> node_value_constraints;
  common::ColumnSampler column_sampler;
  FeatureInteractionConstraintDevice interaction_constraints;

  std::unique_ptr<GradientBasedSampler> sampler;

  GPUHistMakerDevice(int _device_id,
                     EllpackPageImpl* _page,
                     bst_uint _n_rows,
                     TrainParam _param,
                     uint32_t column_sampler_seed,
                     uint32_t n_features,
                     bool deterministic_histogram,
                     BatchParam _batch_param)
      : device_id(_device_id),
        page(_page),
        param(std::move(_param)),
        column_sampler(column_sampler_seed),
        interaction_constraints(param, n_features),
        deterministic_histogram{deterministic_histogram},
        batch_param(_batch_param) {
    sampler.reset(new GradientBasedSampler(
        page, _n_rows, batch_param, param.subsample, param.sampling_method));
    if (!param.monotone_constraints.empty()) {
      // Copy assigning an empty vector causes an exception in MSVC debug builds
      monotone_constraints = param.monotone_constraints;
    }
    node_sum_gradients.resize(param.MaxNodes());

    // Init histogram
    hist.Init(device_id, page->Cuts().TotalBins());
    monitor.Init(std::string("GPUHistMakerDevice") + std::to_string(device_id));
  }

  ~GPUHistMakerDevice() {  // NOLINT
    dh::safe_cuda(cudaSetDevice(device_id));
    for (auto& stream : streams) {
      dh::safe_cuda(cudaStreamDestroy(stream));
    }
  }

  // Get vector of at least n initialised streams
  std::vector<cudaStream_t>& GetStreams(int n) {
    if (n > streams.size()) {
      for (auto& stream : streams) {
        dh::safe_cuda(cudaStreamDestroy(stream));
      }

      streams.clear();
      streams.resize(n);

      for (auto& stream : streams) {
        dh::safe_cuda(cudaStreamCreate(&stream));
      }
    }

    return streams;
  }

  // Reset values for each update iteration
  // Note that the column sampler must be passed by value because it is not
  // thread safe
  void Reset(HostDeviceVector<GradientPair>* dh_gpair, DMatrix* dmat, int64_t num_columns) {
    this->column_sampler.Init(num_columns, param.colsample_bynode,
      param.colsample_bylevel, param.colsample_bytree);
    dh::safe_cuda(cudaSetDevice(device_id));
    this->interaction_constraints.Reset();
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(),
              GradientPair());

    auto sample = sampler->Sample(dh_gpair->DeviceSpan(), dmat);
    page = sample.page;
    gpair = sample.gpair;

    if (deterministic_histogram) {
      histogram_rounding = CreateRoundingFactor<GradientSumT>(this->gpair);
    } else {
      histogram_rounding = GradientSumT{0.0, 0.0};
    }

    row_partitioner.reset();  // Release the device memory first before reallocating
    row_partitioner.reset(new RowPartitioner(device_id,  sample.sample_rows));
    hist.Reset();
  }


  DeviceSplitCandidate EvaluateRootSplit(GradientPair root_sum) {
    int nidx = 0;
    dh::TemporaryArray<DeviceSplitCandidate> splits_out(1);
    GPUTrainingParam gpu_param(param);
    auto sampled_features = column_sampler.GetFeatureSet(0);
    sampled_features->SetDevice(device_id);
    common::Span<bst_feature_t> feature_set =
        interaction_constraints.Query(sampled_features->DeviceSpan(), nidx);
    auto matrix = page->GetDeviceAccessor(device_id);
    EvaluateSplitInputs<GradientSumT> inputs{
        nidx,
        {root_sum.GetGrad(), root_sum.GetHess()},
        gpu_param,
        feature_set,
        matrix.feature_segments,
        matrix.gidx_fvalue_map,
        matrix.min_fvalue,
        hist.GetNodeHistogram(nidx),
        node_value_constraints[nidx],
        dh::ToSpan(monotone_constraints)};
    EvaluateSingleSplit(dh::ToSpan(splits_out), inputs);
    std::vector<DeviceSplitCandidate> result(1);
    dh::safe_cuda(cudaMemcpy(result.data(), splits_out.data().get(),
                             sizeof(DeviceSplitCandidate) * splits_out.size(),
                             cudaMemcpyDeviceToHost));
    return result.front();
  }

  void EvaluateLeftRightSplits(
      ExpandEntry candidate, int left_nidx, int right_nidx, const RegTree& tree,
      common::Span<ExpandEntry> pinned_candidates_out) {
    dh::TemporaryArray<DeviceSplitCandidate> splits_out(2);
    GPUTrainingParam gpu_param(param);
    auto left_sampled_features =
        column_sampler.GetFeatureSet(tree.GetDepth(left_nidx));
    left_sampled_features->SetDevice(device_id);
    common::Span<bst_feature_t> left_feature_set =
        interaction_constraints.Query(left_sampled_features->DeviceSpan(),
                                      left_nidx);
    auto right_sampled_features =
        column_sampler.GetFeatureSet(tree.GetDepth(right_nidx));
    right_sampled_features->SetDevice(device_id);
    common::Span<bst_feature_t> right_feature_set =
        interaction_constraints.Query(right_sampled_features->DeviceSpan(),
                                      left_nidx);
    auto matrix = page->GetDeviceAccessor(device_id);

    EvaluateSplitInputs<GradientSumT> left{left_nidx,
                                           {candidate.split.left_sum.GetGrad(),
                                            candidate.split.left_sum.GetHess()},
                                           gpu_param,
                                           left_feature_set,
                                           matrix.feature_segments,
                                           matrix.gidx_fvalue_map,
                                           matrix.min_fvalue,
                                           hist.GetNodeHistogram(left_nidx),
                                           node_value_constraints[left_nidx],
                                           dh::ToSpan(monotone_constraints)};
    EvaluateSplitInputs<GradientSumT> right{
        right_nidx,
        {candidate.split.right_sum.GetGrad(),
         candidate.split.right_sum.GetHess()},
        gpu_param,
        right_feature_set,
        matrix.feature_segments,
        matrix.gidx_fvalue_map,
        matrix.min_fvalue,
        hist.GetNodeHistogram(right_nidx),
        node_value_constraints[right_nidx],
        dh::ToSpan(monotone_constraints)};
    auto d_splits_out = dh::ToSpan(splits_out);
    EvaluateSplits(d_splits_out, left, right);
    dh::TemporaryArray<ExpandEntry> entries(2);
    auto d_entries = entries.data().get();
    dh::LaunchN(device_id, 1, [=] __device__(size_t idx) {
      d_entries[0] =
          ExpandEntry(left_nidx, candidate.depth + 1, d_splits_out[0]);
      d_entries[1] =
          ExpandEntry(right_nidx, candidate.depth + 1, d_splits_out[1]);
    });
    dh::safe_cuda(cudaMemcpyAsync(
        pinned_candidates_out.data(), entries.data().get(),
        sizeof(ExpandEntry) * entries.size(), cudaMemcpyDeviceToHost));
  }

  void BuildHist(int nidx) {
    hist.AllocateHistogram(nidx);
    auto d_node_hist = hist.GetNodeHistogram(nidx);
    auto d_ridx = row_partitioner->GetRows(nidx);
    BuildGradientHistogram(page->GetDeviceAccessor(device_id), gpair, d_ridx, d_node_hist,
                           histogram_rounding);
  }

  void SubtractionTrick(int nidx_parent, int nidx_histogram,
                        int nidx_subtraction) {
    auto d_node_hist_parent = hist.GetNodeHistogram(nidx_parent);
    auto d_node_hist_histogram = hist.GetNodeHistogram(nidx_histogram);
    auto d_node_hist_subtraction = hist.GetNodeHistogram(nidx_subtraction);

    dh::LaunchN(device_id, page->Cuts().TotalBins(), [=] __device__(size_t idx) {
      d_node_hist_subtraction[idx] =
          d_node_hist_parent[idx] - d_node_hist_histogram[idx];
    });
  }

  bool CanDoSubtractionTrick(int nidx_parent, int nidx_histogram,
                             int nidx_subtraction) {
    // Make sure histograms are already allocated
    hist.AllocateHistogram(nidx_subtraction);
    return hist.HistogramExists(nidx_histogram) &&
           hist.HistogramExists(nidx_parent);
  }

  void UpdatePosition(int nidx, RegTree::Node split_node) {
    auto d_matrix = page->GetDeviceAccessor(device_id);

    row_partitioner->UpdatePosition(
        nidx, split_node.LeftChild(), split_node.RightChild(),
        [=] __device__(bst_uint ridx) {
          // given a row index, returns the node id it belongs to
          bst_float cut_value =
              d_matrix.GetFvalue(ridx, split_node.SplitIndex());
          // Missing value
          int new_position = 0;
          if (isnan(cut_value)) {
            new_position = split_node.DefaultChild();
          } else {
            if (cut_value <= split_node.SplitCond()) {
              new_position = split_node.LeftChild();
            } else {
              new_position = split_node.RightChild();
            }
          }
          return new_position;
        });
  }

  // After tree update is finished, update the position of all training
  // instances to their final leaf. This information is used later to update the
  // prediction cache
  void FinalisePosition(RegTree const* p_tree, DMatrix* p_fmat) {
    dh::TemporaryArray<RegTree::Node> d_nodes(p_tree->GetNodes().size());
    dh::safe_cuda(cudaMemcpy(d_nodes.data().get(), p_tree->GetNodes().data(),
                             d_nodes.size() * sizeof(RegTree::Node),
                             cudaMemcpyHostToDevice));

    if (row_partitioner->GetRows().size() != p_fmat->Info().num_row_) {
      row_partitioner.reset();  // Release the device memory first before reallocating
      row_partitioner.reset(new RowPartitioner(device_id, p_fmat->Info().num_row_));
    }
    if (page->n_rows == p_fmat->Info().num_row_) {
      FinalisePositionInPage(page, dh::ToSpan(d_nodes));
    } else {
      for (auto& batch : p_fmat->GetBatches<EllpackPage>(batch_param)) {
        FinalisePositionInPage(batch.Impl(), dh::ToSpan(d_nodes));
      }
    }
  }

  void FinalisePositionInPage(EllpackPageImpl* page, const common::Span<RegTree::Node> d_nodes) {
    auto d_matrix = page->GetDeviceAccessor(device_id);
    row_partitioner->FinalisePosition(
        [=] __device__(size_t row_id, int position) {
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
          if (element <= node.SplitCond()) {
            position = node.LeftChild();
          } else {
            position = node.RightChild();
          }
        }
        node = d_nodes[position];
      }
      return position;
    });
  }

  void UpdatePredictionCache(bst_float* out_preds_d) {
    dh::safe_cuda(cudaSetDevice(device_id));
    auto d_ridx = row_partitioner->GetRows();
    if (prediction_cache.size() != d_ridx.size()) {
      prediction_cache.resize(d_ridx.size());
      dh::safe_cuda(cudaMemcpyAsync(prediction_cache.data().get(), out_preds_d,
                                    prediction_cache.size() * sizeof(bst_float),
                                    cudaMemcpyDefault));
    }

    CalcWeightTrainParam param_d(param);
    dh::TemporaryArray<GradientPair> device_node_sum_gradients(node_sum_gradients.size());

    dh::safe_cuda(
        cudaMemcpyAsync(device_node_sum_gradients.data().get(), node_sum_gradients.data(),
                        sizeof(GradientPair) * node_sum_gradients.size(),
                        cudaMemcpyHostToDevice));
    auto d_position = row_partitioner->GetPosition();
    auto d_node_sum_gradients = device_node_sum_gradients.data().get();
    auto d_prediction_cache = prediction_cache.data().get();

    dh::LaunchN(
        device_id, prediction_cache.size(), [=] __device__(int local_idx) {
          int pos = d_position[local_idx];
          bst_float weight = CalcWeight(param_d, d_node_sum_gradients[pos]);
          d_prediction_cache[d_ridx[local_idx]] +=
              weight * param_d.learning_rate;
        });

    dh::safe_cuda(cudaMemcpy(
        out_preds_d, prediction_cache.data().get(),
        prediction_cache.size() * sizeof(bst_float), cudaMemcpyDefault));
    row_partitioner.reset();
  }

  void AllReduceHist(int nidx, dh::AllReducer* reducer) {
    monitor.Start("AllReduce");
    auto d_node_hist = hist.GetNodeHistogram(nidx).data();
    reducer->AllReduceSum(
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        page->Cuts().TotalBins() * (sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT)));

    monitor.Stop("AllReduce");
  }

  /**
   * \brief Build GPU local histograms for the left and right child of some parent node
   */
  void BuildHistLeftRight(const ExpandEntry &candidate, int nidx_left,
        int nidx_right, dh::AllReducer* reducer) {
    auto build_hist_nidx = nidx_left;
    auto subtraction_trick_nidx = nidx_right;

    // Decide whether to build the left histogram or right histogram
    // Use sum of Hessian as a heuristic to select node with fewest training instances
    bool fewer_right = candidate.split.right_sum.GetHess() < candidate.split.left_sum.GetHess();
    if (fewer_right) {
      std::swap(build_hist_nidx, subtraction_trick_nidx);
    }

    this->BuildHist(build_hist_nidx);
    this->AllReduceHist(build_hist_nidx, reducer);

    // Check whether we can use the subtraction trick to calculate the other
    bool do_subtraction_trick = this->CanDoSubtractionTrick(
        candidate.nid, build_hist_nidx, subtraction_trick_nidx);

    if (do_subtraction_trick) {
      // Calculate other histogram using subtraction trick
      this->SubtractionTrick(candidate.nid, build_hist_nidx,
                             subtraction_trick_nidx);
    } else {
      // Calculate other histogram manually
      this->BuildHist(subtraction_trick_nidx);
      this->AllReduceHist(subtraction_trick_nidx, reducer);
    }
  }

  void ApplySplit(const ExpandEntry& candidate, RegTree* p_tree) {
    RegTree& tree = *p_tree;

    node_value_constraints.resize(tree.GetNodes().size());
    auto parent_sum = candidate.split.left_sum + candidate.split.right_sum;
    auto base_weight = node_value_constraints[candidate.nid].CalcWeight(
        param, parent_sum);
    auto left_weight = node_value_constraints[candidate.nid].CalcWeight(
                           param, candidate.split.left_sum) *
                       param.learning_rate;
    auto right_weight = node_value_constraints[candidate.nid].CalcWeight(
                            param, candidate.split.right_sum) *
                        param.learning_rate;
    tree.ExpandNode(candidate.nid, candidate.split.findex,
                    candidate.split.fvalue, candidate.split.dir == kLeftDir,
                    base_weight, left_weight, right_weight,
                    candidate.split.loss_chg, parent_sum.GetHess(),
                     candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    // Set up child constraints
    node_value_constraints.resize(tree.GetNodes().size());
    node_value_constraints[candidate.nid].SetChild(
        param, tree[candidate.nid].SplitIndex(), candidate.split.left_sum,
        candidate.split.right_sum,
        &node_value_constraints[tree[candidate.nid].LeftChild()],
        &node_value_constraints[tree[candidate.nid].RightChild()]);
    node_sum_gradients[tree[candidate.nid].LeftChild()] =
        candidate.split.left_sum;
    node_sum_gradients[tree[candidate.nid].RightChild()] =
        candidate.split.right_sum;

    interaction_constraints.Split(
        candidate.nid, tree[candidate.nid].SplitIndex(),
        tree[candidate.nid].LeftChild(),
                                  tree[candidate.nid].RightChild());
  }

  ExpandEntry InitRoot(RegTree* p_tree, dh::AllReducer* reducer) {
    constexpr bst_node_t kRootNIdx = 0;
    dh::XGBCachingDeviceAllocator<char> alloc;
    GradientPair root_sum = thrust::reduce(
        thrust::cuda::par(alloc),
        thrust::device_ptr<GradientPair const>(gpair.data()),
        thrust::device_ptr<GradientPair const>(gpair.data() + gpair.size()));
    rabit::Allreduce<rabit::op::Sum, float>(reinterpret_cast<float*>(&root_sum),
                                            2);

    this->BuildHist(kRootNIdx);
    this->AllReduceHist(kRootNIdx, reducer);

    // Remember root stats
    node_sum_gradients[kRootNIdx] = root_sum;
    p_tree->Stat(kRootNIdx).sum_hess = root_sum.GetHess();
    auto weight = CalcWeight(param, root_sum);
    p_tree->Stat(kRootNIdx).base_weight = weight;
    (*p_tree)[kRootNIdx].SetLeaf(param.learning_rate * weight);

    // Initialise root constraint
    node_value_constraints.resize(p_tree->GetNodes().size());

    // Generate first split
    auto split = this->EvaluateRootSplit(root_sum);
    return ExpandEntry(kRootNIdx, p_tree->GetDepth(kRootNIdx), split);
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat,
                  RegTree* p_tree, dh::AllReducer* reducer) {
    auto& tree = *p_tree;
    Driver driver(static_cast<TrainParam::TreeGrowPolicy>(param.grow_policy));

    monitor.Start("Reset");
    this->Reset(gpair_all, p_fmat, p_fmat->Info().num_col_);
    monitor.Stop("Reset");

    monitor.Start("InitRoot");
    driver.Push({ this->InitRoot(p_tree, reducer) });
    monitor.Stop("InitRoot");

    auto num_leaves = 1;

    // The set of leaves that can be expanded asynchronously
    auto expand_set = driver.Pop();
    while (!expand_set.empty()) {
      auto new_candidates =
          pinned.GetSpan<ExpandEntry>(expand_set.size() * 2, ExpandEntry());

      for (auto i = 0ull; i < expand_set.size(); i++) {
        auto candidate = expand_set.at(i);
        if (!candidate.IsValid(param, num_leaves)) {
          continue;
        }
        this->ApplySplit(candidate, p_tree);

        num_leaves++;

        int left_child_nidx = tree[candidate.nid].LeftChild();
        int right_child_nidx = tree[candidate.nid].RightChild();
        // Only create child entries if needed
        if (ExpandEntry::ChildIsValid(param, tree.GetDepth(left_child_nidx),
          num_leaves)) {
          monitor.Start("UpdatePosition");
          this->UpdatePosition(candidate.nid, (*p_tree)[candidate.nid]);
          monitor.Stop("UpdatePosition");

          monitor.Start("BuildHist");
          this->BuildHistLeftRight(candidate, left_child_nidx, right_child_nidx, reducer);
          monitor.Stop("BuildHist");

          monitor.Start("EvaluateSplits");
          this->EvaluateLeftRightSplits(candidate, left_child_nidx,
                                        right_child_nidx, *p_tree,
                                        new_candidates.subspan(i * 2, 2));
          monitor.Stop("EvaluateSplits");
        } else {
          // Set default
          new_candidates[i * 2] = ExpandEntry();
          new_candidates[i * 2 + 1] = ExpandEntry();
        }
      }
      dh::safe_cuda(cudaDeviceSynchronize());
      driver.Push(new_candidates.begin(), new_candidates.end());
      expand_set = driver.Pop();
    }

    monitor.Start("FinalisePosition");
    this->FinalisePosition(p_tree, p_fmat);
    monitor.Stop("FinalisePosition");
  }
};

template <typename GradientSumT>
class GPUHistMakerSpecialised {
 public:
  GPUHistMakerSpecialised() = default;
  void Configure(const Args& args, GenericParameter const* generic_param) {
    param_.UpdateAllowUnknown(args);
    generic_param_ = generic_param;
    hist_maker_param_.UpdateAllowUnknown(args);
    dh::CheckComputeCapability();

    monitor_.Init("updater_gpu_hist");
  }

  ~GPUHistMakerSpecialised() {  // NOLINT
    dh::GlobalMemoryLogger().Log();
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) {
    monitor_.Start("Update");

    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    ValueConstraint::Init(&param_, dmat->Info().num_col_);
    // build tree
    try {
      for (xgboost::RegTree* tree : trees) {
        this->UpdateTree(gpair, dmat, tree);

        if (hist_maker_param_.debug_synchronize) {
          this->CheckTreesSynchronized(tree);
        }
      }
      dh::safe_cuda(cudaGetLastError());
    } catch (const std::exception& e) {
      LOG(FATAL) << "Exception in gpu_hist: " << e.what() << std::endl;
    }

    param_.learning_rate = lr;
    monitor_.Stop("Update");
  }

  void InitDataOnce(DMatrix* dmat) {
    device_ = generic_param_->gpu_id;
    CHECK_GE(device_, 0) << "Must have at least one device";
    info_ = &dmat->Info();
    reducer_.Init({device_});  // NOLINT

    // Synchronise the column sampling seed
    uint32_t column_sampling_seed = common::GlobalRandom()();
    rabit::Broadcast(&column_sampling_seed, sizeof(column_sampling_seed), 0);

    BatchParam batch_param{
      device_,
      param_.max_bin,
      generic_param_->gpu_page_size
    };
    auto page = (*dmat->GetBatches<EllpackPage>(batch_param).begin()).Impl();
    dh::safe_cuda(cudaSetDevice(device_));
    maker.reset(new GPUHistMakerDevice<GradientSumT>(device_,
                                                     page,
                                                     info_->num_row_,
                                                     param_,
                                                     column_sampling_seed,
                                                     info_->num_col_,
                                                     hist_maker_param_.deterministic_histogram,
                                                     batch_param));

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(DMatrix* dmat) {
    if (!initialised_) {
      monitor_.Start("InitDataOnce");
      this->InitDataOnce(dmat);
      monitor_.Stop("InitDataOnce");
    }
  }

  // Only call this method for testing
  void CheckTreesSynchronized(RegTree* local_tree) const {
    std::string s_model;
    common::MemoryBufferStream fs(&s_model);
    int rank = rabit::GetRank();
    if (rank == 0) {
      local_tree->Save(&fs);
    }
    fs.Seek(0);
    rabit::Broadcast(&s_model, 0);
    RegTree reference_tree {};  // rank 0 tree
    reference_tree.Load(&fs);
    CHECK(*local_tree == reference_tree);
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat,
                  RegTree* p_tree) {
    monitor_.Start("InitData");
    this->InitData(p_fmat);
    monitor_.Stop("InitData");

    gpair->SetDevice(device_);
    maker->UpdateTree(gpair, p_fmat, p_tree, &reducer_);
  }

  bool UpdatePredictionCache(const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) {
    if (maker == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.Start("UpdatePredictionCache");
    p_out_preds->SetDevice(device_);
    maker->UpdatePredictionCache(p_out_preds->DevicePointer());
    monitor_.Stop("UpdatePredictionCache");
    return true;
  }

  TrainParam param_;   // NOLINT
  MetaInfo* info_{};   // NOLINT

  std::unique_ptr<GPUHistMakerDevice<GradientSumT>> maker;  // NOLINT

 private:
  bool initialised_ { false };

  GPUHistMakerTrainParam hist_maker_param_;
  GenericParameter const* generic_param_;

  dh::AllReducer reducer_;

  DMatrix* p_last_fmat_ { nullptr };
  int device_{-1};

  common::Monitor monitor_;
};

class GPUHistMaker : public TreeUpdater {
 public:
  void Configure(const Args& args) override {
    // Used in test to count how many configurations are performed
    LOG(DEBUG) << "[GPU Hist]: Configure";
    hist_maker_param_.UpdateAllowUnknown(args);
    // The passed in args can be empty, if we simply purge the old maker without
    // preserving parameters then we can't do Update on it.
    TrainParam param;
    if (float_maker_) {
      param = float_maker_->param_;
    } else if (double_maker_) {
      param = double_maker_->param_;
    }
    if (hist_maker_param_.single_precision_histogram) {
      float_maker_.reset(new GPUHistMakerSpecialised<GradientPair>());
      float_maker_->param_ = param;
      float_maker_->Configure(args, tparam_);
    } else {
      double_maker_.reset(new GPUHistMakerSpecialised<GradientPairPrecise>());
      double_maker_->param_ = param;
      double_maker_->Configure(args, tparam_);
    }
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("gpu_hist_train_param"), &this->hist_maker_param_);
    if (hist_maker_param_.single_precision_histogram) {
      float_maker_.reset(new GPUHistMakerSpecialised<GradientPair>());
      FromJson(config.at("train_param"), &float_maker_->param_);
    } else {
      double_maker_.reset(new GPUHistMakerSpecialised<GradientPairPrecise>());
      FromJson(config.at("train_param"), &double_maker_->param_);
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["gpu_hist_train_param"] = ToJson(hist_maker_param_);
    if (hist_maker_param_.single_precision_histogram) {
      out["train_param"] = ToJson(float_maker_->param_);
    } else {
      out["train_param"] = ToJson(double_maker_->param_);
    }
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    if (hist_maker_param_.single_precision_histogram) {
      float_maker_->Update(gpair, dmat, trees);
    } else {
      double_maker_->Update(gpair, dmat, trees);
    }
  }

  bool UpdatePredictionCache(
      const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) override {
    if (hist_maker_param_.single_precision_histogram) {
      return float_maker_->UpdatePredictionCache(data, p_out_preds);
    } else {
      return double_maker_->UpdatePredictionCache(data, p_out_preds);
    }
  }

  char const* Name() const override {
    return "grow_gpu_hist";
  }

 private:
  GPUHistMakerTrainParam hist_maker_param_;
  std::unique_ptr<GPUHistMakerSpecialised<GradientPair>> float_maker_;
  std::unique_ptr<GPUHistMakerSpecialised<GradientPairPrecise>> double_maker_;
};

#if !defined(GTEST_TEST)
XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
#endif  // !defined(GTEST_TEST)

}  // namespace tree
}  // namespace xgboost
