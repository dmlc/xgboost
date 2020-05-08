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

#define PRECISE 1

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

struct ExpandEntry {
  int nid;
  int depth;
  DeviceSplitCandidate split;
  uint64_t timestamp;
  ExpandEntry() = default;
  ExpandEntry(int nid, int depth, DeviceSplitCandidate split,
              uint64_t timestamp)
      : nid(nid), depth(depth), split(std::move(split)), timestamp(timestamp) {}
  bool IsValid(const TrainParam& param, int num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0) {
      return false;
    }
    if (split.loss_chg < param.min_split_loss) { return false; }
    if (param.max_depth > 0 && depth == param.max_depth) {return false; }
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) { return false; }
    return true;
  }

  static bool ChildIsValid(const TrainParam& param, int depth, int num_leaves) {
    if (param.max_depth > 0 && depth >= param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves >= param.max_leaves) return false;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const ExpandEntry& e) {
    os << "ExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "left_sum: " << e.split.left_sum << "\n";
    os << "right_sum: " << e.split.right_sum << "\n";
    return os;
  }
};

inline static bool DepthWise(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  if (lhs.depth == rhs.depth) {
    return lhs.timestamp > rhs.timestamp;  // favor small timestamp
  } else {
    return lhs.depth > rhs.depth;  // favor small depth
  }
}
inline static bool LossGuide(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  if (lhs.split.loss_chg == rhs.split.loss_chg) {
    return lhs.timestamp > rhs.timestamp;  // favor small timestamp
  } else {
    return lhs.split.loss_chg < rhs.split.loss_chg;  // favor large loss_chg
  }
}

// With constraints
template <typename GradientPairT>
XGBOOST_DEVICE float inline LossChangeMissing(
    const GradientPairT& scan, const GradientPairT& missing, const GradientPairT& parent_sum,
    const float& parent_gain, const GPUTrainingParam& param, int constraint,
    const ValueConstraint& value_constraint,
    bool& missing_left_out) {  // NOLINT
  float missing_left_gain = value_constraint.CalcSplitGain(
      param, constraint, GradStats(scan + missing),
      GradStats(parent_sum - (scan + missing)));
  float missing_right_gain = value_constraint.CalcSplitGain(
      param, constraint, GradStats(scan), GradStats(parent_sum - scan));

  if (missing_left_gain >= missing_right_gain) {
    missing_left_out = true;
    return missing_left_gain - parent_gain;
  } else {
    missing_left_out = false;
    return missing_right_gain - parent_gain;
  }
}

/*!
 * \brief
 *
 * \tparam ReduceT     BlockReduce Type.
 * \tparam TempStorage Cub Shared memory
 *
 * \param begin
 * \param end
 * \param temp_storage Shared memory for intermediate result.
 */
template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT, typename GradientSumT>
__device__ GradientSumT ReduceFeature(common::Span<const GradientSumT> feature_histogram,
                                      TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientSumT> uninitialized_sum;
  GradientSumT& shared_sum = uninitialized_sum.Alias();

  GradientSumT local_sum = GradientSumT();
  // For loop sums features into one block size
  auto begin = feature_histogram.data();
  auto end = begin + feature_histogram.size();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientSumT bin = thread_active ? *(itr + threadIdx.x) : GradientSumT();
    local_sum += bin;
  }
  local_sum = ReduceT(temp_storage->sum_reduce).Reduce(local_sum, cub::Sum());
  // Reduction result is stored in thread 0.
  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();
  return shared_sum;
}

/*! \brief Find the thread with best gain. */
template <int BLOCK_THREADS, typename ReduceT, typename ScanT,
          typename MaxReduceT, typename TempStorageT, typename GradientSumT>
__device__ void EvaluateFeature(
    int fidx, common::Span<const GradientSumT> node_histogram,
    const EllpackDeviceAccessor& matrix,
    DeviceSplitCandidate* best_split,  // shared memory storing best split
    const DeviceNodeStats& node, const GPUTrainingParam& param,
    TempStorageT* temp_storage,  // temp memory for cub operations
    int constraint,              // monotonic_constraints
    const ValueConstraint& value_constraint) {
  // Use pointer from cut to indicate begin and end of bins for each feature.
  uint32_t gidx_begin = matrix.feature_segments[fidx];  // begining bin
  uint32_t gidx_end = matrix.feature_segments[fidx + 1];  // end bin for i^th feature

  // Sum histogram bins for current feature
  GradientSumT const feature_sum = ReduceFeature<BLOCK_THREADS, ReduceT>(
      node_histogram.subspan(gidx_begin, gidx_end - gidx_begin), temp_storage);

  GradientSumT const parent_sum = GradientSumT(node.sum_gradients);
  GradientSumT const missing = parent_sum - feature_sum;
  float const null_gain = -std::numeric_limits<bst_float>::infinity();

  SumCallbackOp<GradientSumT> prefix_op =
      SumCallbackOp<GradientSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = (scan_begin + threadIdx.x) < gidx_end;

    // Gradient value for current bin.
    GradientSumT bin =
        thread_active ? node_histogram[scan_begin + threadIdx.x] : GradientSumT();
    ScanT(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Whether the gradient of missing values is put to the left side.
    bool missing_left = true;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(bin, missing, parent_sum, node.root_gain, param,
                               constraint, value_constraint, missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        MaxReduceT(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

    __shared__ cub::KeyValuePair<int, float> block_max;
    if (threadIdx.x == 0) {
      block_max = best;
    }

    __syncthreads();

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      int split_gidx = (scan_begin + threadIdx.x) - 1;
      float fvalue;
      if (split_gidx < static_cast<int>(gidx_begin)) {
        fvalue =  matrix.min_fvalue[fidx];
      } else {
        fvalue = matrix.gidx_fvalue_map[split_gidx];
      }
      GradientSumT left = missing_left ? bin + missing : bin;
      GradientSumT right = parent_sum - left;
      best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue,
                         fidx, GradientPair(left), GradientPair(right), param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS, typename GradientSumT>
__global__ void EvaluateSplitKernel(
    common::Span<const GradientSumT> node_histogram,  // histogram for gradients
    common::Span<const bst_feature_t> feature_set,    // Selected features
    DeviceNodeStats node,
    xgboost::EllpackDeviceAccessor matrix,
    GPUTrainingParam gpu_param,
    common::Span<DeviceSplitCandidate> split_candidates,  // resulting split
    ValueConstraint value_constraint,
    common::Span<int> d_monotonic_constraints) {
  // KeyValuePair here used as threadIdx.x -> gain_value
  using ArgMaxT = cub::KeyValuePair<int, float>;
  using BlockScanT =
      cub::BlockScan<GradientSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
  using MaxReduceT = cub::BlockReduce<ArgMaxT, BLOCK_THREADS>;

  using SumReduceT = cub::BlockReduce<GradientSumT, BLOCK_THREADS>;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  // Aligned && shared storage for best_split
  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate& best_split = uninitialized_split.Alias();
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  // One block for each feature. Features are sampled, so fidx != blockIdx.x
  int fidx = feature_set[blockIdx.x];

  int constraint = d_monotonic_constraints[fidx];
  EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT>(
      fidx, node_histogram, matrix, &best_split, node, gpu_param, &temp_storage,
      constraint, value_constraint);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss for each feature
    split_candidates[blockIdx.x] = best_split;
  }
}

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

  /*! \brief Gradient pair for each row. */
  common::Span<GradientPair> gpair;

  dh::caching_device_vector<int> monotone_constraints;
  dh::caching_device_vector<bst_float> prediction_cache;

  /*! \brief Sum gradient for each node. */
  std::vector<GradientPair> host_node_sum_gradients;
  dh::caching_device_vector<GradientPair> node_sum_gradients;
  bst_uint n_rows;

  TrainParam param;
  bool deterministic_histogram;

  GradientSumT histogram_rounding;

  dh::PinnedMemory pinned_memory;

  std::vector<cudaStream_t> streams{};

  common::Monitor monitor;
  std::vector<ValueConstraint> node_value_constraints;
  common::ColumnSampler column_sampler;
  FeatureInteractionConstraintDevice interaction_constraints;

  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;
  std::unique_ptr<ExpandQueue> qexpand;

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
        n_rows(_n_rows),
        param(std::move(_param)),
        column_sampler(column_sampler_seed),
        interaction_constraints(param, n_features),
        deterministic_histogram{deterministic_histogram},
        batch_param(_batch_param) {
    sampler.reset(new GradientBasedSampler(page,
                                           n_rows,
                                           batch_param,
                                           param.subsample,
                                           param.sampling_method));
    monitor.Init(std::string("GPUHistMakerDevice") + std::to_string(device_id));
  }

  void InitHistogram();

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
    if (param.grow_policy == TrainParam::kLossGuide) {
      qexpand.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand.reset(new ExpandQueue(DepthWise));
    }
    this->column_sampler.Init(num_columns, param.colsample_bynode,
      param.colsample_bylevel, param.colsample_bytree);
    dh::safe_cuda(cudaSetDevice(device_id));
    this->interaction_constraints.Reset();
    std::fill(host_node_sum_gradients.begin(), host_node_sum_gradients.end(),
              GradientPair());

    auto sample = sampler->Sample(dh_gpair->DeviceSpan(), dmat);
    n_rows = sample.sample_rows;
    page = sample.page;
    gpair = sample.gpair;

    if (deterministic_histogram) {
      histogram_rounding = CreateRoundingFactor<GradientSumT>(this->gpair);
    } else {
      histogram_rounding = GradientSumT{0.0, 0.0};
    }

    row_partitioner.reset();  // Release the device memory first before reallocating
    row_partitioner.reset(new RowPartitioner(device_id, n_rows));
    hist.Reset();
  }

  std::vector<DeviceSplitCandidate> EvaluateSplits(
      std::vector<int> nidxs, const RegTree& tree,
      size_t num_columns) {
    auto result_all = pinned_memory.GetSpan<DeviceSplitCandidate>(nidxs.size());

    // Work out cub temporary memory requirement
    GPUTrainingParam gpu_param(param);
    DeviceSplitCandidateReduceOp op(gpu_param);

    dh::TemporaryArray<DeviceSplitCandidate> d_result_all(nidxs.size());
    dh::TemporaryArray<DeviceSplitCandidate> split_candidates_all(nidxs.size()*num_columns);

    auto& streams = this->GetStreams(nidxs.size());
    for (auto i = 0ull; i < nidxs.size(); i++) {
      auto nidx = nidxs[i];
      auto p_feature_set = column_sampler.GetFeatureSet(tree.GetDepth(nidx));
      p_feature_set->SetDevice(device_id);
      common::Span<bst_feature_t> d_sampled_features =
          p_feature_set->DeviceSpan();
      common::Span<bst_feature_t> d_feature_set =
          interaction_constraints.Query(d_sampled_features, nidx);
      common::Span<DeviceSplitCandidate> d_split_candidates(
          split_candidates_all.data().get() + i * num_columns,
          d_feature_set.size());

      DeviceNodeStats node(host_node_sum_gradients[nidx], nidx, param);

      common::Span<DeviceSplitCandidate> d_result(d_result_all.data().get() + i, 1);
      if (d_feature_set.empty()) {
        // Acting as a device side constructor for DeviceSplitCandidate.
        // DeviceSplitCandidate::IsValid is false so that ApplySplit can reject this
        // candidate.
        auto worst_candidate = DeviceSplitCandidate();
        dh::safe_cuda(cudaMemcpyAsync(d_result.data(), &worst_candidate,
                                      sizeof(DeviceSplitCandidate),
                                      cudaMemcpyHostToDevice));
        continue;
      }

      // One block for each feature
      uint32_t constexpr kBlockThreads = 256;
      dh::LaunchKernel {uint32_t(d_feature_set.size()), kBlockThreads, 0, streams[i]} (
          EvaluateSplitKernel<kBlockThreads, GradientSumT>,
          hist.GetNodeHistogram(nidx), d_feature_set, node, page->GetDeviceAccessor(device_id),
          gpu_param, d_split_candidates, node_value_constraints[nidx],
          dh::ToSpan(monotone_constraints));

      // Reduce over features to find best feature
      size_t cub_bytes = 0;
      cub::DeviceReduce::Reduce(nullptr,
                                cub_bytes, d_split_candidates.data(),
                                d_result.data(), d_split_candidates.size(), op,
                                DeviceSplitCandidate(), streams[i]);
      dh::TemporaryArray<char> cub_temp(cub_bytes);
      cub::DeviceReduce::Reduce(reinterpret_cast<void*>(cub_temp.data().get()),
                                cub_bytes, d_split_candidates.data(),
                                d_result.data(), d_split_candidates.size(), op,
                                DeviceSplitCandidate(), streams[i]);
    }

    dh::safe_cuda(cudaMemcpy(result_all.data(), d_result_all.data().get(),
                             sizeof(DeviceSplitCandidate) * d_result_all.size(),
                             cudaMemcpyDeviceToHost));
    return std::vector<DeviceSplitCandidate>(result_all.begin(), result_all.end());
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

    dh::safe_cuda(
        cudaMemcpyAsync(node_sum_gradients.data().get(), host_node_sum_gradients.data(),
                        sizeof(GradientPair) * host_node_sum_gradients.size(),
                        cudaMemcpyHostToDevice));
    auto d_position = row_partitioner->GetPosition();
    auto d_node_sum_gradients = node_sum_gradients.data().get();
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
    monitor.StartCuda("AllReduce");
    auto d_node_hist = hist.GetNodeHistogram(nidx).data();
    reducer->AllReduceSum(
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        reinterpret_cast<typename GradientSumT::ValueT*>(d_node_hist),
        page->Cuts().TotalBins() * (sizeof(GradientSumT) / sizeof(typename GradientSumT::ValueT)));
    reducer->Synchronize();

    monitor.StopCuda("AllReduce");
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

    GradStats left_stats{};
    left_stats.Add(candidate.split.left_sum);
    GradStats right_stats{};
    right_stats.Add(candidate.split.right_sum);
    GradStats parent_sum{};
    parent_sum.Add(left_stats);
    parent_sum.Add(right_stats);
    node_value_constraints.resize(tree.GetNodes().size());
    auto base_weight = node_value_constraints[candidate.nid].CalcWeight(param, parent_sum);
    auto left_weight =
        node_value_constraints[candidate.nid].CalcWeight(param, left_stats)*param.learning_rate;
    auto right_weight =
        node_value_constraints[candidate.nid].CalcWeight(param, right_stats)*param.learning_rate;
    tree.ExpandNode(candidate.nid, candidate.split.findex,
                    candidate.split.fvalue, candidate.split.dir == kLeftDir,
                    base_weight, left_weight, right_weight,
                    candidate.split.loss_chg, parent_sum.sum_hess,
                    left_stats.GetHess(), right_stats.GetHess());
    // Set up child constraints
    node_value_constraints.resize(tree.GetNodes().size());
    node_value_constraints[candidate.nid].SetChild(
        param, tree[candidate.nid].SplitIndex(), left_stats, right_stats,
        &node_value_constraints[tree[candidate.nid].LeftChild()],
        &node_value_constraints[tree[candidate.nid].RightChild()]);
    host_node_sum_gradients[tree[candidate.nid].LeftChild()] =
        candidate.split.left_sum;
    host_node_sum_gradients[tree[candidate.nid].RightChild()] =
        candidate.split.right_sum;

    interaction_constraints.Split(candidate.nid, tree[candidate.nid].SplitIndex(),
                                  tree[candidate.nid].LeftChild(),
                                  tree[candidate.nid].RightChild());
  }

  void InitRoot(RegTree* p_tree, dh::AllReducer* reducer, int64_t num_columns) {
    constexpr bst_node_t kRootNIdx = 0;
    dh::XGBCachingDeviceAllocator<char> alloc;
    GradientPair root_sum = thrust::reduce(
        thrust::cuda::par(alloc),
        thrust::device_ptr<GradientPair const>(gpair.data()),
        thrust::device_ptr<GradientPair const>(gpair.data() + gpair.size()));
    dh::safe_cuda(cudaMemcpyAsync(node_sum_gradients.data().get(), &root_sum, sizeof(root_sum),
                                  cudaMemcpyHostToDevice));
    reducer->AllReduceSum(
        reinterpret_cast<float*>(node_sum_gradients.data().get()),
        reinterpret_cast<float*>(node_sum_gradients.data().get()), 2);
    reducer->Synchronize();
    dh::safe_cuda(cudaMemcpyAsync(host_node_sum_gradients.data(),
                                  node_sum_gradients.data().get(), sizeof(GradientPair),
                                  cudaMemcpyDeviceToHost));

    this->BuildHist(kRootNIdx);
    this->AllReduceHist(kRootNIdx, reducer);

    // Remember root stats
    p_tree->Stat(kRootNIdx).sum_hess = host_node_sum_gradients[kRootNIdx].GetHess();
    auto weight = CalcWeight(param, host_node_sum_gradients[kRootNIdx]);
    p_tree->Stat(kRootNIdx).base_weight = weight;
    (*p_tree)[kRootNIdx].SetLeaf(param.learning_rate * weight);

    // Initialise root constraint
    node_value_constraints.resize(p_tree->GetNodes().size());

    // Generate first split
    auto split = this->EvaluateSplits({kRootNIdx}, *p_tree, num_columns);
    qexpand->push(
        ExpandEntry(kRootNIdx, p_tree->GetDepth(kRootNIdx), split.at(0), 0));
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair_all, DMatrix* p_fmat,
                  RegTree* p_tree, dh::AllReducer* reducer) {
    auto& tree = *p_tree;

    monitor.StartCuda("Reset");
    this->Reset(gpair_all, p_fmat, p_fmat->Info().num_col_);
    monitor.StopCuda("Reset");

    monitor.StartCuda("InitRoot");
    this->InitRoot(p_tree, reducer, p_fmat->Info().num_col_);
    monitor.StopCuda("InitRoot");

    auto timestamp = qexpand->size();
    auto num_leaves = 1;

    while (!qexpand->empty()) {
      ExpandEntry candidate = qexpand->top();
      qexpand->pop();
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
        monitor.StartCuda("UpdatePosition");
        this->UpdatePosition(candidate.nid, (*p_tree)[candidate.nid]);
        monitor.StopCuda("UpdatePosition");

        monitor.StartCuda("BuildHist");
        this->BuildHistLeftRight(candidate, left_child_nidx, right_child_nidx, reducer);
        monitor.StopCuda("BuildHist");

        monitor.StartCuda("EvaluateSplits");
        auto splits = this->EvaluateSplits({left_child_nidx, right_child_nidx},
                                           *p_tree, p_fmat->Info().num_col_);
        monitor.StopCuda("EvaluateSplits");

        qexpand->push(ExpandEntry(left_child_nidx,
                                   tree.GetDepth(left_child_nidx), splits.at(0),
                                   timestamp++));
        qexpand->push(ExpandEntry(right_child_nidx,
                                   tree.GetDepth(right_child_nidx),
                                   splits.at(1), timestamp++));
      }
    }

    monitor.StartCuda("FinalisePosition");
    this->FinalisePosition(p_tree, p_fmat);
    monitor.StopCuda("FinalisePosition");
  }
};

template <typename GradientSumT>
inline void GPUHistMakerDevice<GradientSumT>::InitHistogram() {
  if (!param.monotone_constraints.empty()) {
    // Copy assigning an empty vector causes an exception in MSVC debug builds
    monotone_constraints = param.monotone_constraints;
  }
  host_node_sum_gradients.resize(param.MaxNodes());
  node_sum_gradients.resize(param.MaxNodes());

  // Init histogram
  hist.Init(device_id, page->Cuts().TotalBins());
}

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
    monitor_.StartCuda("Update");

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
    monitor_.StopCuda("Update");
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

    monitor_.StartCuda("InitHistogram");
    dh::safe_cuda(cudaSetDevice(device_));
    maker->InitHistogram();
    monitor_.StopCuda("InitHistogram");

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(DMatrix* dmat) {
    if (!initialised_) {
      monitor_.StartCuda("InitDataOnce");
      this->InitDataOnce(dmat);
      monitor_.StopCuda("InitDataOnce");
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
    monitor_.StartCuda("InitData");
    if (!initialised_ && PRECISE==0)
      CheckGradientMax(gpair);

    this->InitData(p_fmat);
    monitor_.StopCuda("InitData");

    gpair->SetDevice(device_);
    maker->UpdateTree(gpair, p_fmat, p_tree, &reducer_);
  }

  bool UpdatePredictionCache(const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) {
    if (maker == nullptr || p_last_fmat_ == nullptr || p_last_fmat_ != data) {
      return false;
    }
    monitor_.StartCuda("UpdatePredictionCache");
    p_out_preds->SetDevice(device_);
    maker->UpdatePredictionCache(p_out_preds->DevicePointer());
    monitor_.StopCuda("UpdatePredictionCache");
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
