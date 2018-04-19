/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <xgboost/tree_updater.h>
#include <algorithm>
#include <memory>
#include <queue>
#include <utility>
#include <vector>
#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"
#include "../common/host_device_vector.h"
#include "../common/timer.h"
#include "param.h"
#include "updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);

using GradientPairSumT = GradientPairPrecise;

template <int BLOCK_THREADS, typename ReduceT, typename TempStorageT>
__device__ GradientPairSumT ReduceFeature(const GradientPairSumT* begin,
                                     const GradientPairSumT* end,
                                     TempStorageT* temp_storage) {
  __shared__ cub::Uninitialized<GradientPairSumT> uninitialized_sum;
  GradientPairSumT& shared_sum = uninitialized_sum.Alias();

  GradientPairSumT local_sum = GradientPairSumT();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    GradientPairSumT bin = thread_active ? *(itr + threadIdx.x) : GradientPairSumT();

    local_sum += ReduceT(temp_storage->sum_reduce).Reduce(bin, cub::Sum());
  }

  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();

  return shared_sum;
}

template <int BLOCK_THREADS, typename ReduceT, typename scan_t,
          typename max_ReduceT, typename TempStorageT>
__device__ void EvaluateFeature(int fidx, const GradientPairSumT* hist,
                                const int* feature_segments, float min_fvalue,
                                const float* gidx_fvalue_map,
                                DeviceSplitCandidate* best_split,
                                const DeviceNodeStats& node,
                                const GPUTrainingParam& param,
                                TempStorageT* temp_storage, int constraint,
                                const ValueConstraint& value_constraint) {
  int gidx_begin = feature_segments[fidx];
  int gidx_end = feature_segments[fidx + 1];

  GradientPairSumT feature_sum = ReduceFeature<BLOCK_THREADS, ReduceT>(
      hist + gidx_begin, hist + gidx_end, temp_storage);

  auto prefix_op = SumCallbackOp<GradientPairSumT>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = scan_begin + threadIdx.x < gidx_end;

    GradientPairSumT bin =
        thread_active ? hist[scan_begin + threadIdx.x] : GradientPairSumT();
    scan_t(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Calculate  gain
    GradientPairSumT parent_sum = GradientPairSumT(node.sum_gradients);

    GradientPairSumT missing = parent_sum - feature_sum;

    bool missing_left = true;
    const float null_gain = -FLT_MAX;
    float gain = null_gain;
    if (thread_active) {
      gain = LossChangeMissing(bin, missing, parent_sum, node.root_gain, param,
                              constraint, value_constraint, missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        max_ReduceT(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

    __shared__ cub::KeyValuePair<int, float> block_max;
    if (threadIdx.x == 0) {
      block_max = best;
    }

    __syncthreads();

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      int gidx = scan_begin + threadIdx.x;
      float fvalue =
          gidx == gidx_begin ? min_fvalue : gidx_fvalue_map[gidx - 1];

      GradientPairSumT left = missing_left ? bin + missing : bin;
      GradientPairSumT right = parent_sum - left;

      best_split->Update(gain, missing_left ? kLeftDir : kRightDir, fvalue, fidx,
                         GradientPair(left), GradientPair(right), param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS>
__global__ void evaluate_split_kernel(
    const GradientPairSumT* d_hist, int nidx, uint64_t n_features,
    DeviceNodeStats nodes, const int* d_feature_segments,
    const float* d_fidx_min_map, const float* d_gidx_fvalue_map,
    GPUTrainingParam gpu_param, DeviceSplitCandidate* d_split,
    ValueConstraint value_constraint, int* d_monotonic_constraints) {
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<GradientPairSumT, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;

  typedef cub::BlockReduce<GradientPairSumT, BLOCK_THREADS> SumReduceT;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  __shared__ cub::Uninitialized<DeviceSplitCandidate> uninitialized_split;
  DeviceSplitCandidate& best_split = uninitialized_split.Alias();
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    best_split = DeviceSplitCandidate();
  }

  __syncthreads();

  auto fidx = blockIdx.x;
  auto constraint = d_monotonic_constraints[fidx];
  EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT>(
      fidx, d_hist, d_feature_segments, d_fidx_min_map[fidx], d_gidx_fvalue_map,
      &best_split, nodes, gpu_param, &temp_storage, constraint,
      value_constraint);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss
    d_split[fidx] = best_split;
  }
}

// Find a gidx value for a given feature otherwise return -1 if not found
template <typename GidxIterT>
__device__ int BinarySearchRow(bst_uint begin, bst_uint end, GidxIterT data,
                               int fidx_begin, int fidx_end) {
  bst_uint previous_middle = UINT32_MAX;
  while (end != begin) {
    auto middle = begin + (end - begin) / 2;
    if (middle == previous_middle) {
      break;
    }
    previous_middle = middle;

    auto gidx = data[middle];

    if (gidx >= fidx_begin && gidx < fidx_end) {
      return gidx;
    } else if (gidx < fidx_begin) {
      begin = middle;
    } else {
      end = middle;
    }
  }
  // Value is missing
  return -1;
}

struct DeviceHistogram {
  dh::BulkAllocator<dh::MemoryType::kDevice> ba;
  dh::DVec<GradientPairSumT> data;
  int n_bins;
  void Init(int device_idx, int max_nodes, int n_bins, bool silent) {
    this->n_bins = n_bins;
    ba.Allocate(device_idx, silent, &data, size_t(max_nodes) * size_t(n_bins));
  }

  void Reset() { data.Fill(GradientPairSumT()); }
  GradientPairSumT* GetHistPtr(int nidx) { return data.Data() + nidx * n_bins; }

  void PrintNidx(int nidx) const {
    auto h_data = data.AsVector();
    std::cout << "nidx " << nidx << ":\n";
    for (int i = n_bins * nidx; i < n_bins * (nidx + 1); i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << "\n";
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
struct DeviceShard {
  struct Segment {
    size_t begin;
    size_t end;

    Segment() : begin(0), end(0) {}

    Segment(size_t begin, size_t end) : begin(begin), end(end) {
      CHECK_GE(end, begin);
    }
    size_t Size() const { return end - begin; }
  };

  int device_idx;
  int normalised_device_idx;  // Device index counting from param.gpu_id
  dh::BulkAllocator<dh::MemoryType::kDevice> ba;
  dh::DVec<common::CompressedByteT> gidx_buffer;
  dh::DVec<GradientPair> gpair;
  dh::DVec2<bst_uint> ridx;  // Row index relative to this shard
  dh::DVec2<int> position;
  std::vector<Segment> ridx_segments;
  dh::DVec<int> feature_segments;
  dh::DVec<float> gidx_fvalue_map;
  dh::DVec<float> min_fvalue;
  dh::DVec<int> monotone_constraints;
  dh::DVec<bst_float> prediction_cache;
  std::vector<GradientPair> node_sum_gradients;
  dh::DVec<GradientPair> node_sum_gradients_d;
  common::CompressedIterator<uint32_t> gidx;
  int row_stride;
  bst_uint row_begin_idx;  // The row offset for this shard
  bst_uint row_end_idx;
  bst_uint n_rows;
  int n_bins;
  int null_gidx_value;
  DeviceHistogram hist;
  TrainParam param;
  bool prediction_cache_initialised;

  int64_t* tmp_pinned;  // Small amount of staging memory

  std::vector<cudaStream_t> streams;

  dh::CubMemory temp_memory;

  DeviceShard(int device_idx, int normalised_device_idx,
              const common::GHistIndexMatrix& gmat, bst_uint row_begin,
              bst_uint row_end, int n_bins, TrainParam param)
      : device_idx(device_idx),
        normalised_device_idx(normalised_device_idx),
        row_begin_idx(row_begin),
        row_end_idx(row_end),
        n_rows(row_end - row_begin),
        n_bins(n_bins),
        null_gidx_value(n_bins),
        param(param),
        prediction_cache_initialised(false) {
    // Convert to ELLPACK matrix representation
    int max_elements_row = 0;
    for (auto i = row_begin; i < row_end; i++) {
      max_elements_row =
          (std::max)(max_elements_row,
                     static_cast<int>(gmat.row_ptr[i + 1] - gmat.row_ptr[i]));
    }
    row_stride = max_elements_row;
    std::vector<int> ellpack_matrix(row_stride * n_rows, null_gidx_value);

    for (auto i = row_begin; i < row_end; i++) {
      int row_count = 0;
      for (auto j = gmat.row_ptr[i]; j < gmat.row_ptr[i + 1]; j++) {
        ellpack_matrix[(i - row_begin) * row_stride + row_count] =
            gmat.index[j];
        row_count++;
      }
    }

    // Allocate
    int num_symbols = n_bins + 1;
    size_t compressed_size_bytes =
        common::CompressedBufferWriter::CalculateBufferSize(
            ellpack_matrix.size(), num_symbols);

    CHECK(!(param.max_leaves == 0 && param.max_depth == 0))
        << "Max leaves and max depth cannot both be unconstrained for "
           "gpu_hist.";
    int max_nodes =
        param.max_leaves > 0 ? param.max_leaves * 2 : MaxNodesDepth(param.max_depth);
    ba.Allocate(device_idx, param.silent, &gidx_buffer, compressed_size_bytes,
                &gpair, n_rows, &ridx, n_rows, &position, n_rows,
                &prediction_cache, n_rows, &node_sum_gradients_d, max_nodes,
                &feature_segments, gmat.cut->row_ptr.size(), &gidx_fvalue_map,
                gmat.cut->cut.size(), &min_fvalue, gmat.cut->min_val.size(),
                &monotone_constraints, param.monotone_constraints.size());
    gidx_fvalue_map = gmat.cut->cut;
    min_fvalue = gmat.cut->min_val;
    feature_segments = gmat.cut->row_ptr;
    monotone_constraints = param.monotone_constraints;

    node_sum_gradients.resize(max_nodes);
    ridx_segments.resize(max_nodes);

    // Compress gidx
    common::CompressedBufferWriter cbw(num_symbols);
    std::vector<common::CompressedByteT> host_buffer(gidx_buffer.Size());
    cbw.Write(host_buffer.data(), ellpack_matrix.begin(), ellpack_matrix.end());
    gidx_buffer = host_buffer;
    gidx =
        common::CompressedIterator<uint32_t>(gidx_buffer.Data(), num_symbols);

    common::CompressedIterator<uint32_t> ci_host(host_buffer.data(),
                                                 num_symbols);

    // Init histogram
    hist.Init(device_idx, max_nodes, gmat.cut->row_ptr.back(), param.silent);

    dh::safe_cuda(cudaMallocHost(&tmp_pinned, sizeof(int64_t)));
  }

  ~DeviceShard() {
    for (auto& stream : streams) {
      dh::safe_cuda(cudaStreamDestroy(stream));
    }
    dh::safe_cuda(cudaFreeHost(tmp_pinned));
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
  void Reset(HostDeviceVector<GradientPair>* dh_gpair, int device) {
    auto begin = dh_gpair->tbegin(device);
    dh::safe_cuda(cudaSetDevice(device_idx));
    position.CurrentDVec().Fill(0);
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(),
              GradientPair());

    thrust::sequence(ridx.CurrentDVec().tbegin(), ridx.CurrentDVec().tend());

    std::fill(ridx_segments.begin(), ridx_segments.end(), Segment(0, 0));
    ridx_segments.front() = Segment(0, ridx.Size());
    this->gpair.copy(begin + row_begin_idx, begin + row_end_idx);
    SubsampleGradientPair(&gpair, param.subsample, row_begin_idx);
    hist.Reset();
  }

  void BuildHist(int nidx) {
    auto segment = ridx_segments[nidx];
    auto d_node_hist = hist.GetHistPtr(nidx);
    auto d_gidx = gidx;
    auto d_ridx = ridx.Current();
    auto d_gpair = gpair.Data();
    auto row_stride = this->row_stride;
    auto null_gidx_value = this->null_gidx_value;
    auto n_elements = segment.Size() * row_stride;

    dh::LaunchN(device_idx, n_elements, [=] __device__(size_t idx) {
      int ridx = d_ridx[(idx / row_stride) + segment.begin];
      int gidx = d_gidx[ridx * row_stride + idx % row_stride];

      if (gidx != null_gidx_value) {
        AtomicAddGpair(d_node_hist + gidx, d_gpair[ridx]);
      }
    });
  }
  void SubtractionTrick(int nidx_parent, int nidx_histogram,
                        int nidx_subtraction) {
    auto d_node_hist_parent = hist.GetHistPtr(nidx_parent);
    auto d_node_hist_histogram = hist.GetHistPtr(nidx_histogram);
    auto d_node_hist_subtraction = hist.GetHistPtr(nidx_subtraction);

    dh::LaunchN(device_idx, hist.n_bins, [=] __device__(size_t idx) {
      d_node_hist_subtraction[idx] =
          d_node_hist_parent[idx] - d_node_hist_histogram[idx];
    });
  }

  __device__ void CountLeft(int64_t* d_count, int val, int left_nidx) {
    unsigned ballot = __ballot(val == left_nidx);
    if (threadIdx.x % 32 == 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(d_count),    // NOLINT
                static_cast<unsigned long long>(__popc(ballot)));  // NOLINT
    }
  }

  void UpdatePosition(int nidx, int left_nidx, int right_nidx, int fidx,
                      int split_gidx, bool default_dir_left, bool is_dense,
                      int fidx_begin, int fidx_end) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    temp_memory.LazyAllocate(sizeof(int64_t));
    auto d_left_count = temp_memory.Pointer<int64_t>();
    dh::safe_cuda(cudaMemset(d_left_count, 0, sizeof(int64_t)));
    auto segment = ridx_segments[nidx];
    auto d_ridx = ridx.Current();
    auto d_position = position.Current();
    auto d_gidx = gidx;
    auto row_stride = this->row_stride;
    dh::LaunchN<1, 512>(
        device_idx, segment.Size(), [=] __device__(bst_uint idx) {
          idx += segment.begin;
          auto ridx = d_ridx[idx];
          auto row_begin = row_stride * ridx;
          auto row_end = row_begin + row_stride;
          auto gidx = -1;
          if (is_dense) {
            gidx = d_gidx[row_begin + fidx];
          } else {
            gidx = BinarySearchRow(row_begin, row_end, d_gidx, fidx_begin,
                                   fidx_end);
          }

          int position;
          if (gidx >= 0) {
            // Feature is found
            position = gidx <= split_gidx ? left_nidx : right_nidx;
          } else {
            // Feature is missing
            position = default_dir_left ? left_nidx : right_nidx;
          }

          CountLeft(d_left_count, position, left_nidx);
          d_position[idx] = position;
        });

    dh::safe_cuda(cudaMemcpy(tmp_pinned, d_left_count, sizeof(int64_t),
                             cudaMemcpyDeviceToHost));
    auto left_count = *tmp_pinned;

    SortPosition(segment, left_nidx, right_nidx);
    // dh::safe_cuda(cudaStreamSynchronize(stream));
    ridx_segments[left_nidx] =
        Segment(segment.begin, segment.begin + left_count);
    ridx_segments[right_nidx] =
        Segment(segment.begin + left_count, segment.end);
  }

  void SortPosition(const Segment& segment, int left_nidx, int right_nidx) {
    int min_bits = 0;
    int max_bits = static_cast<int>(
        std::ceil(std::log2((std::max)(left_nidx, right_nidx) + 1)));

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, position.Current() + segment.begin,
        position.other() + segment.begin, ridx.Current() + segment.begin,
        ridx.other() + segment.begin, segment.Size(), min_bits, max_bits);

    temp_memory.LazyAllocate(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_memory.d_temp_storage, temp_memory.temp_storage_bytes,
        position.Current() + segment.begin, position.other() + segment.begin,
        ridx.Current() + segment.begin, ridx.other() + segment.begin,
        segment.Size(), min_bits, max_bits);
    dh::safe_cuda(cudaMemcpy(
        position.Current() + segment.begin, position.other() + segment.begin,
        segment.Size() * sizeof(int), cudaMemcpyDeviceToDevice));
    dh::safe_cuda(cudaMemcpy(
        ridx.Current() + segment.begin, ridx.other() + segment.begin,
        segment.Size() * sizeof(bst_uint), cudaMemcpyDeviceToDevice));
  }

  void UpdatePredictionCache(bst_float* out_preds_d) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    if (!prediction_cache_initialised) {
      dh::safe_cuda(cudaMemcpy(
          prediction_cache.Data(), &out_preds_d[row_begin_idx],
          prediction_cache.Size() * sizeof(bst_float), cudaMemcpyDefault));
    }
    prediction_cache_initialised = true;

    CalcWeightTrainParam param_d(param);

    thrust::copy(node_sum_gradients.begin(), node_sum_gradients.end(),
                 node_sum_gradients_d.tbegin());
    auto d_position = position.Current();
    auto d_ridx = ridx.Current();
    auto d_node_sum_gradients = node_sum_gradients_d.Data();
    auto d_prediction_cache = prediction_cache.Data();

    dh::LaunchN(
        device_idx, prediction_cache.Size(), [=] __device__(int local_idx) {
          int pos = d_position[local_idx];
          bst_float weight = CalcWeight(param_d, d_node_sum_gradients[pos]);
          d_prediction_cache[d_ridx[local_idx]] +=
              weight * param_d.learning_rate;
        });

    dh::safe_cuda(cudaMemcpy(
        &out_preds_d[row_begin_idx], prediction_cache.Data(),
        prediction_cache.Size() * sizeof(bst_float), cudaMemcpyDefault));
  }
};

class GPUHistMaker : public TreeUpdater {
 public:
  struct ExpandEntry;

  GPUHistMaker() : initialised_(false), p_last_fmat_(nullptr) {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param_.InitAllowUnknown(args);
    CHECK(param_.n_gpus != 0) << "Must have at least one device";
    n_devices_ = param_.n_gpus;

    dh::CheckComputeCapability();

    if (param_.grow_policy == TrainParam::kLossGuide) {
      qexpand_.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand_.reset(new ExpandQueue(DepthWise));
    }

    monitor_.Init("updater_gpu_hist", param_.debug_verbose);
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    monitor_.Start("Update", device_list_);
    GradStats::CheckInfo(dmat->Info());
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    ValueConstraint::Init(&param_, dmat->Info().num_col_);
    // build tree
    try {
      for (size_t i = 0; i < trees.size(); ++i) {
        this->UpdateTree(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param_.learning_rate = lr;
    monitor_.Stop("Update", device_list_);
  }

  void InitDataOnce(DMatrix* dmat) {
    info_ = &dmat->Info();
    monitor_.Start("Quantiles", device_list_);
    hmat_.Init(dmat, param_.max_bin);
    gmat_.cut = &hmat_;
    gmat_.Init(dmat);
    monitor_.Stop("Quantiles", device_list_);
    n_bins_ = hmat_.row_ptr.back();

    int n_devices = dh::NDevices(param_.n_gpus, info_->num_row_);

    bst_uint row_begin = 0;
    bst_uint shard_size =
        std::ceil(static_cast<double>(info_->num_row_) / n_devices);

    device_list_.resize(n_devices);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      int device_idx = (param_.gpu_id + d_idx) % dh::NVisibleDevices();
      device_list_[d_idx] = device_idx;
    }

    reducer_.Init(device_list_);

    // Partition input matrix into row segments
    std::vector<size_t> row_segments;
    shards_.resize(n_devices);
    row_segments.push_back(0);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      bst_uint row_end =
          std::min(static_cast<size_t>(row_begin + shard_size), info_->num_row_);
      row_segments.push_back(row_end);
      row_begin = row_end;
    }

    // Create device shards
    omp_set_num_threads(shards_.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      shards_[cpu_thread_id] = std::unique_ptr<DeviceShard>(
          new DeviceShard(device_list_[cpu_thread_id], cpu_thread_id, gmat_,
                          row_segments[cpu_thread_id],
                          row_segments[cpu_thread_id + 1], n_bins_, param_));
    }

    p_last_fmat_ = dmat;
    initialised_ = true;
  }

  void InitData(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
                const RegTree& tree) {
    monitor_.Start("InitDataOnce", device_list_);
    if (!initialised_) {
      this->InitDataOnce(dmat);
    }
    monitor_.Stop("InitDataOnce", device_list_);

    column_sampler_.Init(info_->num_col_, param_);

    // Copy gpair & reset memory
    monitor_.Start("InitDataReset", device_list_);
    omp_set_num_threads(shards_.size());

    // TODO(canonizer): make it parallel again once HostDeviceVector is
    // thread-safe
    for (int shard = 0; shard < shards_.size(); ++shard)
      shards_[shard]->Reset(gpair, param_.gpu_id);
    monitor_.Stop("InitDataReset", device_list_);
  }

  void AllReduceHist(int nidx) {
    for (auto& shard : shards_) {
      auto d_node_hist = shard->hist.GetHistPtr(nidx);
      reducer_.AllReduceSum(
          shard->normalised_device_idx,
          reinterpret_cast<GradientPairSumT::ValueT*>(d_node_hist),
          reinterpret_cast<GradientPairSumT::ValueT*>(d_node_hist),
          n_bins_ * (sizeof(GradientPairSumT) / sizeof(GradientPairSumT::ValueT)));
    }

    reducer_.Synchronize();
  }

  void BuildHistLeftRight(int nidx_parent, int nidx_left, int nidx_right) {
    size_t left_node_max_elements = 0;
    size_t right_node_max_elements = 0;
    for (auto& shard : shards_) {
      left_node_max_elements = (std::max)(
          left_node_max_elements, shard->ridx_segments[nidx_left].Size());
      right_node_max_elements = (std::max)(
          right_node_max_elements, shard->ridx_segments[nidx_right].Size());
    }

    auto build_hist_nidx = nidx_left;
    auto subtraction_trick_nidx = nidx_right;

    if (right_node_max_elements < left_node_max_elements) {
      build_hist_nidx = nidx_right;
      subtraction_trick_nidx = nidx_left;
    }

    for (auto& shard : shards_) {
      shard->BuildHist(build_hist_nidx);
    }

    this->AllReduceHist(build_hist_nidx);

    for (auto& shard : shards_) {
      shard->SubtractionTrick(nidx_parent, build_hist_nidx,
                              subtraction_trick_nidx);
    }
  }

  // Returns best loss
  std::vector<DeviceSplitCandidate> EvaluateSplits(
      const std::vector<int>& nidx_set, RegTree* p_tree) {
    auto columns = info_->num_col_;
    std::vector<DeviceSplitCandidate> best_splits(nidx_set.size());
    std::vector<DeviceSplitCandidate> candidate_splits(nidx_set.size() *
                                                       columns);
    // Use first device
    auto& shard = shards_.front();
    dh::safe_cuda(cudaSetDevice(shard->device_idx));
    shard->temp_memory.LazyAllocate(sizeof(DeviceSplitCandidate) * columns *
                                    nidx_set.size());
    auto d_split = shard->temp_memory.Pointer<DeviceSplitCandidate>();

    auto& streams = shard->GetStreams(static_cast<int>(nidx_set.size()));

    // Use streams to process nodes concurrently
    for (auto i = 0; i < nidx_set.size(); i++) {
      auto nidx = nidx_set[i];
      DeviceNodeStats node(shard->node_sum_gradients[nidx], nidx, param_);

      const int BLOCK_THREADS = 256;
      evaluate_split_kernel<BLOCK_THREADS>
          <<<uint32_t(columns), BLOCK_THREADS, 0, streams[i]>>>(
              shard->hist.GetHistPtr(nidx), nidx, info_->num_col_, node,
              shard->feature_segments.Data(), shard->min_fvalue.Data(),
              shard->gidx_fvalue_map.Data(), GPUTrainingParam(param_),
              d_split + i * columns, node_value_constraints_[nidx],
              shard->monotone_constraints.Data());
    }

    dh::safe_cuda(
        cudaMemcpy(candidate_splits.data(), shard->temp_memory.d_temp_storage,
                   sizeof(DeviceSplitCandidate) * columns * nidx_set.size(),
                   cudaMemcpyDeviceToHost));

    for (auto i = 0; i < nidx_set.size(); i++) {
      auto nidx = nidx_set[i];
      DeviceSplitCandidate nidx_best;
      for (auto fidx = 0; fidx < columns; fidx++) {
        auto& candidate = candidate_splits[i * columns + fidx];
        if (column_sampler_.ColumnUsed(candidate.findex,
                                      p_tree->GetDepth(nidx))) {
          nidx_best.Update(candidate_splits[i * columns + fidx], param_);
        }
      }
      best_splits[i] = nidx_best;
    }
    return std::move(best_splits);
  }

  void InitRoot(RegTree* p_tree) {
    auto root_nidx = 0;
    // Sum gradients
    std::vector<GradientPair> tmp_sums(shards_.size());
    omp_set_num_threads(shards_.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      auto& shard = shards_[cpu_thread_id];
      dh::safe_cuda(cudaSetDevice(shard->device_idx));
      tmp_sums[cpu_thread_id] = dh::SumReduction(
          shard->temp_memory, shard->gpair.Data(), shard->gpair.Size());
    }
    auto sum_gradient =
        std::accumulate(tmp_sums.begin(), tmp_sums.end(), GradientPair());

    // Generate root histogram
    for (auto& shard : shards_) {
      shard->BuildHist(root_nidx);
    }

    this->AllReduceHist(root_nidx);

    // Remember root stats
    p_tree->Stat(root_nidx).sum_hess = sum_gradient.GetHess();
    auto weight = CalcWeight(param_, sum_gradient);
    p_tree->Stat(root_nidx).base_weight = weight;
    (*p_tree)[root_nidx].SetLeaf(param_.learning_rate * weight);

    // Store sum gradients
    for (auto& shard : shards_) {
      shard->node_sum_gradients[root_nidx] = sum_gradient;
    }

    // Initialise root constraint
    node_value_constraints_.resize(p_tree->GetNodes().size());

    // Generate first split
    auto splits = this->EvaluateSplits({root_nidx}, p_tree);
    qexpand_->push(
        ExpandEntry(root_nidx, p_tree->GetDepth(root_nidx), splits.front(), 0));
  }

  void UpdatePosition(const ExpandEntry& candidate, RegTree* p_tree) {
    auto nidx = candidate.nid;
    auto left_nidx = (*p_tree)[nidx].LeftChild();
    auto right_nidx = (*p_tree)[nidx].RightChild();

    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    auto split_gidx = -1;
    auto fidx = candidate.split.findex;
    auto default_dir_left = candidate.split.dir == kLeftDir;
    auto fidx_begin = hmat_.row_ptr[fidx];
    auto fidx_end = hmat_.row_ptr[fidx + 1];
    for (auto i = fidx_begin; i < fidx_end; ++i) {
      if (candidate.split.fvalue == hmat_.cut[i]) {
        split_gidx = static_cast<int32_t>(i);
      }
    }

    auto is_dense = info_->num_nonzero_ == info_->num_row_ * info_->num_col_;

    omp_set_num_threads(shards_.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      shards_[cpu_thread_id]->UpdatePosition(nidx, left_nidx, right_nidx, fidx,
                                            split_gidx, default_dir_left,
                                            is_dense, fidx_begin, fidx_end);
    }
  }

  void ApplySplit(const ExpandEntry& candidate, RegTree* p_tree) {
    // Add new leaves
    RegTree& tree = *p_tree;
    tree.AddChilds(candidate.nid);
    auto& parent = tree[candidate.nid];
    parent.SetSplit(candidate.split.findex, candidate.split.fvalue,
                     candidate.split.dir == kLeftDir);
    tree.Stat(candidate.nid).loss_chg = candidate.split.loss_chg;

    // Set up child constraints
    node_value_constraints_.resize(tree.GetNodes().size());
    GradStats left_stats(param_);
    left_stats.Add(candidate.split.left_sum);
    GradStats right_stats(param_);
    right_stats.Add(candidate.split.right_sum);
    node_value_constraints_[candidate.nid].SetChild(
        param_, parent.SplitIndex(), left_stats, right_stats,
        &node_value_constraints_[parent.LeftChild()],
        &node_value_constraints_[parent.RightChild()]);

    // Configure left child
    auto left_weight =
        node_value_constraints_[parent.LeftChild()].CalcWeight(param_, left_stats);
    tree[parent.LeftChild()].SetLeaf(left_weight * param_.learning_rate, 0);
    tree.Stat(parent.LeftChild()).base_weight = left_weight;
    tree.Stat(parent.LeftChild()).sum_hess = candidate.split.left_sum.GetHess();

    // Configure right child
    auto right_weight =
        node_value_constraints_[parent.RightChild()].CalcWeight(param_, right_stats);
    tree[parent.RightChild()].SetLeaf(right_weight * param_.learning_rate, 0);
    tree.Stat(parent.RightChild()).base_weight = right_weight;
    tree.Stat(parent.RightChild()).sum_hess = candidate.split.right_sum.GetHess();
    // Store sum gradients
    for (auto& shard : shards_) {
      shard->node_sum_gradients[parent.LeftChild()] = candidate.split.left_sum;
      shard->node_sum_gradients[parent.RightChild()] = candidate.split.right_sum;
    }
    this->UpdatePosition(candidate, p_tree);
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat,
                  RegTree* p_tree) {
    // Temporarily store number of threads so we can change it back later
    int nthread = omp_get_max_threads();

    auto& tree = *p_tree;

    monitor_.Start("InitData", device_list_);
    this->InitData(gpair, p_fmat, *p_tree);
    monitor_.Stop("InitData", device_list_);
    monitor_.Start("InitRoot", device_list_);
    this->InitRoot(p_tree);
    monitor_.Stop("InitRoot", device_list_);

    auto timestamp = qexpand_->size();
    auto num_leaves = 1;

    while (!qexpand_->empty()) {
      auto candidate = qexpand_->top();
      qexpand_->pop();
      if (!candidate.IsValid(param_, num_leaves)) continue;
      // std::cout << candidate;
      monitor_.Start("ApplySplit", device_list_);
      this->ApplySplit(candidate, p_tree);
      monitor_.Stop("ApplySplit", device_list_);
      num_leaves++;

      auto left_child_nidx = tree[candidate.nid].LeftChild();
      auto right_child_nidx = tree[candidate.nid].RightChild();

      // Only create child entries if needed
      if (ExpandEntry::ChildIsValid(param_, tree.GetDepth(left_child_nidx),
                                    num_leaves)) {
        monitor_.Start("BuildHist", device_list_);
        this->BuildHistLeftRight(candidate.nid, left_child_nidx,
                                 right_child_nidx);
        monitor_.Stop("BuildHist", device_list_);

        monitor_.Start("EvaluateSplits", device_list_);
        auto splits =
            this->EvaluateSplits({left_child_nidx, right_child_nidx}, p_tree);
        qexpand_->push(ExpandEntry(left_child_nidx,
                                   tree.GetDepth(left_child_nidx), splits[0],
                                   timestamp++));
        qexpand_->push(ExpandEntry(right_child_nidx,
                                   tree.GetDepth(right_child_nidx), splits[1],
                                   timestamp++));
        monitor_.Stop("EvaluateSplits", device_list_);
      }
    }
    // Reset omp num threads
    omp_set_num_threads(nthread);
  }

  bool UpdatePredictionCache(
      const DMatrix* data, HostDeviceVector<bst_float>* p_out_preds) override {
    monitor_.Start("UpdatePredictionCache", device_list_);
    if (shards_.empty() || p_last_fmat_ == nullptr || p_last_fmat_ != data)
      return false;

    bst_float* out_preds_d = p_out_preds->DevicePointer(param_.gpu_id);

#pragma omp parallel for schedule(static, 1)
    for (int shard = 0; shard < shards_.size(); ++shard) {
      shards_[shard]->UpdatePredictionCache(out_preds_d);
    }
    monitor_.Stop("UpdatePredictionCache", device_list_);
    return true;
  }

  struct ExpandEntry {
    int nid;
    int depth;
    DeviceSplitCandidate split;
    uint64_t timestamp;
    ExpandEntry(int nid, int depth, const DeviceSplitCandidate& split,
                uint64_t timestamp)
        : nid(nid), depth(depth), split(split), timestamp(timestamp) {}
    bool IsValid(const TrainParam& param, int num_leaves) const {
      if (split.loss_chg <= kRtEps) return false;
      if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0)
        return false;
      if (param.max_depth > 0 && depth == param.max_depth) return false;
      if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;
      return true;
    }

    static bool ChildIsValid(const TrainParam& param, int depth,
                             int num_leaves) {
      if (param.max_depth > 0 && depth == param.max_depth) return false;
      if (param.max_leaves > 0 && num_leaves == param.max_leaves) return false;
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

  inline static bool DepthWise(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.depth == rhs.depth) {
      return lhs.timestamp > rhs.timestamp;  // favor small timestamp
    } else {
      return lhs.depth > rhs.depth;  // favor small depth
    }
  }
  inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.split.loss_chg == rhs.split.loss_chg) {
      return lhs.timestamp > rhs.timestamp;  // favor small timestamp
    } else {
      return lhs.split.loss_chg < rhs.split.loss_chg;  // favor large loss_chg
    }
  }
  TrainParam param_;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo* info_;
  bool initialised_;
  int n_devices_;
  int n_bins_;

  std::vector<std::unique_ptr<DeviceShard>> shards_;
  ColumnSampler column_sampler_;
  typedef std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                              std::function<bool(ExpandEntry, ExpandEntry)>>
      ExpandQueue;
  std::unique_ptr<ExpandQueue> qexpand_;
  common::Monitor monitor_;
  dh::AllReducer reducer_;
  std::vector<ValueConstraint> node_value_constraints_;
  std::vector<int> device_list_;

  DMatrix* p_last_fmat_;
};

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
}  // namespace tree
}  // namespace xgboost
