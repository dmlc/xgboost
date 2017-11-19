/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
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
#include "../common/timer.h"
#include "param.h"
#include "updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu_hist_experimental);

typedef bst_gpair_precise gpair_sum_t;

template <int BLOCK_THREADS, typename reduce_t, typename temp_storage_t>
__device__ gpair_sum_t ReduceFeature(const gpair_sum_t* begin,
                                     const gpair_sum_t* end,
                                     temp_storage_t* temp_storage) {
  __shared__ cub::Uninitialized<gpair_sum_t> uninitialized_sum;
  gpair_sum_t& shared_sum = uninitialized_sum.Alias();

  gpair_sum_t local_sum = gpair_sum_t();
  for (auto itr = begin; itr < end; itr += BLOCK_THREADS) {
    bool thread_active = itr + threadIdx.x < end;
    // Scan histogram
    gpair_sum_t bin = thread_active ? *(itr + threadIdx.x) : gpair_sum_t();

    local_sum += reduce_t(temp_storage->sum_reduce).Reduce(bin, cub::Sum());
  }

  if (threadIdx.x == 0) {
    shared_sum = local_sum;
  }
  __syncthreads();

  return shared_sum;
}

template <int BLOCK_THREADS, typename reduce_t, typename scan_t,
          typename max_reduce_t, typename temp_storage_t>
__device__ void EvaluateFeature(int fidx, const gpair_sum_t* hist,
                                const int* feature_segments, float min_fvalue,
                                const float* gidx_fvalue_map,
                                DeviceSplitCandidate* best_split,
                                const DeviceNodeStats& node,
                                const GPUTrainingParam& param,
                                temp_storage_t* temp_storage) {
  int gidx_begin = feature_segments[fidx];
  int gidx_end = feature_segments[fidx + 1];

  gpair_sum_t feature_sum = ReduceFeature<BLOCK_THREADS, reduce_t>(
      hist + gidx_begin, hist + gidx_end, temp_storage);

  auto prefix_op = SumCallbackOp<gpair_sum_t>();
  for (int scan_begin = gidx_begin; scan_begin < gidx_end;
       scan_begin += BLOCK_THREADS) {
    bool thread_active = scan_begin + threadIdx.x < gidx_end;

    gpair_sum_t bin =
        thread_active ? hist[scan_begin + threadIdx.x] : gpair_sum_t();
    scan_t(temp_storage->scan).ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

    // Calculate  gain
    gpair_sum_t parent_sum = gpair_sum_t(node.sum_gradients);

    gpair_sum_t missing = parent_sum - feature_sum;

    bool missing_left = true;
    const float null_gain = -FLT_MAX;
    float gain = null_gain;
    if (thread_active) {
      gain = loss_chg_missing(bin, missing, parent_sum, node.root_gain, param,
                              missing_left);
    }

    __syncthreads();

    // Find thread with best gain
    cub::KeyValuePair<int, float> tuple(threadIdx.x, gain);
    cub::KeyValuePair<int, float> best =
        max_reduce_t(temp_storage->max_reduce).Reduce(tuple, cub::ArgMax());

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

      gpair_sum_t left = missing_left ? bin + missing : bin;
      gpair_sum_t right = parent_sum - left;

      best_split->Update(gain, missing_left ? LeftDir : RightDir, fvalue, fidx,
                         left, right, param);
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS>
__global__ void evaluate_split_kernel(
    const gpair_sum_t* d_hist, int nidx, uint64_t n_features,
    DeviceNodeStats nodes, const int* d_feature_segments,
    const float* d_fidx_min_map, const float* d_gidx_fvalue_map,
    GPUTrainingParam gpu_param, DeviceSplitCandidate* d_split) {
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<gpair_sum_t, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;

  typedef cub::BlockReduce<gpair_sum_t, BLOCK_THREADS> SumReduceT;

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
  EvaluateFeature<BLOCK_THREADS, SumReduceT, BlockScanT, MaxReduceT>(
      fidx, d_hist, d_feature_segments, d_fidx_min_map[fidx], d_gidx_fvalue_map,
      &best_split, nodes, gpu_param, &temp_storage);

  __syncthreads();

  if (threadIdx.x == 0) {
    // Record best loss
    d_split[fidx] = best_split;
  }
}

// Find a gidx value for a given feature otherwise return -1 if not found
template <typename gidx_iter_t>
__device__ int BinarySearchRow(bst_uint begin, bst_uint end, gidx_iter_t data,
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
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;
  dh::dvec<gpair_sum_t> data;
  int n_bins;
  void Init(int device_idx, int max_nodes, int n_bins, bool silent) {
    this->n_bins = n_bins;
    ba.allocate(device_idx, silent, &data, size_t(max_nodes) * size_t(n_bins));
  }

  void Reset() { data.fill(gpair_sum_t()); }
  gpair_sum_t* GetHistPtr(int nidx) { return data.data() + nidx * n_bins; }

  void PrintNidx(int nidx) const {
    auto h_data = data.as_vector();
    std::cout << "nidx " << nidx << ":\n";
    for (int i = n_bins * nidx; i < n_bins * (nidx + 1); i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << "\n";
  }
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
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;
  dh::dvec<common::compressed_byte_t> gidx_buffer;
  dh::dvec<bst_gpair> gpair;
  dh::dvec2<bst_uint> ridx;  // Row index relative to this shard
  dh::dvec2<int> position;
  std::vector<Segment> ridx_segments;
  dh::dvec<int> feature_segments;
  dh::dvec<float> gidx_fvalue_map;
  dh::dvec<float> min_fvalue;
  std::vector<bst_gpair> node_sum_gradients;
  common::CompressedIterator<uint32_t> gidx;
  int row_stride;
  bst_uint row_begin_idx;  // The row offset for this shard
  bst_uint row_end_idx;
  bst_uint n_rows;
  int n_bins;
  int null_gidx_value;
  DeviceHistogram hist;
  TrainParam param;

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
        param(param) {
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
    int max_nodes =
        param.max_leaves > 0 ? param.max_leaves * 2 : n_nodes(param.max_depth);
    ba.allocate(device_idx, param.silent, &gidx_buffer, compressed_size_bytes,
                &gpair, n_rows, &ridx, n_rows, &position, n_rows,
                &feature_segments, gmat.cut->row_ptr.size(), &gidx_fvalue_map,
                gmat.cut->cut.size(), &min_fvalue, gmat.cut->min_val.size());
    gidx_fvalue_map = gmat.cut->cut;
    min_fvalue = gmat.cut->min_val;
    feature_segments = gmat.cut->row_ptr;

    node_sum_gradients.resize(max_nodes);
    ridx_segments.resize(max_nodes);

    // Compress gidx
    common::CompressedBufferWriter cbw(num_symbols);
    std::vector<common::compressed_byte_t> host_buffer(gidx_buffer.size());
    cbw.Write(host_buffer.data(), ellpack_matrix.begin(), ellpack_matrix.end());
    gidx_buffer = host_buffer;
    gidx =
        common::CompressedIterator<uint32_t>(gidx_buffer.data(), num_symbols);

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
  void Reset(const std::vector<bst_gpair>& host_gpair) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    position.current_dvec().fill(0);
    std::fill(node_sum_gradients.begin(), node_sum_gradients.end(),
              bst_gpair());

    thrust::sequence(ridx.current_dvec().tbegin(), ridx.current_dvec().tend());

    std::fill(ridx_segments.begin(), ridx_segments.end(), Segment(0, 0));
    ridx_segments.front() = Segment(0, ridx.size());
    this->gpair.copy(host_gpair.begin() + row_begin_idx,
                     host_gpair.begin() + row_end_idx);
    subsample_gpair(&gpair, param.subsample, row_begin_idx);
    hist.Reset();
  }

  void BuildHist(int nidx) {
    auto segment = ridx_segments[nidx];
    auto d_node_hist = hist.GetHistPtr(nidx);
    auto d_gidx = gidx;
    auto d_ridx = ridx.current();
    auto d_gpair = gpair.data();
    auto row_stride = this->row_stride;
    auto null_gidx_value = this->null_gidx_value;
    auto n_elements = segment.Size() * row_stride;

    dh::launch_n(device_idx, n_elements, [=] __device__(size_t idx) {
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

    dh::launch_n(device_idx, hist.n_bins, [=] __device__(size_t idx) {
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
    auto d_ridx = ridx.current();
    auto d_position = position.current();
    auto d_gidx = gidx;
    auto row_stride = this->row_stride;
    dh::launch_n<1, 512>(
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
        nullptr, temp_storage_bytes, position.current() + segment.begin,
        position.other() + segment.begin, ridx.current() + segment.begin,
        ridx.other() + segment.begin, segment.Size(), min_bits, max_bits);

    temp_memory.LazyAllocate(temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        temp_memory.d_temp_storage, temp_memory.temp_storage_bytes,
        position.current() + segment.begin, position.other() + segment.begin,
        ridx.current() + segment.begin, ridx.other() + segment.begin,
        segment.Size(), min_bits, max_bits);
    dh::safe_cuda(cudaMemcpy(
        position.current() + segment.begin, position.other() + segment.begin,
        segment.Size() * sizeof(int), cudaMemcpyDeviceToDevice));
    dh::safe_cuda(cudaMemcpy(
        ridx.current() + segment.begin, ridx.other() + segment.begin,
        segment.Size() * sizeof(bst_uint), cudaMemcpyDeviceToDevice));
  }
};

class GPUHistMakerExperimental : public TreeUpdater {
 public:
  struct ExpandEntry;

  GPUHistMakerExperimental() : initialised(false) {}
  ~GPUHistMakerExperimental() {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    CHECK(param.n_gpus != 0) << "Must have at least one device";
    n_devices = param.n_gpus;

    dh::check_compute_capability();

    if (param.grow_policy == TrainParam::kLossGuide) {
      qexpand_.reset(new ExpandQueue(loss_guide));
    } else {
      qexpand_.reset(new ExpandQueue(depth_wise));
    }

    monitor.Init("updater_gpu_hist_experimental", param.debug_verbose);
  }
  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    monitor.Start("Update");
    GradStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    // build tree
    try {
      for (size_t i = 0; i < trees.size(); ++i) {
        this->UpdateTree(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param.learning_rate = lr;
    monitor.Stop("Update");
  }

  void InitDataOnce(DMatrix* dmat) {
    info = &dmat->info();
    monitor.Start("Quantiles");
    hmat_.Init(dmat, param.max_bin);
    gmat_.cut = &hmat_;
    gmat_.Init(dmat);
    monitor.Stop("Quantiles");
    n_bins = hmat_.row_ptr.back();

    int n_devices = dh::n_devices(param.n_gpus, info->num_row);

    bst_uint row_begin = 0;
    bst_uint shard_size =
        std::ceil(static_cast<double>(info->num_row) / n_devices);

    std::vector<int> dList(n_devices);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      int device_idx = (param.gpu_id + d_idx) % dh::n_visible_devices();
      dList[d_idx] = device_idx;
    }

    reducer.Init(dList);

    // Partition input matrix into row segments
    std::vector<size_t> row_segments;
    shards.resize(n_devices);
    row_segments.push_back(0);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      bst_uint row_end =
          std::min(static_cast<size_t>(row_begin + shard_size), info->num_row);
      row_segments.push_back(row_end);
      row_begin = row_end;
    }

    // Create device shards
    omp_set_num_threads(shards.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      shards[cpu_thread_id] = std::unique_ptr<DeviceShard>(
          new DeviceShard(dList[cpu_thread_id], cpu_thread_id, gmat_,
                          row_segments[cpu_thread_id],
                          row_segments[cpu_thread_id + 1], n_bins, param));
    }

    initialised = true;
  }

  void InitData(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
                const RegTree& tree) {
    monitor.Start("InitDataOnce");
    if (!initialised) {
      CheckGradientMax(gpair);
      this->InitDataOnce(dmat);
    }
    monitor.Stop("InitDataOnce");

    column_sampler.Init(info->num_col, param);

    // Copy gpair & reset memory
    monitor.Start("InitDataReset");
    omp_set_num_threads(shards.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      shards[cpu_thread_id]->Reset(gpair);
    }
    monitor.Stop("InitDataReset");
  }

  void AllReduceHist(int nidx) {
    for (auto& shard : shards) {
      auto d_node_hist = shard->hist.GetHistPtr(nidx);
      reducer.AllReduceSum(
          shard->normalised_device_idx,
          reinterpret_cast<gpair_sum_t::value_t*>(d_node_hist),
          reinterpret_cast<gpair_sum_t::value_t*>(d_node_hist),
          n_bins * (sizeof(gpair_sum_t) / sizeof(gpair_sum_t::value_t)));
    }

    reducer.Synchronize();
  }

  void BuildHistLeftRight(int nidx_parent, int nidx_left, int nidx_right) {
    size_t left_node_max_elements = 0;
    size_t right_node_max_elements = 0;
    for (auto& shard : shards) {
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

    for (auto& shard : shards) {
      shard->BuildHist(build_hist_nidx);
    }

    this->AllReduceHist(build_hist_nidx);

    for (auto& shard : shards) {
      shard->SubtractionTrick(nidx_parent, build_hist_nidx,
                              subtraction_trick_nidx);
    }
  }

  // Returns best loss
  std::vector<DeviceSplitCandidate> EvaluateSplits(
      const std::vector<int>& nidx_set, RegTree* p_tree) {
    auto columns = info->num_col;
    std::vector<DeviceSplitCandidate> best_splits(nidx_set.size());
    std::vector<DeviceSplitCandidate> candidate_splits(nidx_set.size() *
                                                       columns);
    // Use first device
    auto& shard = shards.front();
    dh::safe_cuda(cudaSetDevice(shard->device_idx));
    shard->temp_memory.LazyAllocate(sizeof(DeviceSplitCandidate) * columns *
                                    nidx_set.size());
    auto d_split = shard->temp_memory.Pointer<DeviceSplitCandidate>();

    auto& streams = shard->GetStreams(static_cast<int>(nidx_set.size()));

    // Use streams to process nodes concurrently
    for (auto i = 0; i < nidx_set.size(); i++) {
      auto nidx = nidx_set[i];
      DeviceNodeStats node(shard->node_sum_gradients[nidx], nidx, param);

      const int BLOCK_THREADS = 256;
      evaluate_split_kernel<BLOCK_THREADS>
          <<<uint32_t(columns), BLOCK_THREADS, 0, streams[i]>>>(
              shard->hist.GetHistPtr(nidx), nidx, info->num_col, node,
              shard->feature_segments.data(), shard->min_fvalue.data(),
              shard->gidx_fvalue_map.data(), GPUTrainingParam(param),
              d_split + i * columns);
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
        if (column_sampler.ColumnUsed(candidate.findex,
                                      p_tree->GetDepth(nidx))) {
          nidx_best.Update(candidate_splits[i * columns + fidx], param);
        }
      }
      best_splits[i] = nidx_best;
    }
    return std::move(best_splits);
  }

  void InitRoot(const std::vector<bst_gpair>& gpair, RegTree* p_tree) {
    auto root_nidx = 0;
    // Sum gradients
    std::vector<bst_gpair> tmp_sums(shards.size());
    omp_set_num_threads(shards.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      dh::safe_cuda(cudaSetDevice(shards[cpu_thread_id]->device_idx));
      tmp_sums[cpu_thread_id] =
          thrust::reduce(thrust::cuda::par(shards[cpu_thread_id]->temp_memory),
                         shards[cpu_thread_id]->gpair.tbegin(),
                         shards[cpu_thread_id]->gpair.tend());
    }
    auto sum_gradient =
        std::accumulate(tmp_sums.begin(), tmp_sums.end(), bst_gpair());

    // Generate root histogram
    for (auto& shard : shards) {
      shard->BuildHist(root_nidx);
    }

    this->AllReduceHist(root_nidx);

    // Remember root stats
    p_tree->stat(root_nidx).sum_hess = sum_gradient.GetHess();
    p_tree->stat(root_nidx).base_weight = CalcWeight(param, sum_gradient);

    // Store sum gradients
    for (auto& shard : shards) {
      shard->node_sum_gradients[root_nidx] = sum_gradient;
    }

    // Generate first split
    auto splits = this->EvaluateSplits({root_nidx}, p_tree);
    qexpand_->push(
        ExpandEntry(root_nidx, p_tree->GetDepth(root_nidx), splits.front(), 0));
  }

  void UpdatePosition(const ExpandEntry& candidate, RegTree* p_tree) {
    auto nidx = candidate.nid;
    auto left_nidx = (*p_tree)[nidx].cleft();
    auto right_nidx = (*p_tree)[nidx].cright();

    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    auto split_gidx = -1;
    auto fidx = candidate.split.findex;
    auto default_dir_left = candidate.split.dir == LeftDir;
    auto fidx_begin = hmat_.row_ptr[fidx];
    auto fidx_end = hmat_.row_ptr[fidx + 1];
    for (auto i = fidx_begin; i < fidx_end; ++i) {
      if (candidate.split.fvalue == hmat_.cut[i]) {
        split_gidx = static_cast<int32_t>(i);
      }
    }

    auto is_dense = info->num_nonzero == info->num_row * info->num_col;

    omp_set_num_threads(shards.size());
#pragma omp parallel
    {
      auto cpu_thread_id = omp_get_thread_num();
      shards[cpu_thread_id]->UpdatePosition(nidx, left_nidx, right_nidx, fidx,
                                            split_gidx, default_dir_left,
                                            is_dense, fidx_begin, fidx_end);
    }
  }

  void ApplySplit(const ExpandEntry& candidate, RegTree* p_tree) {
    // Add new leaves
    RegTree& tree = *p_tree;
    tree.AddChilds(candidate.nid);
    auto& parent = tree[candidate.nid];
    parent.set_split(candidate.split.findex, candidate.split.fvalue,
                     candidate.split.dir == LeftDir);
    tree.stat(candidate.nid).loss_chg = candidate.split.loss_chg;

    // Configure left child
    auto left_weight = CalcWeight(param, candidate.split.left_sum);
    tree[parent.cleft()].set_leaf(left_weight * param.learning_rate, 0);
    tree.stat(parent.cleft()).base_weight = left_weight;
    tree.stat(parent.cleft()).sum_hess = candidate.split.left_sum.GetHess();

    // Configure right child
    auto right_weight = CalcWeight(param, candidate.split.right_sum);
    tree[parent.cright()].set_leaf(right_weight * param.learning_rate, 0);
    tree.stat(parent.cright()).base_weight = right_weight;
    tree.stat(parent.cright()).sum_hess = candidate.split.right_sum.GetHess();
    // Store sum gradients
    for (auto& shard : shards) {
      shard->node_sum_gradients[parent.cleft()] = candidate.split.left_sum;
      shard->node_sum_gradients[parent.cright()] = candidate.split.right_sum;
    }
    this->UpdatePosition(candidate, p_tree);
  }

  void UpdateTree(const std::vector<bst_gpair>& gpair, DMatrix* p_fmat,
                  RegTree* p_tree) {
    auto& tree = *p_tree;

    monitor.Start("InitData");
    this->InitData(gpair, p_fmat, *p_tree);
    monitor.Stop("InitData");
    monitor.Start("InitRoot");
    this->InitRoot(gpair, p_tree);
    monitor.Stop("InitRoot");

    auto timestamp = qexpand_->size();
    auto num_leaves = 1;

    while (!qexpand_->empty()) {
      auto candidate = qexpand_->top();
      qexpand_->pop();
      if (!candidate.IsValid(param, num_leaves)) continue;
      // std::cout << candidate;
      monitor.Start("ApplySplit");
      this->ApplySplit(candidate, p_tree);
      monitor.Stop("ApplySplit");
      num_leaves++;

      auto left_child_nidx = tree[candidate.nid].cleft();
      auto right_child_nidx = tree[candidate.nid].cright();

      // Only create child entries if needed
      if (ExpandEntry::ChildIsValid(param, tree.GetDepth(left_child_nidx),
                                    num_leaves)) {
        monitor.Start("BuildHist");
        this->BuildHistLeftRight(candidate.nid, left_child_nidx,
                                 right_child_nidx);
        monitor.Stop("BuildHist");

        monitor.Start("EvaluateSplits");
        auto splits =
            this->EvaluateSplits({left_child_nidx, right_child_nidx}, p_tree);
        qexpand_->push(ExpandEntry(left_child_nidx,
                                   tree.GetDepth(left_child_nidx), splits[0],
                                   timestamp++));
        qexpand_->push(ExpandEntry(right_child_nidx,
                                   tree.GetDepth(right_child_nidx), splits[1],
                                   timestamp++));
        monitor.Stop("EvaluateSplits");
      }
    }
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
      if (split.loss_chg <= rt_eps) return false;
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

  inline static bool depth_wise(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.depth == rhs.depth) {
      return lhs.timestamp > rhs.timestamp;  // favor small timestamp
    } else {
      return lhs.depth > rhs.depth;  // favor small depth
    }
  }
  inline static bool loss_guide(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.split.loss_chg == rhs.split.loss_chg) {
      return lhs.timestamp > rhs.timestamp;  // favor small timestamp
    } else {
      return lhs.split.loss_chg < rhs.split.loss_chg;  // favor large loss_chg
    }
  }
  TrainParam param;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo* info;
  bool initialised;
  int n_devices;
  int n_bins;

  std::vector<std::unique_ptr<DeviceShard>> shards;
  ColumnSampler column_sampler;
  typedef std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                              std::function<bool(ExpandEntry, ExpandEntry)>>
      ExpandQueue;
  std::unique_ptr<ExpandQueue> qexpand_;
  common::Monitor monitor;
  dh::AllReducer reducer;
};

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMakerExperimental,
                              "grow_gpu_hist_experimental")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMakerExperimental(); });
}  // namespace tree
}  // namespace xgboost
