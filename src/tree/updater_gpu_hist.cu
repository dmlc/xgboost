/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include <memory>
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

DMLC_REGISTRY_FILE_TAG(updater_gpu_hist);

typedef bst_gpair_integer gpair_sum_t;

// Helper for explicit template specialisation
template <int N>
struct Int {};

struct DeviceGMat {
  dh::dvec<common::compressed_byte_t> gidx_buffer;
  common::CompressedIterator<uint32_t> gidx;
  dh::dvec<size_t> row_ptr;
  void Init(int device_idx, const common::GHistIndexMatrix& gmat,
            bst_ulong element_begin, bst_ulong element_end, bst_ulong row_begin,
            bst_ulong row_end, int n_bins) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    CHECK(gidx_buffer.size()) << "gidx_buffer must be externally allocated";
    CHECK_EQ(row_ptr.size(), (row_end - row_begin) + 1)
        << "row_ptr must be externally allocated";

    common::CompressedBufferWriter cbw(n_bins);
    std::vector<common::compressed_byte_t> host_buffer(gidx_buffer.size());
    cbw.Write(host_buffer.data(), gmat.index.begin() + element_begin,
              gmat.index.begin() + element_end);
    gidx_buffer = host_buffer;
    gidx = common::CompressedIterator<uint32_t>(gidx_buffer.data(), n_bins);

    // row_ptr
    dh::safe_cuda(cudaMemcpy(row_ptr.data(), gmat.row_ptr.data() + row_begin,
                             row_ptr.size() * sizeof(size_t),
                             cudaMemcpyHostToDevice));
    // normalise row_ptr
    size_t start = gmat.row_ptr[row_begin];
    auto d_row_ptr = row_ptr.data();
    dh::launch_n(row_ptr.device_idx(), row_ptr.size(),
                 [=] __device__(size_t idx) { d_row_ptr[idx] -= start; });
  }
};

struct HistHelper {
  gpair_sum_t* d_hist;
  int n_bins;
  __host__ __device__ HistHelper(gpair_sum_t* ptr, int n_bins)
      : d_hist(ptr), n_bins(n_bins) {}

  __device__ void Add(bst_gpair gpair, int gidx, int nidx) const {
    int hist_idx = nidx * n_bins + gidx;

    AtomicAddGpair(d_hist + hist_idx, gpair);
  }
  __device__ gpair_sum_t Get(int gidx, int nidx) const {
    return d_hist[nidx * n_bins + gidx];
  }
};

struct DeviceHist {
  int n_bins;
  dh::dvec<gpair_sum_t> data;

  void Init(int n_bins_in) {
    this->n_bins = n_bins_in;
    CHECK(!data.empty()) << "DeviceHist must be externally allocated";
  }

  void Reset(int device_idx) {
    cudaSetDevice(device_idx);
    data.fill(gpair_sum_t());
  }

  HistHelper GetBuilder() { return HistHelper(data.data(), n_bins); }

  gpair_sum_t* GetLevelPtr(int depth) {
    return data.data() + n_nodes(depth - 1) * n_bins;
  }

  int LevelSize(int depth) { return n_bins * n_nodes_level(depth); }
};

template <int BLOCK_THREADS>
__global__ void find_split_kernel(
    const gpair_sum_t* d_level_hist, int* d_feature_segments, int depth,
    uint64_t n_features, int n_bins, DeviceNodeStats* d_nodes,
    int nodes_offset_device, float* d_fidx_min_map, float* d_gidx_fvalue_map,
    GPUTrainingParam gpu_param, bool* d_left_child_smallest_temp,
    bool colsample, int* d_feature_flags) {
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
  DeviceSplitCandidate& split = uninitialized_split.Alias();
  __shared__ cub::Uninitialized<gpair_sum_t> uninitialized_sum;
  gpair_sum_t& shared_sum = uninitialized_sum.Alias();
  __shared__ ArgMaxT block_max;
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    split = DeviceSplitCandidate();
  }

  __syncthreads();

  // below two are for accessing full-sized node list stored on each device
  // always one block per node, BLOCK_THREADS threads per block
  int level_node_idx = blockIdx.x + nodes_offset_device;
  int node_idx = n_nodes(depth - 1) + level_node_idx;

  for (int fidx = 0; fidx < n_features; fidx++) {
    if (colsample && d_feature_flags[fidx] == 0) continue;

    int begin = d_feature_segments[level_node_idx * n_features + fidx];
    int end = d_feature_segments[level_node_idx * n_features + fidx + 1];

    gpair_sum_t feature_sum = gpair_sum_t();
    for (int reduce_begin = begin; reduce_begin < end;
         reduce_begin += BLOCK_THREADS) {
      bool thread_active = reduce_begin + threadIdx.x < end;
      // Scan histogram
      gpair_sum_t bin = thread_active ? d_level_hist[reduce_begin + threadIdx.x]
                                      : gpair_sum_t();

      feature_sum +=
          SumReduceT(temp_storage.sum_reduce).Reduce(bin, cub::Sum());
    }

    if (threadIdx.x == 0) {
      shared_sum = feature_sum;
    }
    //    __syncthreads(); // no need to synch because below there is a Scan

    auto prefix_op = SumCallbackOp<gpair_sum_t>();
    for (int scan_begin = begin; scan_begin < end;
         scan_begin += BLOCK_THREADS) {
      bool thread_active = scan_begin + threadIdx.x < end;
      gpair_sum_t bin = thread_active ? d_level_hist[scan_begin + threadIdx.x]
                                      : gpair_sum_t();

      BlockScanT(temp_storage.scan)
          .ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

      // Calculate gain
      gpair_sum_t parent_sum = gpair_sum_t(d_nodes[node_idx].sum_gradients);
      float parent_gain = d_nodes[node_idx].root_gain;

      gpair_sum_t missing = parent_sum - shared_sum;

      bool missing_left;
      float gain = thread_active
                       ? loss_chg_missing(bin, missing, parent_sum, parent_gain,
                                          gpu_param, missing_left)
                       : -FLT_MAX;
      __syncthreads();

      // Find thread with best gain
      ArgMaxT tuple(threadIdx.x, gain);
      ArgMaxT best =
          MaxReduceT(temp_storage.max_reduce).Reduce(tuple, cub::ArgMax());

      if (threadIdx.x == 0) {
        block_max = best;
      }

      __syncthreads();

      // Best thread updates split
      if (threadIdx.x == block_max.key) {
        float fvalue;
        int gidx = (scan_begin - (level_node_idx * n_bins)) + threadIdx.x;
        if (threadIdx.x == 0 &&
            begin == scan_begin) {  // check at start of first tile
          fvalue = d_fidx_min_map[fidx];
        } else {
          fvalue = d_gidx_fvalue_map[gidx - 1];
        }

        gpair_sum_t left = missing_left ? bin + missing : bin;
        gpair_sum_t right = parent_sum - left;

        split.Update(gain, missing_left ? LeftDir : RightDir, fvalue, fidx,
                     left, right, gpu_param);
      }
      __syncthreads();
    }  // end scan
  }    // end over features

  // Create node
  if (threadIdx.x == 0 && split.IsValid()) {
    d_nodes[node_idx].SetSplit(split);

    DeviceNodeStats& left_child = d_nodes[left_child_nidx(node_idx)];
    DeviceNodeStats& right_child = d_nodes[right_child_nidx(node_idx)];
    bool& left_child_smallest = d_left_child_smallest_temp[node_idx];
    left_child =
        DeviceNodeStats(split.left_sum, left_child_nidx(node_idx), gpu_param);

    right_child =
        DeviceNodeStats(split.right_sum, right_child_nidx(node_idx), gpu_param);

    // Record smallest node
    if (split.left_sum.GetHess() <= split.right_sum.GetHess()) {
      left_child_smallest = true;
    } else {
      left_child_smallest = false;
    }
  }
}
class GPUHistMaker : public TreeUpdater {
 public:
  GPUHistMaker()
      : initialised(false),
        is_dense(false),
        p_last_fmat_(nullptr),
        prediction_cache_initialised(false) {}
  ~GPUHistMaker() {}
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    CHECK(param.max_depth < 16) << "Tree depth too large.";
    CHECK(param.max_depth != 0) << "Tree depth cannot be 0.";
    CHECK(param.grow_policy != TrainParam::kLossGuide)
        << "Loss guided growth policy not supported. Use CPU algorithm.";
    this->param = param;

    CHECK(param.n_gpus != 0) << "Must have at least one device";
  }
  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
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
  }

  void InitData(const std::vector<bst_gpair>& gpair, DMatrix& fmat,  // NOLINT
                const RegTree& tree) {
    common::Timer time1;
    // set member num_rows and n_devices for rest of GPUHistBuilder members
    info = &fmat.info();
    CHECK(info->num_row < std::numeric_limits<bst_uint>::max());
    num_rows = static_cast<bst_uint>(info->num_row);
    n_devices = dh::n_devices(param.n_gpus, num_rows);

    if (!initialised) {
      // Check gradients are within acceptable size range
      CheckGradientMax(gpair);

      // Check compute capability is high enough
      dh::check_compute_capability();

      // reset static timers used across iterations
      cpu_init_time = 0;
      gpu_init_time = 0;
      cpu_time.Reset();
      gpu_time = 0;

      // set dList member
      dList.resize(n_devices);
      for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
        int device_idx = (param.gpu_id + d_idx) % dh::n_visible_devices();
        dList[d_idx] = device_idx;
      }

      // initialize nccl
      reducer.Init(dList);

      is_dense = info->num_nonzero == info->num_col * info->num_row;
      common::Timer time0;
      hmat_.Init(&fmat, param.max_bin);
      cpu_init_time += time0.ElapsedSeconds();
      if (param.debug_verbose) {  // Only done once for each training session
        LOG(CONSOLE) << "[GPU Plug-in] CPU Time for hmat_.Init "
                     << time0.ElapsedSeconds() << " sec";
        fflush(stdout);
      }
      time0.Reset();

      gmat_.cut = &hmat_;
      cpu_init_time += time0.ElapsedSeconds();
      if (param.debug_verbose) {  // Only done once for each training session
        LOG(CONSOLE) << "[GPU Plug-in] CPU Time for gmat_.cut "
                     << time0.ElapsedSeconds() << " sec";
        fflush(stdout);
      }
      time0.Reset();

      gmat_.Init(&fmat);
      cpu_init_time += time0.ElapsedSeconds();
      if (param.debug_verbose) {  // Only done once for each training session
        LOG(CONSOLE) << "[GPU Plug-in] CPU Time for gmat_.Init() "
                     << time0.ElapsedSeconds() << " sec";
        fflush(stdout);
      }
      time0.Reset();

      if (param.debug_verbose) {  // Only done once for each training session
        LOG(CONSOLE)
            << "[GPU Plug-in] CPU Time for hmat_.Init, gmat_.cut, gmat_.Init "
            << cpu_init_time << " sec";
        fflush(stdout);
      }

      int n_bins = static_cast<int>(hmat_.row_ptr.back());
      int n_features = static_cast<int>(hmat_.row_ptr.size() - 1);

      // deliniate data onto multiple gpus
      device_row_segments.push_back(0);
      device_element_segments.push_back(0);
      bst_uint offset = 0;
      bst_uint shard_size = static_cast<bst_uint>(
          std::ceil(static_cast<double>(num_rows) / n_devices));
      for (int d_idx = 0; d_idx < n_devices; d_idx++) {
        int device_idx = dList[d_idx];
        offset += shard_size;
        offset = std::min(offset, num_rows);
        device_row_segments.push_back(offset);
        device_element_segments.push_back(gmat_.row_ptr[offset]);
      }

      // Build feature segments
      std::vector<int> h_feature_segments;
      for (int node = 0; node < n_nodes_level(param.max_depth - 1); node++) {
        for (int fidx = 0; fidx < n_features; fidx++) {
          h_feature_segments.push_back(hmat_.row_ptr[fidx] + node * n_bins);
        }
      }
      h_feature_segments.push_back(n_nodes_level(param.max_depth - 1) * n_bins);

      // Construct feature map
      std::vector<int> h_gidx_feature_map(n_bins);
      for (int fidx = 0; fidx < n_features; fidx++) {
        for (auto i = hmat_.row_ptr[fidx]; i < hmat_.row_ptr[fidx + 1]; i++) {
          h_gidx_feature_map[i] = fidx;
        }
      }

      int level_max_bins = n_nodes_level(param.max_depth - 1) * n_bins;

      // allocate unique common data that reside on master device (NOTE: None
      // currently)
      //    int master_device=dList[0];
      //    ba.allocate(master_device, );

      // allocate vectors across all devices
      temp_memory.resize(n_devices);
      hist_vec.resize(n_devices);
      nodes.resize(n_devices);
      left_child_smallest.resize(n_devices);
      feature_flags.resize(n_devices);
      fidx_min_map.resize(n_devices);
      feature_segments.resize(n_devices);
      prediction_cache.resize(n_devices);
      position.resize(n_devices);
      position_tmp.resize(n_devices);
      device_matrix.resize(n_devices);
      device_gpair.resize(n_devices);
      gidx_feature_map.resize(n_devices);
      gidx_fvalue_map.resize(n_devices);

      // num_rows_segment: for sharding rows onto gpus for splitting data
      // num_elements_segment: for sharding rows (of elements) onto gpus for
      // splitting data
      // max_num_nodes_device: for sharding nodes onto gpus for split finding
      // All other variables have full copy on gpu, with copy either being
      // identical or just current portion (like for histogram) before
      // AllReduce
      for (int d_idx = 0; d_idx < n_devices; d_idx++) {
        int device_idx = dList[d_idx];
        bst_uint num_rows_segment =
            device_row_segments[d_idx + 1] - device_row_segments[d_idx];
        bst_ulong num_elements_segment =
            device_element_segments[d_idx + 1] - device_element_segments[d_idx];

        // ensure allocation doesn't overflow
        size_t hist_size = static_cast<size_t>(n_nodes(param.max_depth - 1)) *
                           static_cast<size_t>(n_bins);
        size_t nodes_size = static_cast<size_t>(n_nodes(param.max_depth));
        size_t hmat_size = static_cast<size_t>(hmat_.min_val.size());
        size_t buffer_size = static_cast<size_t>(
            common::CompressedBufferWriter::CalculateBufferSize(
                static_cast<size_t>(num_elements_segment),
                static_cast<size_t>(n_bins)));

        ba.allocate(
            device_idx, param.silent, &(hist_vec[d_idx].data), hist_size,
            &nodes[d_idx], n_nodes(param.max_depth),
            &left_child_smallest[d_idx], nodes_size, &feature_flags[d_idx],
            n_features,  // may change but same on all devices
            &fidx_min_map[d_idx],
            hmat_size,  // constant and same on all devices
            &feature_segments[d_idx],
            h_feature_segments.size(),  // constant and same on all devices
            &prediction_cache[d_idx], num_rows_segment, &position[d_idx],
            num_rows_segment, &position_tmp[d_idx], num_rows_segment,
            &device_gpair[d_idx], num_rows_segment,
            &device_matrix[d_idx].gidx_buffer,
            buffer_size,  // constant and same on all devices
            &device_matrix[d_idx].row_ptr, num_rows_segment + 1,
            &gidx_feature_map[d_idx],
            n_bins,  // constant and same on all devices
            &gidx_fvalue_map[d_idx],
            hmat_.cut.size());  // constant and same on all devices

        // Copy Host to Device (assumes comes after ba.allocate that sets
        // device)
        device_matrix[d_idx].Init(
            device_idx, gmat_, device_element_segments[d_idx],
            device_element_segments[d_idx + 1], device_row_segments[d_idx],
            device_row_segments[d_idx + 1], n_bins);
        gidx_feature_map[d_idx] = h_gidx_feature_map;
        gidx_fvalue_map[d_idx] = hmat_.cut;
        feature_segments[d_idx] = h_feature_segments;
        fidx_min_map[d_idx] = hmat_.min_val;

        // Initialize, no copy
        hist_vec[d_idx].Init(n_bins);     // init host object
        prediction_cache[d_idx].fill(0);  // init device object (assumes comes
                                          // after ba.allocate that sets device)
        feature_flags[d_idx].fill(
            1);  // init device object (assumes comes after
                 // ba.allocate that sets device)
      }
    }

    // copy or init to do every iteration
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      nodes[d_idx].fill(DeviceNodeStats());

      position[d_idx].fill(0);

      device_gpair[d_idx].copy(gpair.begin() + device_row_segments[d_idx],
                               gpair.begin() + device_row_segments[d_idx + 1]);

      subsample_gpair(&device_gpair[d_idx], param.subsample,
                      device_row_segments[d_idx]);

      hist_vec[d_idx].Reset(device_idx);

      // left_child_smallest and left_child_smallest_temp don't need to be
      // initialized
    }

    dh::synchronize_n_devices(n_devices, dList);

    if (!initialised) {
      gpu_init_time = time1.ElapsedSeconds() - cpu_init_time;
      gpu_time = -cpu_init_time;
      if (param.debug_verbose) {  // Only done once for each training session
        LOG(CONSOLE) << "[GPU Plug-in] Time for GPU operations during First "
                        "Call to InitData() "
                     << gpu_init_time << " sec";
        fflush(stdout);
      }
    }

    p_last_fmat_ = &fmat;

    initialised = true;
  }

  void BuildHist(int depth) {
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      size_t begin = device_element_segments[d_idx];
      size_t end = device_element_segments[d_idx + 1];
      size_t row_begin = device_row_segments[d_idx];
      size_t row_end = device_row_segments[d_idx + 1];

      auto d_gidx = device_matrix[d_idx].gidx;
      auto d_row_ptr = device_matrix[d_idx].row_ptr.tbegin();
      auto d_position = position[d_idx].data();
      auto d_gpair = device_gpair[d_idx].data();
      auto d_left_child_smallest = left_child_smallest[d_idx].data();
      auto hist_builder = hist_vec[d_idx].GetBuilder();
      dh::TransformLbs(
          device_idx, &temp_memory[d_idx], end - begin, d_row_ptr,
          row_end - row_begin, is_dense,
          [=] __device__(size_t local_idx, int local_ridx) {
            int nidx = d_position[local_ridx];  // OPTMARK: latency
            if (!is_active(nidx, depth)) return;

            // Only increment smallest node
            bool is_smallest = (d_left_child_smallest[parent_nidx(nidx)] &&
                                is_left_child(nidx)) ||
                               (!d_left_child_smallest[parent_nidx(nidx)] &&
                                !is_left_child(nidx));
            if (!is_smallest && depth > 0) return;

            int gidx = d_gidx[local_idx];
            bst_gpair gpair = d_gpair[local_ridx];

            hist_builder.Add(gpair, gidx,
                             nidx);  // OPTMARK: This is slow, could use
                                     // shared memory or cache results
                                     // intead of writing to global
                                     // memory every time in atomic way.
          });
    }

    dh::synchronize_n_devices(n_devices, dList);

    //  time.printElapsed("Add Time");

    // (in-place) reduce each element of histogram (for only current level)
    // across multiple gpus
    // TODO(JCM): use out of place with pre-allocated buffer, but then have to
    // copy
    // back on device
    //  fprintf(stderr,"sizeof(bst_gpair)/sizeof(float)=%d\n",sizeof(bst_gpair)/sizeof(float));
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      reducer.AllReduceSum(device_idx,
                           reinterpret_cast<gpair_sum_t::value_t*>(
                               hist_vec[d_idx].GetLevelPtr(depth)),
                           reinterpret_cast<gpair_sum_t::value_t*>(
                               hist_vec[d_idx].GetLevelPtr(depth)),
                           hist_vec[d_idx].LevelSize(depth) *
                               sizeof(gpair_sum_t) /
                               sizeof(gpair_sum_t::value_t));
    }
    reducer.Synchronize();

    //  time.printElapsed("Reduce-Add Time");

    // Subtraction trick (applied to all devices in same way -- to avoid doing
    // on master and then Bcast)
    if (depth > 0) {
      for (int d_idx = 0; d_idx < n_devices; d_idx++) {
        int device_idx = dList[d_idx];
        dh::safe_cuda(cudaSetDevice(device_idx));

        auto hist_builder = hist_vec[d_idx].GetBuilder();
        auto d_left_child_smallest = left_child_smallest[d_idx].data();
        int n_sub_bins = (n_nodes_level(depth) / 2) * hist_builder.n_bins;

        dh::launch_n(device_idx, n_sub_bins, [=] __device__(int idx) {
          int nidx = n_nodes(depth - 1) + ((idx / hist_builder.n_bins) * 2);
          bool left_smallest = d_left_child_smallest[parent_nidx(nidx)];
          if (left_smallest) {
            nidx++;  // If left is smallest switch to right child
          }

          int gidx = idx % hist_builder.n_bins;
          gpair_sum_t parent = hist_builder.Get(gidx, parent_nidx(nidx));
          int other_nidx = left_smallest ? nidx - 1 : nidx + 1;
          gpair_sum_t other = hist_builder.Get(gidx, other_nidx);
          gpair_sum_t sub = parent - other;
          hist_builder.Add(
              bst_gpair(sub.GetGrad(), sub.GetHess()), gidx,
              nidx);  // OPTMARK: This is slow, could use shared
                      // memory or cache results intead of writing to
                      // global memory every time in atomic way.
        });
      }
      dh::synchronize_n_devices(n_devices, dList);
    }
  }
#define MIN_BLOCK_THREADS 128
#define CHUNK_BLOCK_THREADS 128
// MAX_BLOCK_THREADS of 1024 is hard-coded maximum block size due
// to CUDA capability 35 and above requirement
// for Maximum number of threads per block
#define MAX_BLOCK_THREADS 512

  void FindSplit(int depth) {
    // Specialised based on max_bins
    this->FindSplitSpecialize(depth, Int<MIN_BLOCK_THREADS>());
  }

  template <int BLOCK_THREADS>
  void FindSplitSpecialize(int depth, Int<BLOCK_THREADS>) {
    if (param.max_bin <= BLOCK_THREADS) {
      LaunchFindSplit<BLOCK_THREADS>(depth);
    } else {
      this->FindSplitSpecialize(depth,
                                Int<BLOCK_THREADS + CHUNK_BLOCK_THREADS>());
    }
  }

  void FindSplitSpecialize(int depth, Int<MAX_BLOCK_THREADS>) {
    this->LaunchFindSplit<MAX_BLOCK_THREADS>(depth);
  }

  template <int BLOCK_THREADS>
  void LaunchFindSplit(int depth) {
    bool colsample =
        param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;

    int num_nodes_device = n_nodes_level(depth);
    const int GRID_SIZE = num_nodes_device;

    // all GPUs do same work
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      int nodes_offset_device = 0;
      find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
          hist_vec[d_idx].GetLevelPtr(depth), feature_segments[d_idx].data(),
          depth, info->num_col, hmat_.row_ptr.back(), nodes[d_idx].data(),
          nodes_offset_device, fidx_min_map[d_idx].data(),
          gidx_fvalue_map[d_idx].data(), GPUTrainingParam(param),
          left_child_smallest[d_idx].data(), colsample,
          feature_flags[d_idx].data());
    }

    // NOTE: No need to syncrhonize with host as all above pure P2P ops or
    // on-device ops
  }
  void InitFirstNode(const std::vector<bst_gpair>& gpair) {
    // Perform asynchronous reduction on each gpu
    std::vector<bst_gpair> device_sums(n_devices);
#pragma omp parallel for num_threads(n_devices)
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));
      auto begin = device_gpair[d_idx].tbegin();
      auto end = device_gpair[d_idx].tend();
      bst_gpair init = bst_gpair();
      auto binary_op = thrust::plus<bst_gpair>();
      device_sums[d_idx] = thrust::reduce(begin, end, init, binary_op);
    }

    bst_gpair sum = bst_gpair();
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      sum += device_sums[d_idx];
    }

    // Setup first node so all devices have same first node (here done same on
    // all devices, or could have done one device and Bcast if worried about
    // exact precision issues)
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];

      auto d_nodes = nodes[d_idx].data();
      auto gpu_param = GPUTrainingParam(param);

      dh::launch_n(device_idx, 1, [=] __device__(int idx) {
        bst_gpair sum_gradients = sum;
        d_nodes[idx] = DeviceNodeStats(sum_gradients, 0, gpu_param);
      });
    }
    // synch all devices to host before moving on (No, can avoid because
    // BuildHist calls another kernel in default stream)
    //  dh::synchronize_n_devices(n_devices, dList);
  }
  void UpdatePosition(int depth) {
    if (is_dense) {
      this->UpdatePositionDense(depth);
    } else {
      this->UpdatePositionSparse(depth);
    }
  }
  void UpdatePositionDense(int depth) {
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];

      auto d_position = position[d_idx].data();
      DeviceNodeStats* d_nodes = nodes[d_idx].data();
      auto d_gidx_fvalue_map = gidx_fvalue_map[d_idx].data();
      auto d_gidx = device_matrix[d_idx].gidx;
      auto n_columns = info->num_col;
      size_t begin = device_row_segments[d_idx];
      size_t end = device_row_segments[d_idx + 1];

      dh::launch_n(device_idx, end - begin, [=] __device__(size_t local_idx) {
        int pos = d_position[local_idx];
        if (!is_active(pos, depth)) {
          return;
        }
        DeviceNodeStats node = d_nodes[pos];

        if (node.IsLeaf()) {
          return;
        }

        int gidx = d_gidx[local_idx * static_cast<size_t>(n_columns) +
                          static_cast<size_t>(node.fidx)];

        float fvalue = d_gidx_fvalue_map[gidx];

        if (fvalue <= node.fvalue) {
          d_position[local_idx] = left_child_nidx(pos);
        } else {
          d_position[local_idx] = right_child_nidx(pos);
        }
      });
    }
    dh::synchronize_n_devices(n_devices, dList);
    // dh::safe_cuda(cudaDeviceSynchronize());
  }

  void UpdatePositionSparse(int depth) {
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];

      auto d_position = position[d_idx].data();
      auto d_position_tmp = position_tmp[d_idx].data();
      DeviceNodeStats* d_nodes = nodes[d_idx].data();
      auto d_gidx_feature_map = gidx_feature_map[d_idx].data();
      auto d_gidx_fvalue_map = gidx_fvalue_map[d_idx].data();
      auto d_gidx = device_matrix[d_idx].gidx;
      auto d_row_ptr = device_matrix[d_idx].row_ptr.tbegin();

      size_t row_begin = device_row_segments[d_idx];
      size_t row_end = device_row_segments[d_idx + 1];
      size_t element_begin = device_element_segments[d_idx];
      size_t element_end = device_element_segments[d_idx + 1];

      // Update missing direction
      dh::launch_n(device_idx, row_end - row_begin,
                   [=] __device__(int local_idx) {
                     int pos = d_position[local_idx];
                     if (!is_active(pos, depth)) {
                       d_position_tmp[local_idx] = pos;
                       return;
                     }

                     DeviceNodeStats node = d_nodes[pos];

                     if (node.IsLeaf()) {
                       d_position_tmp[local_idx] = pos;
                       return;
                     } else if (node.dir == LeftDir) {
                       d_position_tmp[local_idx] = pos * 2 + 1;
                     } else {
                       d_position_tmp[local_idx] = pos * 2 + 2;
                     }
                   });

      // Update node based on fvalue where exists
      // OPTMARK: This kernel is very inefficient for both compute and memory,
      // dominated by memory dependency / access patterns

      dh::TransformLbs(
          device_idx, &temp_memory[d_idx], element_end - element_begin,
          d_row_ptr, row_end - row_begin, is_dense,
          [=] __device__(size_t local_idx, int local_ridx) {
            int pos = d_position[local_ridx];
            if (!is_active(pos, depth)) {
              return;
            }

            DeviceNodeStats node = d_nodes[pos];

            if (node.IsLeaf()) {
              return;
            }

            int gidx = d_gidx[local_idx];
            int findex =
                d_gidx_feature_map[gidx];  // OPTMARK: slowest global
                                           // memory access, maybe setup
                                           // position, gidx, etc. as
                                           // combined structure?

            if (findex == node.fidx) {
              float fvalue = d_gidx_fvalue_map[gidx];

              if (fvalue <= node.fvalue) {
                d_position_tmp[local_ridx] = left_child_nidx(pos);
              } else {
                d_position_tmp[local_ridx] = right_child_nidx(pos);
              }
            }
          });
      position[d_idx] = position_tmp[d_idx];
    }
    dh::synchronize_n_devices(n_devices, dList);
  }
  void ColSampleTree() {
    if (param.colsample_bylevel == 1.0 && param.colsample_bytree == 1.0) return;

    feature_set_tree.resize(info->num_col);
    std::iota(feature_set_tree.begin(), feature_set_tree.end(), 0);
    feature_set_tree = col_sample(feature_set_tree, param.colsample_bytree);
  }
  void ColSampleLevel() {
    if (param.colsample_bylevel == 1.0 && param.colsample_bytree == 1.0) return;

    feature_set_level.resize(feature_set_tree.size());
    feature_set_level = col_sample(feature_set_tree, param.colsample_bylevel);
    std::vector<int> h_feature_flags(info->num_col, 0);
    for (auto fidx : feature_set_level) {
      h_feature_flags[fidx] = 1;
    }

    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      feature_flags[d_idx] = h_feature_flags;
    }
    dh::synchronize_n_devices(n_devices, dList);
  }
  bool UpdatePredictionCache(const DMatrix* data,
                             std::vector<bst_float>* p_out_preds) override {
    std::vector<bst_float>& out_preds = *p_out_preds;

    if (nodes.empty() || !p_last_fmat_ || data != p_last_fmat_) {
      return false;
    }

    if (!prediction_cache_initialised) {
      for (int d_idx = 0; d_idx < n_devices; d_idx++) {
        int device_idx = dList[d_idx];
        size_t row_begin = device_row_segments[d_idx];
        size_t row_end = device_row_segments[d_idx + 1];

        prediction_cache[d_idx].copy(out_preds.begin() + row_begin,
                                     out_preds.begin() + row_end);
      }
      prediction_cache_initialised = true;
    }
    dh::synchronize_n_devices(n_devices, dList);

    float eps = param.learning_rate;
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      size_t row_begin = device_row_segments[d_idx];
      size_t row_end = device_row_segments[d_idx + 1];

      auto d_nodes = nodes[d_idx].data();
      auto d_position = position[d_idx].data();
      auto d_prediction_cache = prediction_cache[d_idx].data();

      dh::launch_n(device_idx, prediction_cache[d_idx].size(),
                   [=] __device__(int local_idx) {
                     int pos = d_position[local_idx];
                     d_prediction_cache[local_idx] += d_nodes[pos].weight * eps;
                   });

      dh::safe_cuda(
          cudaMemcpy(&out_preds[row_begin], prediction_cache[d_idx].data(),
                     prediction_cache[d_idx].size() * sizeof(bst_float),
                     cudaMemcpyDeviceToHost));
    }
    dh::synchronize_n_devices(n_devices, dList);

    return true;
  }
  void UpdateTree(const std::vector<bst_gpair>& gpair, DMatrix* p_fmat,
                  RegTree* p_tree) {
    common::Timer time0;

    this->InitData(gpair, *p_fmat, *p_tree);
    this->InitFirstNode(gpair);
    this->ColSampleTree();

    for (int depth = 0; depth < param.max_depth; depth++) {
      this->ColSampleLevel();
      this->BuildHist(depth);
      this->FindSplit(depth);
      this->UpdatePosition(depth);
    }

    // done with multi-GPU, pass back result from master to tree on host
    int master_device = dList[0];
    dh::safe_cuda(cudaSetDevice(master_device));
    dense2sparse_tree(p_tree, nodes[0], param);

    gpu_time += time0.ElapsedSeconds();

    if (param.debug_verbose) {
      LOG(CONSOLE)
          << "[GPU Plug-in] Cumulative GPU Time excluding initial time "
          << (gpu_time - gpu_init_time) << " sec";
      fflush(stdout);
    }

    if (param.debug_verbose) {
      LOG(CONSOLE) << "[GPU Plug-in] Cumulative CPU Time "
                   << cpu_time.ElapsedSeconds() << " sec";
      LOG(CONSOLE)
          << "[GPU Plug-in] Cumulative CPU Time excluding initial time "
          << (cpu_time.ElapsedSeconds() - cpu_init_time - gpu_time) << " sec";
      fflush(stdout);
    }
  }

 protected:
  TrainParam param;
  // std::unique_ptr<GPUHistBuilder> builder;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo* info;
  bool initialised;
  bool is_dense;
  const DMatrix* p_last_fmat_;
  bool prediction_cache_initialised;

  dh::bulk_allocator<dh::memory_type::DEVICE> ba;

  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;

  bst_uint num_rows;
  int n_devices;

  // below vectors are for each devices used
  std::vector<int> dList;
  std::vector<int> device_row_segments;
  std::vector<size_t> device_element_segments;

  std::vector<dh::CubMemory> temp_memory;
  std::vector<DeviceHist> hist_vec;
  std::vector<dh::dvec<DeviceNodeStats>> nodes;
  std::vector<dh::dvec<bool>> left_child_smallest;
  std::vector<dh::dvec<int>> feature_flags;
  std::vector<dh::dvec<float>> fidx_min_map;
  std::vector<dh::dvec<int>> feature_segments;
  std::vector<dh::dvec<bst_float>> prediction_cache;
  std::vector<dh::dvec<int>> position;
  std::vector<dh::dvec<int>> position_tmp;
  std::vector<DeviceGMat> device_matrix;
  std::vector<dh::dvec<bst_gpair>> device_gpair;
  std::vector<dh::dvec<int>> gidx_feature_map;
  std::vector<dh::dvec<float>> gidx_fvalue_map;

  dh::AllReducer reducer;

  double cpu_init_time;
  double gpu_init_time;
  common::Timer cpu_time;
  double gpu_time;
};

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
}  // namespace tree
}  // namespace xgboost
