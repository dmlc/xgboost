/*!
 * Copyright 2017 Rory mitchell
 */
#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>
#include <functional>
#include <future>
#include <numeric>
#include "common.cuh"
#include "device_helpers.cuh"
#include "gpu_hist_builder.cuh"

namespace xgboost {
namespace tree {

void DeviceGMat::Init(int device_idx, const common::GHistIndexMatrix& gmat,
                      bst_uint begin, bst_uint end) {
  dh::safe_cuda(cudaSetDevice(device_idx));
  CHECK_EQ(gidx.size(), end - begin) << "gidx must be externally allocated";
  CHECK_EQ(ridx.size(), end - begin) << "ridx must be externally allocated";

  thrust::copy(gmat.index.data() + begin, gmat.index.data() + end, gidx.tbegin());
  thrust::device_vector<int> row_ptr = gmat.row_ptr;

  auto counting = thrust::make_counting_iterator(begin);
  thrust::upper_bound(row_ptr.begin(), row_ptr.end(), counting,
                      counting + gidx.size(), ridx.tbegin());
  thrust::transform(ridx.tbegin(), ridx.tend(), ridx.tbegin(),
                    [=] __device__(int val) { return val - 1; });
}

void DeviceHist::Init(int n_bins_in) {
  this->n_bins = n_bins_in;
  CHECK(!data.empty()) << "DeviceHist must be externally allocated";
}

void DeviceHist::Reset(int device_idx) {
  cudaSetDevice(device_idx);
  data.fill(bst_gpair());
}

bst_gpair* DeviceHist::GetLevelPtr(int depth) {
  return data.data() + n_nodes(depth - 1) * n_bins;
}

int DeviceHist::LevelSize(int depth) { return n_bins * n_nodes_level(depth); }

HistBuilder DeviceHist::GetBuilder() {
  return HistBuilder(data.data(), n_bins);
}

HistBuilder::HistBuilder(bst_gpair* ptr, int n_bins)
    : d_hist(ptr), n_bins(n_bins) {}

__device__ void HistBuilder::Add(bst_gpair gpair, int gidx, int nidx) const {
  int hist_idx = nidx * n_bins + gidx;
  atomicAdd(&(d_hist[hist_idx].grad), gpair.grad);  // OPTMARK: This and below
                                                      // line lead to about 3X
                                                      // slowdown due to memory
                                                      // dependency and access
                                                      // pattern issues.
  atomicAdd(&(d_hist[hist_idx].hess), gpair.hess);
}

__device__ bst_gpair HistBuilder::Get(int gidx, int nidx) const {
  return d_hist[nidx * n_bins + gidx];
}

GPUHistBuilder::GPUHistBuilder()
    : initialised(false),
      is_dense(false),
      p_last_fmat_(nullptr),
      prediction_cache_initialised(false) {}

GPUHistBuilder::~GPUHistBuilder() {
  if (initialised) {
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      ncclCommDestroy(comms[d_idx]);

      dh::safe_cuda(cudaSetDevice(dList[d_idx]));
      dh::safe_cuda(cudaStreamDestroy(*(streams[d_idx])));
    }
    for (int num_d = 1; num_d <= n_devices;
         ++num_d) {  // loop over number of devices used
      for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
        ncclCommDestroy(find_split_comms[num_d - 1][d_idx]);
      }
    }
  }
}

void GPUHistBuilder::Init(const TrainParam& param) {
  CHECK(param.max_depth < 16) << "Tree depth too large.";
  CHECK(param.grow_policy != TrainParam::kLossGuide)
      << "Loss guided growth policy not supported. Use CPU algorithm.";
  this->param = param;

  CHECK(param.n_gpus != 0) << "Must have at least one device";
  int n_devices_all = dh::n_devices_all(param.n_gpus);
  for (int device_idx = 0; device_idx < n_devices_all; device_idx++) {
    if (!param.silent) {
      size_t free_memory = dh::available_memory(device_idx);
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Device: [" << device_idx << "] "
                   << dh::device_name(device_idx) << " with "
                   << free_memory / mb_size << " MB available device memory.";
    }
  }
}
void GPUHistBuilder::InitData(const std::vector<bst_gpair>& gpair,
                              DMatrix& fmat,  // NOLINT
                              const RegTree& tree) {
  // set member num_rows and n_devices for rest of GPUHistBuilder members
  info = &fmat.info();
  num_rows = info->num_row;
  n_devices = dh::n_devices(param.n_gpus, num_rows);

  if (!initialised) {
    // set dList member
    dList.resize(n_devices);
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      int device_idx = (param.gpu_id + d_idx) % dh::n_visible_devices();
      dList[d_idx] = device_idx;
    }

    // initialize nccl

    comms.resize(n_devices);
    streams.resize(n_devices);
    dh::safe_nccl(ncclCommInitAll(comms.data(), n_devices,
                                  dList.data()));  // initialize communicator
                                                   // (One communicator per
                                                   // process)

    // printf("# NCCL: Using devices\n");
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {
      streams[d_idx] =
          reinterpret_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t)));
      dh::safe_cuda(cudaSetDevice(dList[d_idx]));
      dh::safe_cuda(cudaStreamCreate(streams[d_idx]));

      int cudaDev;
      int rank;
      cudaDeviceProp prop;
      dh::safe_nccl(ncclCommCuDevice(comms[d_idx], &cudaDev));
      dh::safe_nccl(ncclCommUserRank(comms[d_idx], &rank));
      dh::safe_cuda(cudaGetDeviceProperties(&prop, cudaDev));
      // printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev,
      //             prop.pciBusID, prop.name);
      fflush(stdout);
    }

    // local find_split group of comms for each case of reduced number of GPUs
    // to use
    find_split_comms.resize(
        n_devices,
        std::vector<ncclComm_t>(n_devices));  // TODO(JCM): Excessive, but
                                              // ok, and best to do
                                              // here instead of
                                              // repeatedly
    for (int num_d = 1; num_d <= n_devices;
         ++num_d) {  // loop over number of devices used
      dh::safe_nccl(ncclCommInitAll(find_split_comms[num_d - 1].data(), num_d,
                                    dList.data()));  // initialize communicator
                                                     // (One communicator per
                                                     // process)
    }


    CHECK(fmat.SingleColBlock()) << "grow_gpu_hist: must have single column "
                                    "block. Try setting 'tree_method' "
                                    "parameter to 'exact'";
    is_dense = info->num_nonzero == info->num_col * info->num_row;
    hmat_.Init(&fmat, param.max_bin);
    gmat_.cut = &hmat_;
    gmat_.Init(&fmat);
    int n_bins = hmat_.row_ptr.back();
    int n_features = hmat_.row_ptr.size() - 1;

    // deliniate data onto multiple gpus
    device_row_segments.push_back(0);
    device_element_segments.push_back(0);
    bst_uint offset = 0;
    size_t shard_size = std::ceil(static_cast<double>(num_rows) / n_devices);
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
      for (int i = hmat_.row_ptr[fidx]; i < hmat_.row_ptr[fidx + 1]; i++) {
        h_gidx_feature_map[i] = fidx;
      }
    }

    int level_max_bins = n_nodes_level(param.max_depth - 1) * n_bins;

    // allocate unique common data that reside on master device (NOTE: None
    // currently)
    //    int master_device=dList[0];
    //    ba.allocate(master_device, );

    // allocate vectors across all devices
    hist_vec.resize(n_devices);
    nodes.resize(n_devices);
    nodes_temp.resize(n_devices);
    nodes_child_temp.resize(n_devices);
    left_child_smallest.resize(n_devices);
    left_child_smallest_temp.resize(n_devices);
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

    int find_split_n_devices = std::pow(2, std::floor(std::log2(n_devices)));
    find_split_n_devices =
        std::min(n_nodes_level(param.max_depth), find_split_n_devices);
    int max_num_nodes_device =
        n_nodes_level(param.max_depth) / find_split_n_devices;

    // num_rows_segment: for sharding rows onto gpus for splitting data
    // num_elements_segment: for sharding rows (of elements) onto gpus for
    // splitting data
    // max_num_nodes_device: for sharding nodes onto gpus for split finding
    // All other variables have full copy on gpu, with copy either being
    // identical or just current portion (like for histogram) before AllReduce
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      bst_uint num_rows_segment =
          device_row_segments[d_idx + 1] - device_row_segments[d_idx];
      bst_uint num_elements_segment =
          device_element_segments[d_idx + 1] - device_element_segments[d_idx];
      ba.allocate(
          device_idx, &(hist_vec[d_idx].data),
          n_nodes(param.max_depth - 1) * n_bins, &nodes[d_idx],
          n_nodes(param.max_depth), &nodes_temp[d_idx], max_num_nodes_device,
          &nodes_child_temp[d_idx], max_num_nodes_device,
          &left_child_smallest[d_idx], n_nodes(param.max_depth),
          &left_child_smallest_temp[d_idx], max_num_nodes_device,
          &feature_flags[d_idx],
          n_features,  // may change but same on all devices
          &fidx_min_map[d_idx],
          hmat_.min_val.size(),  // constant and same on all devices
          &feature_segments[d_idx],
          h_feature_segments.size(),  // constant and same on all devices
          &prediction_cache[d_idx], num_rows_segment, &position[d_idx],
          num_rows_segment, &position_tmp[d_idx], num_rows_segment,
          &device_gpair[d_idx], num_rows_segment, &device_matrix[d_idx].gidx,
          num_elements_segment,  // constant and same on all devices
          &device_matrix[d_idx].ridx,
          num_elements_segment,              // constant and same on all devices
          &gidx_feature_map[d_idx], n_bins,  // constant and same on all devices
          &gidx_fvalue_map[d_idx],
          hmat_.cut.size());  // constant and same on all devices

      // Copy Host to Device (assumes comes after ba.allocate that sets device)
      device_matrix[d_idx].Init(device_idx, gmat_,
                                device_element_segments[d_idx],
                                device_element_segments[d_idx + 1]);
      gidx_feature_map[d_idx] = h_gidx_feature_map;
      gidx_fvalue_map[d_idx] = hmat_.cut;
      feature_segments[d_idx] = h_feature_segments;
      fidx_min_map[d_idx] = hmat_.min_val;

      // Initialize, no copy
      hist_vec[d_idx].Init(n_bins);     // init host object
      prediction_cache[d_idx].fill(0);  // init device object (assumes comes
                                        // after ba.allocate that sets device)
      feature_flags[d_idx].fill(1);  // init device object (assumes comes after
                                     // ba.allocate that sets device)
    }

    if (!param.silent) {
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Allocated " << ba.size() / mb_size << " MB";
    }

    initialised = true;
  }

  // copy or init to do every iteration
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));

    nodes[d_idx].fill(Node());
    nodes_temp[d_idx].fill(Node());
    nodes_child_temp[d_idx].fill(Node());

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

  p_last_fmat_ = &fmat;
}

void GPUHistBuilder::BuildHist(int depth) {
  //  dh::Timer time;

  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    size_t begin = device_element_segments[d_idx];
    size_t end = device_element_segments[d_idx + 1];
    size_t row_begin = device_row_segments[d_idx];

    auto d_ridx = device_matrix[d_idx].ridx.data();
    auto d_gidx = device_matrix[d_idx].gidx.data();
    auto d_position = position[d_idx].data();
    auto d_gpair = device_gpair[d_idx].data();
    auto d_left_child_smallest = left_child_smallest[d_idx].data();
    auto hist_builder = hist_vec[d_idx].GetBuilder();

    dh::launch_n(device_idx, end - begin, [=] __device__(int local_idx) {
      int ridx = d_ridx[local_idx];             // OPTMARK: latency
      int nidx = d_position[ridx - row_begin];  // OPTMARK: latency
      if (!is_active(nidx, depth)) return;

      // Only increment smallest node
      bool is_smallest =
          (d_left_child_smallest[parent_nidx(nidx)] && is_left_child(nidx)) ||
          (!d_left_child_smallest[parent_nidx(nidx)] && !is_left_child(nidx));
      if (!is_smallest && depth > 0) return;

      int gidx = d_gidx[local_idx];
      bst_gpair gpair = d_gpair[ridx - row_begin];

      hist_builder.Add(gpair, gidx, nidx);  // OPTMARK: This is slow, could use
                                            // shared memory or cache results
                                            // intead of writing to global
                                            // memory every time in atomic way.
    });
  }

  //  dh::safe_cuda(cudaDeviceSynchronize());
  dh::synchronize_n_devices(n_devices, dList);

//  time.printElapsed("Add Time");

  // (in-place) reduce each element of histogram (for only current level) across
  // multiple gpus
  // TODO(JCM): use out of place with pre-allocated buffer, but then have to
  // copy
  // back on device
  //  fprintf(stderr,"sizeof(bst_gpair)/sizeof(float)=%d\n",sizeof(bst_gpair)/sizeof(float));
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));
    dh::safe_nccl(ncclAllReduce(
        reinterpret_cast<const void*>(hist_vec[d_idx].GetLevelPtr(depth)),
        reinterpret_cast<void*>(hist_vec[d_idx].GetLevelPtr(depth)),
        hist_vec[d_idx].LevelSize(depth) * sizeof(bst_gpair) / sizeof(float),
        ncclFloat, ncclSum, comms[d_idx], *(streams[d_idx])));
  }

  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));
    dh::safe_cuda(cudaStreamSynchronize(*(streams[d_idx])));
  }
// if no NCCL, then presume only 1 GPU, then already correct

  //  time.printElapsed("Reduce-Add Time");

  // Subtraction trick (applied to all devices in same way -- to avoid doing on
  // master and then Bcast)
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
        bst_gpair parent = hist_builder.Get(gidx, parent_nidx(nidx));
        int other_nidx = left_smallest ? nidx - 1 : nidx + 1;
        bst_gpair other = hist_builder.Get(gidx, other_nidx);
        hist_builder.Add(parent - other, gidx,
                         nidx);  // OPTMARK: This is slow, could use shared
                                 // memory or cache results intead of writing to
                                 // global memory every time in atomic way.
      });
    }
    dh::synchronize_n_devices(n_devices, dList);
  }
}

template <int BLOCK_THREADS>
__global__ void find_split_kernel(
    const bst_gpair* d_level_hist, int* d_feature_segments, int depth,
    int n_features, int n_bins, Node* d_nodes, Node* d_nodes_temp,
    Node* d_nodes_child_temp, int nodes_offset_device, float* d_fidx_min_map,
    float* d_gidx_fvalue_map, GPUTrainingParam gpu_param,
    bool* d_left_child_smallest_temp, bool colsample, int* d_feature_flags) {
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<bst_gpair, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;
  typedef cub::BlockReduce<bst_gpair, BLOCK_THREADS> SumReduceT;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  struct UninitializedSplit : cub::Uninitialized<Split> {};
  struct UninitializedGpair : cub::Uninitialized<bst_gpair> {};

  __shared__ UninitializedSplit uninitialized_split;
  Split& split = uninitialized_split.Alias();
  __shared__ UninitializedGpair uninitialized_sum;
  bst_gpair& shared_sum = uninitialized_sum.Alias();
  __shared__ ArgMaxT block_max;
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    split = Split();
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
    int gidx = (begin - (level_node_idx * n_bins)) + threadIdx.x;
    bool thread_active = threadIdx.x < end - begin;

    bst_gpair feature_sum = bst_gpair();
    for (int reduce_begin = begin; reduce_begin < end;
         reduce_begin += BLOCK_THREADS) {
      // Scan histogram
      bst_gpair bin = thread_active ? d_level_hist[reduce_begin + threadIdx.x]
                                    : bst_gpair();

      feature_sum +=
          SumReduceT(temp_storage.sum_reduce).Reduce(bin, cub::Sum());
    }

    if (threadIdx.x == 0) {
      shared_sum = feature_sum;
    }
    //    __syncthreads(); // no need to synch because below there is a Scan

    GpairCallbackOp prefix_op = GpairCallbackOp();
    for (int scan_begin = begin; scan_begin < end;
         scan_begin += BLOCK_THREADS) {
      bst_gpair bin =
          thread_active ? d_level_hist[scan_begin + threadIdx.x] : bst_gpair();

      BlockScanT(temp_storage.scan)
          .ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

      // Calculate gain
      bst_gpair parent_sum = d_nodes[node_idx].sum_gradients;
      float parent_gain = d_nodes[node_idx].root_gain;

      bst_gpair missing = parent_sum - shared_sum;

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
        if (threadIdx.x == 0 &&
            begin == scan_begin) {  // check at start of first tile
          fvalue = d_fidx_min_map[fidx];
        } else {
          fvalue = d_gidx_fvalue_map[gidx - 1];
        }

        bst_gpair left = missing_left ? bin + missing : bin;
        bst_gpair right = parent_sum - left;

        split.Update(gain, missing_left, fvalue, fidx, left, right, gpu_param);
      }
      __syncthreads();
    }  // end scan
  }    // end over features

  // Create node
  if (threadIdx.x == 0) {
    if (d_nodes_temp == NULL) {
      d_nodes[node_idx].split = split;
    } else {
      d_nodes_temp[blockIdx.x] = d_nodes[node_idx];  // first copy node values
      d_nodes_temp[blockIdx.x].split = split;        // now assign split
    }

    //    if (depth == 0) {
    // split.Print();
    //    }

    Node *Nodeleft, *Noderight;
    bool* left_child_smallest;
    if (d_nodes_temp == NULL) {
      Nodeleft = &d_nodes[left_child_nidx(node_idx)];
      Noderight = &d_nodes[right_child_nidx(node_idx)];
      left_child_smallest =
          &d_left_child_smallest_temp[node_idx];  // NOTE: not per level, even
                                                  // though _temp variable name
    } else {
      Nodeleft = &d_nodes_child_temp[blockIdx.x * 2 + 0];
      Noderight = &d_nodes_child_temp[blockIdx.x * 2 + 1];
      left_child_smallest = &d_left_child_smallest_temp[blockIdx.x];
    }

    *Nodeleft = Node(
        split.left_sum,
        CalcGain(gpu_param, split.left_sum.grad, split.left_sum.hess),
        CalcWeight(gpu_param, split.left_sum.grad, split.left_sum.hess));

    *Noderight = Node(
        split.right_sum,
        CalcGain(gpu_param, split.right_sum.grad, split.right_sum.hess),
        CalcWeight(gpu_param, split.right_sum.grad, split.right_sum.hess));

    // Record smallest node
    if (split.left_sum.hess <= split.right_sum.hess) {
      *left_child_smallest = true;
    } else {
      *left_child_smallest = false;
    }
  }
}

#define MIN_BLOCK_THREADS 32
#define MAX_BLOCK_THREADS 1024  // hard-coded maximum block size

void GPUHistBuilder::FindSplit(int depth) {
  // Specialised based on max_bins
  this->FindSplitSpecialize<MIN_BLOCK_THREADS>(depth);
}

template <>
void GPUHistBuilder::FindSplitSpecialize<MAX_BLOCK_THREADS>(int depth) {
  LaunchFindSplit<MAX_BLOCK_THREADS>(depth);
}
template <int BLOCK_THREADS>
void GPUHistBuilder::FindSplitSpecialize(int depth) {
  if (param.max_bin <= BLOCK_THREADS) {
    LaunchFindSplit<BLOCK_THREADS>(depth);
  } else {
    this->FindSplitSpecialize<BLOCK_THREADS + 32>(depth);
  }
}

template <int BLOCK_THREADS>
void GPUHistBuilder::LaunchFindSplit(int depth) {
  bool colsample =
      param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;

  int dosimuljob = 1;

  int simuljob = 1;  // whether to do job on single GPU and broadcast (0) or to
                     // do same job on each GPU (1) (could make user parameter,
                     // but too fine-grained maybe)
  int findsplit_shardongpus = 0;  // too expensive generally, disable for now

  if (findsplit_shardongpus) {
    dosimuljob = 0;
    // use power of 2 for split finder because nodes are power of 2 (broadcast
    // result to remaining devices)
    int find_split_n_devices = std::pow(2, std::floor(std::log2(n_devices)));
    find_split_n_devices = std::min(n_nodes_level(depth), find_split_n_devices);
    int num_nodes_device = n_nodes_level(depth) / find_split_n_devices;
    int num_nodes_child_device =
        n_nodes_level(depth + 1) / find_split_n_devices;
    const int GRID_SIZE = num_nodes_device;

    // NOTE: No need to scatter before gather as all devices have same copy of
    // nodes, and within find_split_kernel() nodes_temp is given values from
    // nodes

    // for all nodes (split among devices) find best split per node
    for (int d_idx = 0; d_idx < find_split_n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      int nodes_offset_device = d_idx * num_nodes_device;
      find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
          (const bst_gpair*)(hist_vec[d_idx].GetLevelPtr(depth)),
          feature_segments[d_idx].data(), depth, (info->num_col),
          (hmat_.row_ptr.back()), nodes[d_idx].data(), nodes_temp[d_idx].data(),
          nodes_child_temp[d_idx].data(), nodes_offset_device,
          fidx_min_map[d_idx].data(), gidx_fvalue_map[d_idx].data(),  GPUTrainingParam(param),
          left_child_smallest_temp[d_idx].data(), colsample,
          feature_flags[d_idx].data());
    }

    // nccl only on devices that did split
    dh::synchronize_n_devices(find_split_n_devices, dList);

    for (int d_idx = 0; d_idx < find_split_n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      dh::safe_nccl(ncclAllGather(
          reinterpret_cast<const void*>(nodes_temp[d_idx].data()),
          num_nodes_device * sizeof(Node) / sizeof(char), ncclChar,
          reinterpret_cast<void*>(nodes[d_idx].data() + n_nodes(depth - 1)),
          find_split_comms[find_split_n_devices - 1][d_idx],
          *(streams[d_idx])));

      if (depth !=
          param.max_depth) {  // don't copy over children nodes if no more nodes
        dh::safe_nccl(ncclAllGather(
            reinterpret_cast<const void*>(nodes_child_temp[d_idx].data()),
            num_nodes_child_device * sizeof(Node) / sizeof(char), ncclChar,
            reinterpret_cast<void*>(nodes[d_idx].data() + n_nodes(depth)),
            find_split_comms[find_split_n_devices - 1][d_idx],
            *(streams[d_idx])));  // Note offset by n_nodes(depth)
        // for recvbuff for child nodes
      }

      dh::safe_nccl(ncclAllGather(
          reinterpret_cast<const void*>(left_child_smallest_temp[d_idx].data()),
          num_nodes_device * sizeof(bool) / sizeof(char), ncclChar,
          reinterpret_cast<void*>(left_child_smallest[d_idx].data() +
                                  n_nodes(depth - 1)),
          find_split_comms[find_split_n_devices - 1][d_idx],
          *(streams[d_idx])));
    }

    for (int d_idx = 0; d_idx < find_split_n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));
      dh::safe_cuda(cudaStreamSynchronize(*(streams[d_idx])));
    }

    if (n_devices > find_split_n_devices && n_devices > 1) {
      // if n_devices==1, no need to Bcast
      // if find_split_n_devices==1, this is just a copy operation, else it
      // copies
      // from master to all nodes in case extra devices not involved in split
      for (int d_idx = 0; d_idx < n_devices; d_idx++) {
        int device_idx = dList[d_idx];
        dh::safe_cuda(cudaSetDevice(device_idx));

        int master_device = dList[0];
        dh::safe_nccl(ncclBcast(
            reinterpret_cast<void*>(nodes[d_idx].data() + n_nodes(depth - 1)),
            n_nodes_level(depth) * sizeof(Node) / sizeof(char), ncclChar,
            master_device, comms[d_idx], *(streams[d_idx])));

        if (depth != param.max_depth) {  // don't copy over children nodes if no
                                         // more nodes
          dh::safe_nccl(ncclBcast(
              reinterpret_cast<void*>(nodes[d_idx].data() + n_nodes(depth)),
              n_nodes_level(depth + 1) * sizeof(Node) / sizeof(char), ncclChar,
              master_device, comms[d_idx], *(streams[d_idx])));
        }

        dh::safe_nccl(ncclBcast(
            reinterpret_cast<void*>(left_child_smallest[d_idx].data() +
                                    n_nodes(depth - 1)),
            n_nodes_level(depth) * sizeof(bool) / sizeof(char), ncclChar,
            master_device, comms[d_idx], *(streams[d_idx])));
      }

      for (int d_idx = 0; d_idx < n_devices; d_idx++) {
        int device_idx = dList[d_idx];
        dh::safe_cuda(cudaSetDevice(device_idx));
        dh::safe_cuda(cudaStreamSynchronize(*(streams[d_idx])));
      }
    }
  } else if (simuljob == 0) {
    dosimuljob = 0;
    int num_nodes_device = n_nodes_level(depth);
    const int GRID_SIZE = num_nodes_device;

    int d_idx = 0;
    int master_device = dList[d_idx];
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));

    int nodes_offset_device = d_idx * num_nodes_device;
    find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
        (const bst_gpair*)(hist_vec[d_idx].GetLevelPtr(depth)),
        feature_segments[d_idx].data(), depth, (info->num_col),
        (hmat_.row_ptr.back()), nodes[d_idx].data(), NULL, NULL,
        nodes_offset_device, fidx_min_map[d_idx].data(),
        gidx_fvalue_map[d_idx].data(),  GPUTrainingParam(param),
        left_child_smallest[d_idx].data(), colsample,
        feature_flags[d_idx].data());

    // broadcast result
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      dh::safe_nccl(ncclBcast(
          reinterpret_cast<void*>(nodes[d_idx].data() + n_nodes(depth - 1)),
          n_nodes_level(depth) * sizeof(Node) / sizeof(char), ncclChar,
          master_device, comms[d_idx], *(streams[d_idx])));

      if (depth !=
          param.max_depth) {  // don't copy over children nodes if no more nodes
        dh::safe_nccl(ncclBcast(
            reinterpret_cast<void*>(nodes[d_idx].data() + n_nodes(depth)),
            n_nodes_level(depth + 1) * sizeof(Node) / sizeof(char), ncclChar,
            master_device, comms[d_idx], *(streams[d_idx])));
      }

      dh::safe_nccl(
          ncclBcast(reinterpret_cast<void*>(left_child_smallest[d_idx].data() +
                                            n_nodes(depth - 1)),
                    n_nodes_level(depth) * sizeof(bool) / sizeof(char),
                    ncclChar, master_device, comms[d_idx], *(streams[d_idx])));
    }

    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));
      dh::safe_cuda(cudaStreamSynchronize(*(streams[d_idx])));
    }
  } else {
    dosimuljob = 1;
  }

  if (dosimuljob) {  // if no NCCL or simuljob==1, do this
    int num_nodes_device = n_nodes_level(depth);
    const int GRID_SIZE = num_nodes_device;

    // all GPUs do same work
    for (int d_idx = 0; d_idx < n_devices; d_idx++) {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));

      int nodes_offset_device = 0;
      find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
          (const bst_gpair*)(hist_vec[d_idx].GetLevelPtr(depth)),
          feature_segments[d_idx].data(), depth, (info->num_col),
          (hmat_.row_ptr.back()), nodes[d_idx].data(), NULL, NULL,
          nodes_offset_device, fidx_min_map[d_idx].data(),
          gidx_fvalue_map[d_idx].data(),  GPUTrainingParam(param),
          left_child_smallest[d_idx].data(), colsample,
          feature_flags[d_idx].data());
    }
  }

  // NOTE: No need to syncrhonize with host as all above pure P2P ops or
  // on-device ops
}

void GPUHistBuilder::InitFirstNode(const std::vector<bst_gpair>& gpair) {
#ifdef _WIN32
  // Visual studio complains about C:/Program Files (x86)/Microsoft Visual
  // Studio 14.0/VC/bin/../../VC/INCLUDE\utility(445): error : static assertion
  // failed with "tuple index out of bounds"
  // and C:/Program Files (x86)/Microsoft Visual Studio
  // 14.0/VC/bin/../../VC/INCLUDE\future(1888): error : no instance of function
  // template "std::_Invoke_stored" matches the argument list
  std::vector<bst_gpair> future_results(n_devices);
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];

    auto begin = device_gpair[d_idx].tbegin();
    auto end = device_gpair[d_idx].tend();
    bst_gpair init = bst_gpair();
    auto binary_op = thrust::plus<bst_gpair>();

    dh::safe_cuda(cudaSetDevice(device_idx));
    future_results[d_idx] = thrust::reduce(begin, end, init, binary_op);
  }

  // sum over devices on host (with blocking get())
  bst_gpair sum = bst_gpair();
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    sum += future_results[d_idx];
  }
#else
  // asynch reduce per device

  std::vector<std::future<bst_gpair>> future_results(n_devices);
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    // std::async captures the algorithm parameters by value
    // use std::launch::async to ensure the creation of a new thread
    future_results[d_idx] = std::async(std::launch::async, [=] {
      int device_idx = dList[d_idx];
      dh::safe_cuda(cudaSetDevice(device_idx));
      auto begin = device_gpair[d_idx].tbegin();
      auto end = device_gpair[d_idx].tend();
      bst_gpair init = bst_gpair();
      auto binary_op = thrust::plus<bst_gpair>();
      return thrust::reduce(begin, end, init, binary_op);
    });
  }

  // sum over devices on host (with blocking get())
  bst_gpair sum = bst_gpair();
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    sum += future_results[d_idx].get();
  }
#endif

  // Setup first node so all devices have same first node (here done same on all
  // devices, or could have done one device and Bcast if worried about exact
  // precision issues)
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];

    auto d_nodes = nodes[d_idx].data();
    auto gpu_param = GPUTrainingParam(param);

    dh::launch_n(device_idx, 1, [=] __device__(int idx) {
      bst_gpair sum_gradients = sum;
      d_nodes[idx] = Node(
          sum_gradients,
          CalcGain(gpu_param, sum_gradients.grad, sum_gradients.hess),
          CalcWeight(gpu_param, sum_gradients.grad,
                     sum_gradients.hess));
    });
  }
  // synch all devices to host before moving on (No, can avoid because BuildHist
  // calls another kernel in default stream)
  //  dh::synchronize_n_devices(n_devices, dList);
}

void GPUHistBuilder::UpdatePosition(int depth) {
  if (is_dense) {
    this->UpdatePositionDense(depth);
  } else {
    this->UpdatePositionSparse(depth);
  }
}

void GPUHistBuilder::UpdatePositionDense(int depth) {
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];

    auto d_position = position[d_idx].data();
    Node* d_nodes = nodes[d_idx].data();
    auto d_gidx_fvalue_map = gidx_fvalue_map[d_idx].data();
    auto d_gidx = device_matrix[d_idx].gidx.data();
    int n_columns = info->num_col;
    size_t begin = device_row_segments[d_idx];
    size_t end = device_row_segments[d_idx + 1];

    dh::launch_n(device_idx, end - begin, [=] __device__(int local_idx) {
      int pos = d_position[local_idx];
      if (!is_active(pos, depth)) {
        return;
      }
      Node node = d_nodes[pos];

      if (node.IsLeaf()) {
        return;
      }

      int gidx = d_gidx[local_idx * n_columns + node.split.findex];

      float fvalue = d_gidx_fvalue_map[gidx];

      if (fvalue <= node.split.fvalue) {
        d_position[local_idx] = left_child_nidx(pos);
      } else {
        d_position[local_idx] = right_child_nidx(pos);
      }
    });
  }
  dh::synchronize_n_devices(n_devices, dList);
  // dh::safe_cuda(cudaDeviceSynchronize());
}

void GPUHistBuilder::UpdatePositionSparse(int depth) {
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];

    auto d_position = position[d_idx].data();
    auto d_position_tmp = position_tmp[d_idx].data();
    Node* d_nodes = nodes[d_idx].data();
    auto d_gidx_feature_map = gidx_feature_map[d_idx].data();
    auto d_gidx_fvalue_map = gidx_fvalue_map[d_idx].data();
    auto d_gidx = device_matrix[d_idx].gidx.data();
    auto d_ridx = device_matrix[d_idx].ridx.data();

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

                   Node node = d_nodes[pos];

                   if (node.IsLeaf()) {
                     d_position_tmp[local_idx] = pos;
                     return;
                   } else if (node.split.missing_left) {
                     d_position_tmp[local_idx] = pos * 2 + 1;
                   } else {
                     d_position_tmp[local_idx] = pos * 2 + 2;
                   }
                 });

    // Update node based on fvalue where exists
    // OPTMARK: This kernel is very inefficient for both compute and memory,
    // dominated by memory dependency / access patterns
    dh::launch_n(
        device_idx, element_end - element_begin, [=] __device__(int local_idx) {
          int ridx = d_ridx[local_idx];
          int pos = d_position[ridx - row_begin];
          if (!is_active(pos, depth)) {
            return;
          }

          Node node = d_nodes[pos];

          if (node.IsLeaf()) {
            return;
          }

          int gidx = d_gidx[local_idx];
          int findex = d_gidx_feature_map[gidx];  // OPTMARK: slowest global
                                                  // memory access, maybe setup
                                                  // position, gidx, etc. as
                                                  // combined structure?

          if (findex == node.split.findex) {
            float fvalue = d_gidx_fvalue_map[gidx];

            if (fvalue <= node.split.fvalue) {
              d_position_tmp[ridx - row_begin] = left_child_nidx(pos);
            } else {
              d_position_tmp[ridx - row_begin] = right_child_nidx(pos);
            }
          }
        });
    position[d_idx] = position_tmp[d_idx];
  }
  dh::synchronize_n_devices(n_devices, dList);
}

void GPUHistBuilder::ColSampleTree() {
  if (param.colsample_bylevel == 1.0 && param.colsample_bytree == 1.0) return;

  feature_set_tree.resize(info->num_col);
  std::iota(feature_set_tree.begin(), feature_set_tree.end(), 0);
  feature_set_tree = col_sample(feature_set_tree, param.colsample_bytree);
}

void GPUHistBuilder::ColSampleLevel() {
  if (param.colsample_bylevel == 1.0 && param.colsample_bytree == 1.0) return;

  feature_set_level.resize(feature_set_tree.size());
  feature_set_level = col_sample(feature_set_tree, param.colsample_bylevel);
  std::vector<int> h_feature_flags(info->num_col, 0);
  for (auto fidx : feature_set_level) {
    h_feature_flags[fidx] = 1;
  }

  // copy from Host to Device for all devices
  //  for(auto &f:feature_flags){ // this doesn't set device as should
  //    f = h_feature_flags;
  //  }
  for (int d_idx = 0; d_idx < n_devices; d_idx++) {
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));

    feature_flags[d_idx] = h_feature_flags;
  }
  dh::synchronize_n_devices(n_devices, dList);
}

bool GPUHistBuilder::UpdatePredictionCache(
    const DMatrix* data, std::vector<bst_float>* p_out_preds) {
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

    thrust::copy(prediction_cache[d_idx].tbegin(),
                 prediction_cache[d_idx].tend(), &out_preds[row_begin]);
  }
  dh::synchronize_n_devices(n_devices, dList);

  return true;
}

void GPUHistBuilder::Update(const std::vector<bst_gpair>& gpair,
                            DMatrix* p_fmat, RegTree* p_tree) {
  this->InitData(gpair, *p_fmat, *p_tree);
  this->InitFirstNode(gpair);
  this->ColSampleTree();
  //  long long int elapsed=0;
  for (int depth = 0; depth < param.max_depth; depth++) {
    this->ColSampleLevel();

    //    dh::Timer time;
    this->BuildHist(depth);
    //    elapsed+=time.elapsed();
    //    printf("depth=%d\n",depth);
    //    time.printElapsed("BH Time");

    //    dh::Timer timesplit;
    this->FindSplit(depth);
    //    timesplit.printElapsed("FS Time");

    //    dh::Timer timeupdatepos;
    this->UpdatePosition(depth);
    //    timeupdatepos.printElapsed("UP Time");
  }
  //  printf("Total BuildHist Time=%lld\n",elapsed);

  // done with multi-GPU, pass back result from master to tree on host
  int master_device = dList[0];
  dh::safe_cuda(cudaSetDevice(master_device));
  dense2sparse_tree(p_tree, nodes[0].tbegin(), nodes[0].tend(), param);
}
}  // namespace tree
}  // namespace xgboost
