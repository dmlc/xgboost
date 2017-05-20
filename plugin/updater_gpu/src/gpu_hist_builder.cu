/*!
 * Copyright 2017 Rory mitchell
 */
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <functional>
#include <numeric>
#include "common.cuh"
#include "device_helpers.cuh"
#include "gpu_hist_builder.cuh"

namespace xgboost {
namespace tree {

void DeviceGMat::Init(const common::GHistIndexMatrix& gmat) {
  CHECK_EQ(gidx.size(), gmat.index.size())
      << "gidx must be externally allocated";
  CHECK_EQ(ridx.size(), gmat.index.size())
      << "ridx must be externally allocated";

  gidx = gmat.index;
  thrust::device_vector<int> row_ptr = gmat.row_ptr;

  auto counting = thrust::make_counting_iterator(0);
  thrust::upper_bound(row_ptr.begin(), row_ptr.end(), counting,
                      counting + gidx.size(), ridx.tbegin());
  thrust::transform(ridx.tbegin(), ridx.tend(), ridx.tbegin(),
                    [=] __device__(int val) { return val - 1; });
}

void DeviceHist::Init(int n_bins_in) {
  this->n_bins = n_bins_in;
  CHECK(!hist.empty()) << "DeviceHist must be externally allocated";
}

void DeviceHist::Reset() { hist.fill(gpu_gpair()); }

gpu_gpair* DeviceHist::GetLevelPtr(int depth) {
  return hist.data() + n_nodes(depth - 1) * n_bins;
}

int DeviceHist::LevelSize(int depth) { return n_bins * n_nodes_level(depth); }

HistBuilder DeviceHist::GetBuilder() {
  return HistBuilder(hist.data(), n_bins);
}

HistBuilder::HistBuilder(gpu_gpair* ptr, int n_bins)
    : d_hist(ptr), n_bins(n_bins) {}

__device__ void HistBuilder::Add(gpu_gpair gpair, int gidx, int nidx) const {
  int hist_idx = nidx * n_bins + gidx;
  atomicAdd(&(d_hist[hist_idx]._grad), gpair._grad);
  atomicAdd(&(d_hist[hist_idx]._hess), gpair._hess);
}

__device__ gpu_gpair HistBuilder::Get(int gidx, int nidx) const {
  return d_hist[nidx * n_bins + gidx];
}

GPUHistBuilder::GPUHistBuilder()
    : initialised(false),
      is_dense(false),
      p_last_fmat_(nullptr),
      prediction_cache_initialised(false) {}

GPUHistBuilder::~GPUHistBuilder() {}

void GPUHistBuilder::Init(const TrainParam& param) {
  CHECK(param.max_depth < 16) << "Tree depth too large.";
  CHECK(param.grow_policy != TrainParam::kLossGuide)
      << "Loss guided growth policy not supported. Use CPU algorithm.";
  this->param = param;

  dh::safe_cuda(cudaSetDevice(param.gpu_id));
  if (!param.silent) {
    LOG(CONSOLE) << "Device: [" << param.gpu_id << "] " << dh::device_name();
  }
}

void GPUHistBuilder::BuildHist(int depth) {
  auto d_ridx = device_matrix.ridx.data();
  auto d_gidx = device_matrix.gidx.data();
  auto d_position = position.data();
  auto d_gpair = device_gpair.data();
  auto hist_builder = hist.GetBuilder();
  auto d_left_child_smallest = left_child_smallest.data();

  dh::launch_n(device_matrix.gidx.size(), [=] __device__(int idx) {
    int ridx = d_ridx[idx];
    int pos = d_position[ridx];
    if (!is_active(pos, depth)) return;

    // Only increment smallest node
    bool is_smallest =
        (d_left_child_smallest[parent_nidx(pos)] && is_left_child(pos)) ||
        (!d_left_child_smallest[parent_nidx(pos)] && !is_left_child(pos));
    if (!is_smallest && depth > 0) return;

    int gidx = d_gidx[idx];
    gpu_gpair gpair = d_gpair[ridx];

    hist_builder.Add(gpair, gidx, pos);
  });

  dh::safe_cuda(cudaDeviceSynchronize());

  // Subtraction trick
  int n_sub_bins = (n_nodes_level(depth) / 2) * hist_builder.n_bins;
  if (depth > 0) {
    dh::launch_n(n_sub_bins, [=] __device__(int idx) {
      int nidx = n_nodes(depth - 1) + ((idx / hist_builder.n_bins) * 2);
      bool left_smallest = d_left_child_smallest[parent_nidx(nidx)];
      if (left_smallest) {
        nidx++;  // If left is smallest switch to right child
      }

      int gidx = idx % hist_builder.n_bins;
      gpu_gpair parent = hist_builder.Get(gidx, parent_nidx(nidx));
      int other_nidx = left_smallest ? nidx - 1 : nidx + 1;
      gpu_gpair other = hist_builder.Get(gidx, other_nidx);
      hist_builder.Add(parent - other, gidx, nidx);
    });
  }
  dh::safe_cuda(cudaDeviceSynchronize());
}

template <int BLOCK_THREADS>
__global__ void find_split_kernel(
    const gpu_gpair* d_level_hist, int* d_feature_segments, int depth,
    int n_features, int n_bins, Node* d_nodes, float* d_fidx_min_map,
    float* d_gidx_fvalue_map, GPUTrainingParam gpu_param,
    bool* d_left_child_smallest, bool colsample, int* d_feature_flags) {
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<gpu_gpair, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;
  typedef cub::BlockReduce<gpu_gpair, BLOCK_THREADS> SumReduceT;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  struct UninitializedSplit : cub::Uninitialized<Split> {};
  struct UninitializedGpair : cub::Uninitialized<gpu_gpair> {};

  __shared__ UninitializedSplit uninitialized_split;
  Split& split = uninitialized_split.Alias();
  __shared__ ArgMaxT block_max;
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    split = Split();
  }

  __syncthreads();

  int node_idx = n_nodes(depth - 1) + blockIdx.x;

  for (int fidx = 0; fidx < n_features; fidx++) {
    if (colsample && d_feature_flags[fidx] == 0) continue;

    int begin = d_feature_segments[blockIdx.x * n_features + fidx];
    int end = d_feature_segments[blockIdx.x * n_features + fidx + 1];
    int gidx = (begin - (blockIdx.x * n_bins)) + threadIdx.x;
    bool thread_active = threadIdx.x < end - begin;

    // Scan histogram
    gpu_gpair bin =
        thread_active ? d_level_hist[begin + threadIdx.x] : gpu_gpair();

    gpu_gpair feature_sum;
    BlockScanT(temp_storage.scan)
        .ExclusiveScan(bin, bin, gpu_gpair(), cub::Sum(), feature_sum);

    // Calculate gain
    gpu_gpair parent_sum = d_nodes[node_idx].sum_gradients;
    float parent_gain = d_nodes[node_idx].root_gain;

    gpu_gpair missing = parent_sum - feature_sum;

    bool missing_left;
    float gain = thread_active
                     ? loss_chg_missing(bin, missing, parent_sum, parent_gain,
                                        gpu_param, missing_left)
                     : -FLT_MAX;
    __syncthreads();

    // Find thread with best gain
    ArgMaxT tuple(threadIdx.x, gain);
    ArgMaxT best = MaxReduceT(temp_storage.max_reduce)
                       .Reduce(tuple, cub::ArgMax(), end - begin);

    if (threadIdx.x == 0) {
      block_max = best;
    }

    __syncthreads();

    // Best thread updates split
    if (threadIdx.x == block_max.key) {
      float fvalue;
      if (threadIdx.x == 0) {
        fvalue = d_fidx_min_map[fidx];
      } else {
        fvalue = d_gidx_fvalue_map[gidx - 1];
      }

      gpu_gpair left = missing_left ? bin + missing : bin;
      gpu_gpair right = parent_sum - left;

      split.Update(gain, missing_left, fvalue, fidx, left, right, gpu_param);
    }
    __syncthreads();
  }

  // Create node
  if (threadIdx.x == 0) {
    d_nodes[node_idx].split = split;
    if (depth == 0) {
      // split.Print();
    }

    d_nodes[left_child_nidx(node_idx)] = Node(
        split.left_sum,
        CalcGain(gpu_param, split.left_sum.grad(), split.left_sum.hess()),
        CalcWeight(gpu_param, split.left_sum.grad(), split.left_sum.hess()));

    d_nodes[right_child_nidx(node_idx)] = Node(
        split.right_sum,
        CalcGain(gpu_param, split.right_sum.grad(), split.right_sum.hess()),
        CalcWeight(gpu_param, split.right_sum.grad(), split.right_sum.hess()));

    // Record smallest node
    if (split.left_sum.hess() <= split.right_sum.hess()) {
      d_left_child_smallest[node_idx] = true;
    } else {
      d_left_child_smallest[node_idx] = false;
    }
  }
}

void GPUHistBuilder::FindSplit(int depth) {
  // Specialised based on max_bins
  if (param.max_bin <= 256) {
    this->FindSplit256(depth);
  } else if (param.max_bin <= 1024) {
    this->FindSplit1024(depth);
  } else {
    this->FindSplitLarge(depth);
  }
}

void GPUHistBuilder::FindSplit256(int depth) {
  CHECK_LE(param.max_bin, 256);
  const int BLOCK_THREADS = 256;
  const int GRID_SIZE = n_nodes_level(depth);
  bool colsample =
      param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;
  find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
      hist.GetLevelPtr(depth), feature_segments.data(), depth, info->num_col,
      hmat_.row_ptr.back(), nodes.data(), fidx_min_map.data(),
      gidx_fvalue_map.data(), gpu_param, left_child_smallest.data(), colsample,
      feature_flags.data());

  dh::safe_cuda(cudaDeviceSynchronize());
}
void GPUHistBuilder::FindSplit1024(int depth) {
  CHECK_LE(param.max_bin, 1024);
  const int BLOCK_THREADS = 1024;
  const int GRID_SIZE = n_nodes_level(depth);
  bool colsample =
      param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;
  find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
      hist.GetLevelPtr(depth), feature_segments.data(), depth, info->num_col,
      hmat_.row_ptr.back(), nodes.data(), fidx_min_map.data(),
      gidx_fvalue_map.data(), gpu_param, left_child_smallest.data(), colsample,
      feature_flags.data());

  dh::safe_cuda(cudaDeviceSynchronize());
}
void GPUHistBuilder::FindSplitLarge(int depth) {
  auto counting = thrust::make_counting_iterator(0);
  auto d_gidx_feature_map = gidx_feature_map.data();
  int n_bins = hmat_.row_ptr.back();
  int n_features = hmat_.row_ptr.size() - 1;

  auto feature_boundary = [=] __device__(int idx_a, int idx_b) {
    int gidx_a = idx_a % n_bins;
    int gidx_b = idx_b % n_bins;
    return d_gidx_feature_map[gidx_a] == d_gidx_feature_map[gidx_b];
  };  // NOLINT

  // Reduce node sums
  {
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Reduce(
        nullptr, temp_storage_bytes, hist.GetLevelPtr(depth), node_sums.data(),
        n_nodes_level(depth) * n_features, feature_segments.data(),
        feature_segments.data() + 1, cub::Sum(), gpu_gpair());
    cub_mem.LazyAllocate(temp_storage_bytes);
    cub::DeviceSegmentedReduce::Reduce(
        cub_mem.d_temp_storage, cub_mem.temp_storage_bytes,
        hist.GetLevelPtr(depth), node_sums.data(),
        n_nodes_level(depth) * n_features, feature_segments.data(),
        feature_segments.data() + 1, cub::Sum(), gpu_gpair());
  }

  // Scan
  thrust::exclusive_scan_by_key(
      counting, counting + hist.LevelSize(depth),
      thrust::device_pointer_cast(hist.GetLevelPtr(depth)), hist_scan.tbegin(),
      gpu_gpair(), feature_boundary);

  // Calculate gain
  auto d_gain = gain.data();
  auto d_nodes = nodes.data();
  auto d_node_sums = node_sums.data();
  auto d_hist_scan = hist_scan.data();
  GPUTrainingParam gpu_param_alias =
      gpu_param;  // Must be local variable to be used in device lambda
  bool colsample =
      param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;
  auto d_feature_flags = feature_flags.data();

  dh::launch_n(hist.LevelSize(depth), [=] __device__(int idx) {
    int node_segment = idx / n_bins;
    int node_idx = n_nodes(depth - 1) + node_segment;
    gpu_gpair parent_sum = d_nodes[node_idx].sum_gradients;
    float parent_gain = d_nodes[node_idx].root_gain;
    int gidx = idx % n_bins;
    int findex = d_gidx_feature_map[gidx];

    // colsample
    if (colsample && d_feature_flags[d_gidx_feature_map[gidx]] == 0) {
      d_gain[idx] = 0;
    } else {
      gpu_gpair scan = d_hist_scan[idx];
      gpu_gpair sum = d_node_sums[node_segment * n_features + findex];
      gpu_gpair missing = parent_sum - sum;

      bool missing_left;
      d_gain[idx] = loss_chg_missing(scan, missing, parent_sum, parent_gain,
                                     gpu_param_alias, missing_left);
    }
  });
  dh::safe_cuda(cudaDeviceSynchronize());

  // Find best gain
  {
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::ArgMax(nullptr, temp_storage_bytes, gain.data(),
                                       argmax.data(), n_nodes_level(depth),
                                       hist_node_segments.data(),
                                       hist_node_segments.data() + 1);
    cub_mem.LazyAllocate(temp_storage_bytes);
    cub::DeviceSegmentedReduce::ArgMax(
        cub_mem.d_temp_storage, cub_mem.temp_storage_bytes, gain.data(),
        argmax.data(), n_nodes_level(depth), hist_node_segments.data(),
        hist_node_segments.data() + 1);
  }

  auto d_argmax = argmax.data();
  auto d_gidx_fvalue_map = gidx_fvalue_map.data();
  auto d_fidx_min_map = fidx_min_map.data();
  auto d_left_child_smallest = left_child_smallest.data();

  dh::launch_n(n_nodes_level(depth), [=] __device__(int idx) {
    int max_idx = n_bins * idx + d_argmax[idx].key;
    int gidx = max_idx % n_bins;
    int fidx = d_gidx_feature_map[gidx];
    int node_segment = max_idx / n_bins;
    int node_idx = n_nodes(depth - 1) + node_segment;
    gpu_gpair scan = d_hist_scan[max_idx];
    gpu_gpair parent_sum = d_nodes[node_idx].sum_gradients;
    float parent_gain = d_nodes[node_idx].root_gain;
    gpu_gpair sum = d_node_sums[node_segment * n_features + fidx];
    gpu_gpair missing = parent_sum - sum;

    bool missing_left;
    float loss_chg = loss_chg_missing(scan, missing, parent_sum, parent_gain,
                                      gpu_param_alias, missing_left);

    float fvalue;
    if (gidx == 0 || fidx != d_gidx_feature_map[gidx - 1]) {
      fvalue = d_fidx_min_map[fidx];
    } else {
      fvalue = d_gidx_fvalue_map[gidx - 1];
    }
    gpu_gpair left = missing_left ? scan + missing : scan;
    gpu_gpair right = parent_sum - left;
    d_nodes[node_idx].split.Update(loss_chg, missing_left, fvalue, fidx, left,
                                   right, gpu_param_alias);

    d_nodes[left_child_nidx(node_idx)] =
        Node(left, CalcGain(gpu_param_alias, left.grad(), left.hess()),
             CalcWeight(gpu_param_alias, left.grad(), left.hess()));

    d_nodes[right_child_nidx(node_idx)] =
        Node(right, CalcGain(gpu_param_alias, right.grad(), right.hess()),
             CalcWeight(gpu_param_alias, right.grad(), right.hess()));

    // Record smallest node
    if (left.hess() <= right.hess()) {
      d_left_child_smallest[node_idx] = true;
    } else {
      d_left_child_smallest[node_idx] = false;
    }
  });
  dh::safe_cuda(cudaDeviceSynchronize());
}

void GPUHistBuilder::InitFirstNode() {
  auto d_gpair = device_gpair.data();
  auto d_node_sums = node_sums.data();
  auto d_nodes = nodes.data();
  auto gpu_param_alias = gpu_param;

  size_t temp_storage_bytes;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, d_gpair, d_node_sums,
                            device_gpair.size(), cub::Sum(), gpu_gpair());
  cub_mem.LazyAllocate(temp_storage_bytes);
  cub::DeviceReduce::Reduce(cub_mem.d_temp_storage, cub_mem.temp_storage_bytes,
                            d_gpair, d_node_sums, device_gpair.size(),
                            cub::Sum(), gpu_gpair());

  dh::launch_n(1, [=] __device__(int idx) {
    gpu_gpair sum_gradients = d_node_sums[idx];
    d_nodes[idx] = Node(
        sum_gradients,
        CalcGain(gpu_param_alias, sum_gradients.grad(), sum_gradients.hess()),
        CalcWeight(gpu_param_alias, sum_gradients.grad(),
                   sum_gradients.hess()));
  });
}

void GPUHistBuilder::UpdatePosition(int depth) {
  if (is_dense) {
    this->UpdatePositionDense(depth);
  } else {
    this->UpdatePositionSparse(depth);
  }
}

void GPUHistBuilder::UpdatePositionDense(int depth) {
  auto d_position = position.data();
  Node* d_nodes = nodes.data();
  auto d_gidx_fvalue_map = gidx_fvalue_map.data();
  auto d_gidx = device_matrix.gidx.data();
  int n_columns = info->num_col;

  int gidx_size = device_matrix.gidx.size();

  dh::launch_n(position.size(), [=] __device__(int idx) {
    NodeIdT pos = d_position[idx];
    if (!is_active(pos, depth)) {
      return;
    }
    Node node = d_nodes[pos];

    if (node.IsLeaf()) {
      return;
    }

    int gidx = d_gidx[idx * n_columns + node.split.findex];

    float fvalue = d_gidx_fvalue_map[gidx];

    if (fvalue <= node.split.fvalue) {
      d_position[idx] = left_child_nidx(pos);
    } else {
      d_position[idx] = right_child_nidx(pos);
    }
  });

  dh::safe_cuda(cudaDeviceSynchronize());
}

void GPUHistBuilder::UpdatePositionSparse(int depth) {
  auto d_position = position.data();
  auto d_position_tmp = position_tmp.data();
  Node* d_nodes = nodes.data();
  auto d_gidx_feature_map = gidx_feature_map.data();
  auto d_gidx_fvalue_map = gidx_fvalue_map.data();
  auto d_gidx = device_matrix.gidx.data();
  auto d_ridx = device_matrix.ridx.data();

  // Update missing direction
  dh::launch_n(position.size(), [=] __device__(int idx) {
    NodeIdT pos = d_position[idx];
    if (!is_active(pos, depth)) {
      d_position_tmp[idx] = pos;
      return;
    }

    Node node = d_nodes[pos];

    if (node.IsLeaf()) {
      d_position_tmp[idx] = pos;
      return;
    } else if (node.split.missing_left) {
      d_position_tmp[idx] = pos * 2 + 1;
    } else {
      d_position_tmp[idx] = pos * 2 + 2;
    }
  });

  dh::safe_cuda(cudaDeviceSynchronize());

  // Update node based on fvalue where exists
  dh::launch_n(device_matrix.gidx.size(), [=] __device__(int idx) {
    int ridx = d_ridx[idx];
    NodeIdT pos = d_position[ridx];
    if (!is_active(pos, depth)) {
      return;
    }

    Node node = d_nodes[pos];

    if (node.IsLeaf()) {
      return;
    }

    int gidx = d_gidx[idx];
    int findex = d_gidx_feature_map[gidx];

    if (findex == node.split.findex) {
      float fvalue = d_gidx_fvalue_map[gidx];

      if (fvalue <= node.split.fvalue) {
        d_position_tmp[ridx] = left_child_nidx(pos);
      } else {
        d_position_tmp[ridx] = right_child_nidx(pos);
      }
    }
  });

  dh::safe_cuda(cudaDeviceSynchronize());

  position = position_tmp;
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
  feature_flags = h_feature_flags;
}

void GPUHistBuilder::InitData(const std::vector<bst_gpair>& gpair,
                              DMatrix& fmat,  // NOLINT
                              const RegTree& tree) {
  if (!initialised) {
    CHECK(fmat.SingleColBlock()) << "grow_gpu_hist: must have single column "
                                    "block. Try setting 'tree_method' "
                                    "parameter to 'exact'";
    info = &fmat.info();
    is_dense = info->num_nonzero == info->num_col * info->num_row;
    hmat_.Init(&fmat, param.max_bin);
    gmat_.cut = &hmat_;
    gmat_.Init(&fmat);
    int n_bins = hmat_.row_ptr.back();
    int n_features = hmat_.row_ptr.size() - 1;

    // Build feature segments
    std::vector<int> h_feature_segments;
    for (int node = 0; node < n_nodes_level(param.max_depth - 1); node++) {
      for (int fidx = 0; fidx < hmat_.row_ptr.size() - 1; fidx++) {
        h_feature_segments.push_back(hmat_.row_ptr[fidx] + node * n_bins);
      }
    }
    h_feature_segments.push_back(n_nodes_level(param.max_depth - 1) * n_bins);

    int level_max_bins = n_nodes_level(param.max_depth - 1) * n_bins;

    size_t free_memory = dh::available_memory();
    ba.allocate(&gidx_feature_map, n_bins, &hist_node_segments,
                n_nodes_level(param.max_depth - 1) + 1, &feature_segments,
                h_feature_segments.size(), &gain, level_max_bins, &position,
                gpair.size(), &position_tmp, gpair.size(), &nodes,
                n_nodes(param.max_depth), &gidx_fvalue_map, hmat_.cut.size(),
                &fidx_min_map, hmat_.min_val.size(), &argmax,
                n_nodes_level(param.max_depth - 1), &node_sums,
                n_nodes_level(param.max_depth - 1) * n_features, &hist_scan,
                level_max_bins, &device_gpair, gpair.size(),
                &device_matrix.gidx, gmat_.index.size(), &device_matrix.ridx,
                gmat_.index.size(), &hist.hist,
                n_nodes(param.max_depth - 1) * n_bins, &feature_flags,
                n_features, &left_child_smallest, n_nodes(param.max_depth - 1),
                &prediction_cache, gpair.size());

    if (!param.silent) {
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Allocated " << ba.size() / mb_size << "/"
                   << free_memory / mb_size << " MB on " << dh::device_name();
    }

    // Construct feature map
    std::vector<int> h_gidx_feature_map(n_bins);
    for (int row = 0; row < hmat_.row_ptr.size() - 1; row++) {
      for (int i = hmat_.row_ptr[row]; i < hmat_.row_ptr[row + 1]; i++) {
        h_gidx_feature_map[i] = row;
      }
    }

    gidx_feature_map = h_gidx_feature_map;

    // Construct device matrix
    device_matrix.Init(gmat_);

    gidx_fvalue_map = hmat_.cut;
    fidx_min_map = hmat_.min_val;

    thrust::sequence(hist_node_segments.tbegin(), hist_node_segments.tend(), 0,
                     n_bins);

    feature_flags.fill(1);

    feature_segments = h_feature_segments;

    hist.Init(n_bins);

    prediction_cache.fill(0);

    initialised = true;
  }
  nodes.fill(Node());
  position.fill(0);
  device_gpair = gpair;
  subsample_gpair(&device_gpair, param.subsample);
  hist.Reset();
  p_last_fmat_ = &fmat;
}

bool GPUHistBuilder::UpdatePredictionCache(
    const DMatrix* data, std::vector<bst_float>* p_out_preds) {
  std::vector<bst_float>& out_preds = *p_out_preds;

  if (nodes.empty() || !p_last_fmat_ || data != p_last_fmat_) {
    return false;
  }
  CHECK_EQ(prediction_cache.size(), out_preds.size());

  if (!prediction_cache_initialised) {
    prediction_cache = out_preds;
    prediction_cache_initialised = true;
  }

  auto d_nodes = nodes.data();
  auto d_position = position.data();
  auto d_prediction_cache = prediction_cache.data();
  float eps = param.learning_rate;

  dh::launch_n(prediction_cache.size(), [=] __device__(int idx) {
    int pos = d_position[idx];
    d_prediction_cache[idx] += d_nodes[pos].weight * eps;
  });

  thrust::copy(prediction_cache.tbegin(), prediction_cache.tend(),
               out_preds.data());

  return true;
}

void GPUHistBuilder::Update(const std::vector<bst_gpair>& gpair,
                            DMatrix* p_fmat, RegTree* p_tree) {
  this->InitData(gpair, *p_fmat, *p_tree);
  this->InitFirstNode();
  this->ColSampleTree();
  for (int depth = 0; depth < param.max_depth; depth++) {
    this->ColSampleLevel();
    this->BuildHist(depth);
    this->FindSplit(depth);
    this->UpdatePosition(depth);
  }
  dense2sparse_tree(p_tree, nodes.tbegin(), nodes.tend(), param);
}
}  // namespace tree
}  // namespace xgboost
