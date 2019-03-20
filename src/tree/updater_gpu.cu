/*!
 * Copyright 2017-2018 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include <utility>
#include <vector>
#include <limits>
#include <string>

#include "../common/common.h"
#include "param.h"
#include "updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu);

template <typename GradientPairT>
XGBOOST_DEVICE float inline LossChangeMissing(const GradientPairT& scan,
                                              const GradientPairT& missing,
                                              const GradientPairT& parent_sum,
                                              const float& parent_gain,
                                              const GPUTrainingParam& param,
                                              bool& missing_left_out) {  // NOLINT
  // Put gradients of missing values to left
  float missing_left_loss =
      DeviceCalcLossChange(param, scan + missing, parent_sum, parent_gain);
  float missing_right_loss =
      DeviceCalcLossChange(param, scan, parent_sum, parent_gain);

  if (missing_left_loss >= missing_right_loss) {
    missing_left_out = true;
    return missing_left_loss;
  } else {
    missing_left_out = false;
    return missing_right_loss;
  }
}

/**
 * @brief Absolute BFS order IDs to col-wise unique IDs based on user input
 * @param tid the index of the element that this thread should access
 * @param abs the array of absolute IDs
 * @param colIds the array of column IDs for each element
 * @param nodeStart the start of the node ID at this level
 * @param nKeys number of nodes at this level.
 * @return the uniq key
 */
static HOST_DEV_INLINE NodeIdT Abs2UniqueKey(int tid,
                                             common::Span<const NodeIdT> abs,
                                             common::Span<const int> colIds,
                                             NodeIdT nodeStart, int nKeys) {
  int a = abs[tid];
  if (a == kUnusedNode) return a;
  return ((a - nodeStart) + (colIds[tid] * nKeys));
}

/**
 * @struct Pair
 * @brief Pair used for key basd scan operations on GradientPair
 */
struct Pair {
  int key;
  GradientPair value;
};

/** define a key that's not used at all in the entire boosting process */
static const int kNoneKey = -100;

/**
 * @brief Allocate temporary buffers needed for scan operations
 * @param tmpScans gradient buffer
 * @param tmpKeys keys buffer
 * @param size number of elements that will be scanned
 */
template <int BLKDIM_L1L3 = 256>
int ScanTempBufferSize(int size) {
  int num_blocks = dh::DivRoundUp(size, BLKDIM_L1L3);
  return num_blocks;
}

struct AddByKey {
  template <typename T>
  HOST_DEV_INLINE T operator()(const T& first, const T& second) const {
    T result;
    if (first.key == second.key) {
      result.key = first.key;
      result.value = first.value + second.value;
    } else {
      result.key = second.key;
      result.value = second.value;
    }
    return result;
  }
};

/**
 * @brief Gradient value getter function
 * @param id the index into the vals or instIds array to which to fetch
 * @param vals the gradient value buffer
 * @param instIds instance index buffer
 * @return the expected gradient value
 */
HOST_DEV_INLINE GradientPair Get(int id,
                                 common::Span<const GradientPair> vals,
                                 common::Span<const int> instIds) {
  id = instIds[id];
  return vals[id];
}

template <int BLKDIM_L1L3>
__global__ void CubScanByKeyL1(
    common::Span<GradientPair> scans,
    common::Span<const GradientPair> vals,
    common::Span<const int> instIds,
    common::Span<GradientPair> mScans,
    common::Span<int> mKeys,
    common::Span<const NodeIdT> keys,
    int nUniqKeys,
    common::Span<const int> colIds, NodeIdT nodeStart,
    const int size) {
  Pair rootPair = {kNoneKey, GradientPair(0.f, 0.f)};
  int myKey;
  GradientPair myValue;
  using BlockScan = cub::BlockScan<Pair, BLKDIM_L1L3>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  Pair threadData;
  int tid = blockIdx.x * BLKDIM_L1L3 + threadIdx.x;
  if (tid < size) {
    myKey = Abs2UniqueKey(tid, keys, colIds, nodeStart, nUniqKeys);
    myValue = Get(tid, vals, instIds);
  } else {
    myKey = kNoneKey;
    myValue = {};
  }
  threadData.key = myKey;
  threadData.value = myValue;
  // get previous key, especially needed for the last thread in this block
  // in order to pass on the partial scan values.
  // this statement MUST appear before the checks below!
  // else, the result of this shuffle operation will be undefined
#if (__CUDACC_VER_MAJOR__ >= 9)
  int previousKey = __shfl_up_sync(0xFFFFFFFF, myKey, 1);
#else
  int previousKey = __shfl_up(myKey, 1);
#endif
  // Collectively compute the block-wide exclusive prefix sum
  BlockScan(temp_storage)
      .ExclusiveScan(threadData, threadData, rootPair, AddByKey());
  if (tid < size) {
    scans[tid] = threadData.value;
  } else {
    return;
  }
  if (threadIdx.x == BLKDIM_L1L3 - 1) {
    threadData.value =
        (myKey == previousKey) ? threadData.value : GradientPair(0.0f, 0.0f);
    mKeys[blockIdx.x] = myKey;
    mScans[blockIdx.x] = threadData.value + myValue;
  }
}

template <int BLKSIZE>
__global__ void CubScanByKeyL2(common::Span<GradientPair> mScans,
                               common::Span<int> mKeys, int mLength) {
  using BlockScan = cub::BlockScan<Pair, BLKSIZE, cub::BLOCK_SCAN_WARP_SCANS>;
  Pair threadData;
  __shared__ typename BlockScan::TempStorage temp_storage;
  for (int i = threadIdx.x; i < mLength; i += BLKSIZE - 1) {
    threadData.key = mKeys[i];
    threadData.value = mScans[i];
    BlockScan(temp_storage).InclusiveScan(threadData, threadData, AddByKey());
    mScans[i] = threadData.value;
    __syncthreads();
  }
}

template <int BLKDIM_L1L3>
__global__ void CubScanByKeyL3(common::Span<GradientPair> sums,
                               common::Span<GradientPair> scans,
                               common::Span<const GradientPair> vals,
                               common::Span<const int> instIds,
                               common::Span<const GradientPair> mScans,
                               common::Span<const int> mKeys,
                               common::Span<const NodeIdT> keys,
                               int nUniqKeys,
                               common::Span<const int> colIds, NodeIdT nodeStart,
                               const int size) {
  int relId = threadIdx.x;
  int tid = (blockIdx.x * BLKDIM_L1L3) + relId;
  // to avoid the following warning from nvcc:
  //   __shared__ memory variable with non-empty constructor or destructor
  //     (potential race between threads)
  __shared__ char gradBuff[sizeof(GradientPair)];
  __shared__ int s_mKeys;
  GradientPair* s_mScans = reinterpret_cast<GradientPair*>(gradBuff);
  if (tid >= size) return;
  // cache block-wide partial scan info
  if (relId == 0) {
    s_mKeys = (blockIdx.x > 0) ? mKeys[blockIdx.x - 1] : kNoneKey;
    s_mScans[0] = (blockIdx.x > 0) ? mScans[blockIdx.x - 1] : GradientPair();
  }
  int myKey = Abs2UniqueKey(tid, keys, colIds, nodeStart, nUniqKeys);
  int previousKey =
      tid == 0 ? kNoneKey
               : Abs2UniqueKey(tid - 1, keys, colIds, nodeStart, nUniqKeys);
  GradientPair my_value = scans[tid];
  __syncthreads();
  if (blockIdx.x > 0 && s_mKeys == previousKey) {
    my_value += s_mScans[0];
  }
  if (tid == size - 1) {
    sums[previousKey] = my_value + Get(tid, vals, instIds);
  }
  if ((previousKey != myKey) && (previousKey >= 0)) {
    sums[previousKey] = my_value;
    my_value = GradientPair(0.0f, 0.0f);
  }
  scans[tid] = my_value;
}

/**
 * @brief Performs fused reduce and scan by key functionality. It is assumed
 * that
 *  the keys occur contiguously!
 * @param sums the output gradient reductions for each element performed
 * key-wise
 * @param scans the output gradient scans for each element performed key-wise
 * @param vals the gradients evaluated for each observation.
 * @param instIds instance ids for each element
 * @param keys keys to be used to segment the reductions. They need not occur
 *  contiguously in contrast to scan_by_key. Currently, we need one key per
 *  value in the 'vals' array.
 * @param size number of elements in the 'vals' array
 * @param nUniqKeys max number of uniq keys found per column
 * @param nCols number of columns
 * @param tmpScans temporary scan buffer needed for cub-pyramid algo
 * @param tmpKeys temporary key buffer needed for cub-pyramid algo
 * @param colIds column indices for each element in the array
 * @param nodeStart index of the leftmost node in the current level
 */
template <int BLKDIM_L1L3 = 256, int BLKDIM_L2 = 512>
void ReduceScanByKey(common::Span<GradientPair> sums,
                     common::Span<GradientPair> scans,
                     common::Span<GradientPair> vals,
                     common::Span<const int> instIds,
                     common::Span<const NodeIdT> keys,
                     int size, int nUniqKeys, int nCols,
                     common::Span<GradientPair> tmpScans,
                     common::Span<int> tmpKeys,
                     common::Span<const int> colIds, NodeIdT nodeStart) {
  int nBlks = dh::DivRoundUp(size, BLKDIM_L1L3);
  cudaMemset(sums.data(), 0, nUniqKeys * nCols * sizeof(GradientPair));
  CubScanByKeyL1<BLKDIM_L1L3>
      <<<nBlks, BLKDIM_L1L3>>>(scans, vals, instIds, tmpScans, tmpKeys, keys,
                               nUniqKeys, colIds, nodeStart, size);
  CubScanByKeyL2<BLKDIM_L2><<<1, BLKDIM_L2>>>(tmpScans, tmpKeys, nBlks);
  CubScanByKeyL3<BLKDIM_L1L3>
      <<<nBlks, BLKDIM_L1L3>>>(sums, scans, vals, instIds, tmpScans, tmpKeys,
                               keys, nUniqKeys, colIds, nodeStart, size);
}

/**
 * @struct ExactSplitCandidate
 * @brief Abstraction of a possible split in the decision tree
 */
struct ExactSplitCandidate {
  /** the optimal gain score for this node */
  float score;
  /** index where to split in the DMatrix */
  int index;

  HOST_DEV_INLINE ExactSplitCandidate() : score{-FLT_MAX}, index{INT_MAX} {}

  /**
   * @brief Whether the split info is valid to be used to create a new child
   * @param minSplitLoss minimum score above which decision to split is made
   * @return true if splittable, else false
   */
  HOST_DEV_INLINE bool IsSplittable(float minSplitLoss) const {
    return ((score >= minSplitLoss) && (index != INT_MAX));
  }
};

/**
 * @enum ArgMaxByKeyAlgo best_split_evaluation.cuh
 * @brief Help decide which algorithm to use for multi-argmax operation
 */
enum ArgMaxByKeyAlgo {
  /** simplest, use gmem-atomics for all updates */
  kAbkGmem = 0,
  /** use smem-atomics for updates (when number of keys are less) */
  kAbkSmem
};

/** max depth until which to use shared mem based atomics for argmax */
static const int kMaxAbkLevels = 3;

HOST_DEV_INLINE ExactSplitCandidate MaxSplit(ExactSplitCandidate a,
                                             ExactSplitCandidate b) {
  ExactSplitCandidate out;
  if (a.score < b.score) {
    out.score = b.score;
    out.index = b.index;
  } else if (a.score == b.score) {
    out.score = a.score;
    out.index = (a.index < b.index) ? a.index : b.index;
  } else {
    out.score = a.score;
    out.index = a.index;
  }
  return out;
}

DEV_INLINE void AtomicArgMax(ExactSplitCandidate* address,
                             ExactSplitCandidate val) {
  unsigned long long* intAddress = reinterpret_cast<unsigned long long*>(address);  // NOLINT
  unsigned long long old = *intAddress;                           // NOLINT
  unsigned long long assumed = old;                               // NOLINT
  do {
    assumed = old;
    ExactSplitCandidate res =
        MaxSplit(val, *reinterpret_cast<ExactSplitCandidate*>(&assumed));
    old = atomicCAS(intAddress, assumed, *reinterpret_cast<uint64_t*>(&res));
  } while (assumed != old);
}

DEV_INLINE void ArgMaxWithAtomics(
    int id,
    common::Span<ExactSplitCandidate> nodeSplits,
    common::Span<const GradientPair> gradScans,
    common::Span<const GradientPair> gradSums,
    common::Span<const float> vals,
    common::Span<const int> colIds,
    common::Span<const NodeIdT> nodeAssigns,
    common::Span<const DeviceNodeStats> nodes, int nUniqKeys,
    NodeIdT nodeStart, int len,
    const GPUTrainingParam& param) {
  int nodeId = nodeAssigns[id];
  // @todo: this is really a bad check! but will be fixed when we move
  //  to key-based reduction
  if ((id == 0) ||
      !((nodeId == nodeAssigns[id - 1]) && (colIds[id] == colIds[id - 1]) &&
        (vals[id] == vals[id - 1]))) {
    if (nodeId != kUnusedNode) {
      int sumId = Abs2UniqueKey(id, nodeAssigns, colIds, nodeStart, nUniqKeys);
      GradientPair colSum = gradSums[sumId];
      int uid = nodeId - nodeStart;
      DeviceNodeStats node_stat = nodes[nodeId];
      GradientPair parentSum = node_stat.sum_gradients;
      float parentGain = node_stat.root_gain;
      bool tmp;
      ExactSplitCandidate s;
      GradientPair missing = parentSum - colSum;
      s.score = LossChangeMissing(gradScans[id], missing, parentSum, parentGain,
                                  param, tmp);
      s.index = id;
      AtomicArgMax(&nodeSplits[uid], s);
    }  // end if nodeId != UNUSED_NODE
  }    // end if id == 0 ...
}

__global__ void AtomicArgMaxByKeyGmem(
    common::Span<ExactSplitCandidate> nodeSplits,
    common::Span<const GradientPair> gradScans,
    common::Span<const GradientPair> gradSums,
    common::Span<const float> vals,
    common::Span<const int> colIds,
    common::Span<const NodeIdT> nodeAssigns,
    common::Span<const DeviceNodeStats> nodes,
    int nUniqKeys,
    NodeIdT nodeStart,
    int len,
    const TrainParam param) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    ArgMaxWithAtomics(id, nodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len,
                      GPUTrainingParam(param));
  }
}

__global__ void AtomicArgMaxByKeySmem(
    common::Span<ExactSplitCandidate> nodeSplits,
    common::Span<const GradientPair> gradScans,
    common::Span<const GradientPair> gradSums,
    common::Span<const float> vals,
    common::Span<const int> colIds,
    common::Span<const NodeIdT> nodeAssigns,
    common::Span<const DeviceNodeStats> nodes,
    int nUniqKeys, NodeIdT nodeStart, int len, const GPUTrainingParam param) {
  extern __shared__ char sArr[];
  common::Span<ExactSplitCandidate> sNodeSplits =
      common::Span<ExactSplitCandidate>(
          reinterpret_cast<ExactSplitCandidate*>(sArr),
          static_cast<typename common::Span<ExactSplitCandidate>::index_type>(
              nUniqKeys * sizeof(ExactSplitCandidate)));
  int tid = threadIdx.x;
  ExactSplitCandidate defVal;

  for (int i = tid; i < nUniqKeys; i += blockDim.x) {
    sNodeSplits[i] = defVal;
  }
  __syncthreads();
  int id = tid + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    ArgMaxWithAtomics(id, sNodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len, param);
  }
  __syncthreads();
  for (int i = tid; i < nUniqKeys; i += blockDim.x) {
    ExactSplitCandidate s = sNodeSplits[i];
    AtomicArgMax(&nodeSplits[i], s);
  }
}

/**
 * @brief Performs argmax_by_key functionality but for cases when keys need not
 *  occur contiguously
 * @param nodeSplits will contain information on best split for each node
 * @param gradScans exclusive sum on sorted segments for each col
 * @param gradSums gradient sum for each column in DMatrix based on to node-ids
 * @param vals feature values
 * @param colIds column index for each element in the feature values array
 * @param nodeAssigns node-id assignments to each element in DMatrix
 * @param nodes pointer to all nodes for this tree in BFS order
 * @param nUniqKeys number of unique node-ids in this level
 * @param nodeStart start index of the node-ids in this level
 * @param len number of elements
 * @param param training parameters
 * @param algo which algorithm to use for argmax_by_key
 */
template <int BLKDIM = 256, int ITEMS_PER_THREAD = 4>
void ArgMaxByKey(common::Span<ExactSplitCandidate> nodeSplits,
                 common::Span<const GradientPair> gradScans,
                 common::Span<const GradientPair> gradSums,
                 common::Span<const float> vals,
                 common::Span<const int> colIds,
                 common::Span<const NodeIdT> nodeAssigns,
                 common::Span<const DeviceNodeStats> nodes,
                 int nUniqKeys,
                 NodeIdT nodeStart, int len, const TrainParam param,
                 ArgMaxByKeyAlgo algo) {
  dh::FillConst<ExactSplitCandidate, BLKDIM, ITEMS_PER_THREAD>(
      param.gpu_id, nodeSplits.data(), nUniqKeys,
      ExactSplitCandidate());
  int nBlks = dh::DivRoundUp(len, ITEMS_PER_THREAD * BLKDIM);
  switch (algo) {
    case kAbkGmem:
      AtomicArgMaxByKeyGmem<<<nBlks, BLKDIM>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, param);
      break;
    case kAbkSmem:
      AtomicArgMaxByKeySmem<<<nBlks, BLKDIM,
                              sizeof(ExactSplitCandidate) * nUniqKeys>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, GPUTrainingParam(param));
      break;
    default:
      throw std::runtime_error("argMaxByKey: Bad algo passed!");
  }
}

__global__ void AssignColIds(int* colIds, const int* colOffsets) {
  int myId = blockIdx.x;
  int start = colOffsets[myId];
  int end = colOffsets[myId + 1];
  for (int id = start + threadIdx.x; id < end; id += blockDim.x) {
    colIds[id] = myId;
  }
}

__global__ void FillDefaultNodeIds(NodeIdT* nodeIdsPerInst,
                                   const DeviceNodeStats* nodes, int n_rows) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id >= n_rows) {
    return;
  }
  // if this element belongs to none of the currently active node-id's
  NodeIdT nId = nodeIdsPerInst[id];
  if (nId == kUnusedNode) {
    return;
  }
  const DeviceNodeStats n = nodes[nId];
  NodeIdT result;
  if (n.IsLeaf() || n.IsUnused()) {
    result = kUnusedNode;
  } else if (n.dir == kLeftDir) {
    result = (2 * n.idx) + 1;
  } else {
    result = (2 * n.idx) + 2;
  }
  nodeIdsPerInst[id] = result;
}

__global__ void AssignNodeIds(NodeIdT* nodeIdsPerInst, int* nodeLocations,
                              const NodeIdT* nodeIds, const int* instId,
                              const DeviceNodeStats* nodes,
                              const int* colOffsets, const float* vals,
                              int nVals, int nCols) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < nVals; id += stride) {
    // fusing generation of indices for node locations
    nodeLocations[id] = id;
    // using nodeIds here since the previous kernel would have updated
    // the nodeIdsPerInst with all default assignments
    int nId = nodeIds[id];
    // if this element belongs to none of the currently active node-id's
    if (nId != kUnusedNode) {
      const DeviceNodeStats n = nodes[nId];
      int colId = n.fidx;
      // printf("nid=%d colId=%d id=%d\n", nId, colId, id);
      int start = colOffsets[colId];
      int end = colOffsets[colId + 1];
      // @todo: too much wasteful threads!!
      if ((id >= start) && (id < end) && !(n.IsLeaf() || n.IsUnused())) {
        NodeIdT result = (2 * n.idx) + 1 + (vals[id] >= n.fvalue);
        nodeIdsPerInst[instId[id]] = result;
      }
    }
  }
}

__global__ void MarkLeavesKernel(DeviceNodeStats* nodes, int len) {
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  if ((id < len) && !nodes[id].IsUnused()) {
    int lid = (id << 1) + 1;
    int rid = (id << 1) + 2;
    if ((lid >= len) || (rid >= len)) {
      nodes[id].root_gain = -FLT_MAX;  // bottom-most nodes
    } else if (nodes[lid].IsUnused() && nodes[rid].IsUnused()) {
      nodes[id].root_gain = -FLT_MAX;  // unused child nodes
    }
  }
}

class GPUMaker : public TreeUpdater {
 protected:
  TrainParam param_;
  /** whether we have initialized memory already (so as not to repeat!) */
  bool allocated_;
  /** feature values stored in column-major compressed format */
  dh::DVec2<float> vals_;
  dh::DVec<float> vals_cached_;
  /** corresponding instance id's of these featutre values */
  dh::DVec2<int> instIds_;
  dh::DVec<int> inst_ids_cached_;
  /** column offsets for these feature values */
  dh::DVec<int> colOffsets_;
  dh::DVec<GradientPair> gradsInst_;
  dh::DVec2<NodeIdT> nodeAssigns_;
  dh::DVec2<int> nodeLocations_;
  dh::DVec<DeviceNodeStats> nodes_;
  dh::DVec<NodeIdT> node_assigns_per_inst_;
  dh::DVec<GradientPair> gradsums_;
  dh::DVec<GradientPair> gradscans_;
  dh::DVec<ExactSplitCandidate> nodeSplits_;
  int n_vals_;
  int n_rows_;
  int n_cols_;
  int maxNodes_;
  int maxLeaves_;

  // devices are only used for resharding the HostDeviceVector passed as a parameter;
  // the algorithm works with a single GPU only
  GPUSet devices_;

  dh::CubMemory tmp_mem_;
  dh::DVec<GradientPair> tmpScanGradBuff_;
  dh::DVec<int> tmp_scan_key_buff_;
  dh::DVec<int> colIds_;
  dh::BulkAllocator<dh::MemoryType::kDevice> ba_;

 public:
  GPUMaker() : allocated_{false} {}
  ~GPUMaker() override = default;

  void Init(const std::vector<std::pair<std::string, std::string>> &args) override {
     param_.InitAllowUnknown(args);
     maxNodes_ = (1 << (param_.max_depth + 1)) - 1;
     maxLeaves_ = 1 << param_.max_depth;

     devices_ = GPUSet::All(param_.gpu_id, param_.n_gpus);
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    gpair->Reshard(devices_);

    try {
      // build tree
      for (auto tree : trees) {
        UpdateTree(gpair, dmat, tree);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "grow_gpu exception: " << e.what() << std::endl;
    }
    param_.learning_rate = lr;
  }
  /// @note: Update should be only after Init!!
  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
                  RegTree* hTree) {
    if (!allocated_) {
      SetupOneTimeData(dmat);
    }
    for (int i = 0; i < param_.max_depth; ++i) {
      if (i == 0) {
        // make sure to start on a fresh tree with sorted values!
        vals_.CurrentDVec() = vals_cached_;
        instIds_.CurrentDVec() = inst_ids_cached_;
        TransferGrads(gpair);
      }
      int nNodes = 1 << i;
      NodeIdT nodeStart = nNodes - 1;
      InitNodeData(i, nodeStart, nNodes);
      FindSplit(i, nodeStart, nNodes);
    }
    // mark all the used nodes with unused children as leaf nodes
    MarkLeaves();
    Dense2SparseTree(hTree, nodes_, param_);
  }

  void Split2Node(int nNodes, NodeIdT nodeStart) {
    auto d_nodes = nodes_.GetSpan();
    auto d_gradScans = gradscans_.GetSpan();
    auto d_gradsums = gradsums_.GetSpan();
    auto d_nodeAssigns = nodeAssigns_.CurrentSpan();
    auto d_colIds = colIds_.GetSpan();
    auto d_vals = vals_.Current();
    auto d_nodeSplits = nodeSplits_.Data();
    int nUniqKeys = nNodes;
    float min_split_loss = param_.min_split_loss;
    auto gpu_param = GPUTrainingParam(param_);

    dh::LaunchN(param_.gpu_id, nNodes, [=] __device__(int uid) {
      int absNodeId = uid + nodeStart;
      ExactSplitCandidate s = d_nodeSplits[uid];
      if (s.IsSplittable(min_split_loss)) {
        int idx = s.index;
        int nodeInstId =
            Abs2UniqueKey(idx, d_nodeAssigns, d_colIds, nodeStart, nUniqKeys);
        bool missingLeft = true;
        const DeviceNodeStats& n = d_nodes[absNodeId];
        GradientPair gradScan = d_gradScans[idx];
        GradientPair gradSum = d_gradsums[nodeInstId];
        float thresh = d_vals[idx];
        int colId = d_colIds[idx];
        // get the default direction for the current node
        GradientPair missing = n.sum_gradients - gradSum;
        LossChangeMissing(gradScan, missing, n.sum_gradients, n.root_gain,
                          gpu_param, missingLeft);
        // get the score/weight/id/gradSum for left and right child nodes
        GradientPair lGradSum = missingLeft ? gradScan + missing : gradScan;
        GradientPair rGradSum = n.sum_gradients - lGradSum;

        // Create children
        d_nodes[LeftChildNodeIdx(absNodeId)] =
            DeviceNodeStats(lGradSum, LeftChildNodeIdx(absNodeId), gpu_param);
        d_nodes[RightChildNodeIdx(absNodeId)] =
            DeviceNodeStats(rGradSum, RightChildNodeIdx(absNodeId), gpu_param);
        // Set split for parent
        d_nodes[absNodeId].SetSplit(thresh, colId,
                                    missingLeft ? kLeftDir : kRightDir, lGradSum,
                                    rGradSum);
      } else {
        // cannot be split further, so this node is a leaf!
        d_nodes[absNodeId].root_gain = -FLT_MAX;
      }
    });
  }

  void FindSplit(int level, NodeIdT nodeStart, int nNodes) {
    ReduceScanByKey(gradsums_.GetSpan(), gradscans_.GetSpan(), gradsInst_.GetSpan(),
                    instIds_.CurrentSpan(), nodeAssigns_.CurrentSpan(), n_vals_, nNodes,
                    n_cols_, tmpScanGradBuff_.GetSpan(), tmp_scan_key_buff_.GetSpan(),
                    colIds_.GetSpan(), nodeStart);
    ArgMaxByKey(nodeSplits_.GetSpan(), gradscans_.GetSpan(), gradsums_.GetSpan(),
                vals_.CurrentSpan(), colIds_.GetSpan(), nodeAssigns_.CurrentSpan(),
                nodes_.GetSpan(), nNodes, nodeStart, n_vals_, param_,
                level <= kMaxAbkLevels ? kAbkSmem : kAbkGmem);
    Split2Node(nNodes, nodeStart);
  }

  void AllocateAllData(int offsetSize) {
    int tmpBuffSize = ScanTempBufferSize(n_vals_);
    ba_.Allocate(param_.gpu_id, &vals_, n_vals_,
                 &vals_cached_, n_vals_, &instIds_, n_vals_, &inst_ids_cached_, n_vals_,
                 &colOffsets_, offsetSize, &gradsInst_, n_rows_, &nodeAssigns_, n_vals_,
                 &nodeLocations_, n_vals_, &nodes_, maxNodes_, &node_assigns_per_inst_,
                 n_rows_, &gradsums_, maxLeaves_ * n_cols_, &gradscans_, n_vals_,
                 &nodeSplits_, maxLeaves_, &tmpScanGradBuff_, tmpBuffSize,
                 &tmp_scan_key_buff_, tmpBuffSize, &colIds_, n_vals_);
  }

  void SetupOneTimeData(DMatrix* dmat) {
    if (!dmat->SingleColBlock()) {
      LOG(FATAL) << "exact::GPUBuilder - must have 1 column block";
    }
    std::vector<float> fval;
    std::vector<int> fId;
    std::vector<size_t> offset;
    ConvertToCsc(dmat, &fval, &fId, &offset);
    AllocateAllData(static_cast<int>(offset.size()));
    TransferAndSortData(fval, fId, offset);
    allocated_ = true;
  }

  void ConvertToCsc(DMatrix* dmat, std::vector<float>* fval,
                    std::vector<int>* fId, std::vector<size_t>* offset) {
    const MetaInfo& info = dmat->Info();
    CHECK(info.num_col_ < std::numeric_limits<int>::max());
    CHECK(info.num_row_ < std::numeric_limits<int>::max());
    n_rows_ = static_cast<int>(info.num_row_);
    n_cols_ = static_cast<int>(info.num_col_);
    offset->reserve(n_cols_ + 1);
    offset->push_back(0);
    fval->reserve(n_cols_ * n_rows_);
    fId->reserve(n_cols_ * n_rows_);
    // in case you end up with a DMatrix having no column access
    // then make sure to enable that before copying the data!
    for (const auto& batch : dmat->GetSortedColumnBatches()) {
      for (int i = 0; i < batch.Size(); i++) {
        auto col = batch[i];
        for (const Entry& e : col) {
          int inst_id = static_cast<int>(e.index);
          fval->push_back(e.fvalue);
          fId->push_back(inst_id);
        }
        offset->push_back(fval->size());
      }
    }
    CHECK(fval->size() < std::numeric_limits<int>::max());
    n_vals_ = static_cast<int>(fval->size());
  }

  void TransferAndSortData(const std::vector<float>& fval,
                           const std::vector<int>& fId,
                           const std::vector<size_t>& offset) {
    vals_.CurrentDVec() = fval;
    instIds_.CurrentDVec() = fId;
    colOffsets_ = offset;
    dh::SegmentedSort<float, int>(&tmp_mem_, &vals_, &instIds_, n_vals_, n_cols_,
                                  colOffsets_);
    vals_cached_ = vals_.CurrentDVec();
    inst_ids_cached_ = instIds_.CurrentDVec();
    AssignColIds<<<n_cols_, 512>>>(colIds_.Data(), colOffsets_.Data());
  }

  void TransferGrads(HostDeviceVector<GradientPair>* gpair) {
    gpair->GatherTo(gradsInst_.tbegin(), gradsInst_.tend());
    // evaluate the full-grad reduction for the root node
    dh::SumReduction<GradientPair>(tmp_mem_, gradsInst_, gradsums_, n_rows_);
  }

  void InitNodeData(int level, NodeIdT nodeStart, int nNodes) {
    // all instances belong to root node at the beginning!
    if (level == 0) {
      nodes_.Fill(DeviceNodeStats());
      nodeAssigns_.CurrentDVec().Fill(0);
      node_assigns_per_inst_.Fill(0);
      // for root node, just update the gradient/score/weight/id info
      // before splitting it! Currently all data is on GPU, hence this
      // stupid little kernel
      auto d_nodes = nodes_.Data();
      auto d_sums = gradsums_.Data();
      auto gpu_params = GPUTrainingParam(param_);
      dh::LaunchN(param_.gpu_id, 1, [=] __device__(int idx) {
        d_nodes[0] = DeviceNodeStats(d_sums[0], 0, gpu_params);
      });
    } else {
      const int BlkDim = 256;
      const int ItemsPerThread = 4;
      // assign default node ids first
      int nBlks = dh::DivRoundUp(n_rows_, BlkDim);
      FillDefaultNodeIds<<<nBlks, BlkDim>>>(node_assigns_per_inst_.Data(),
                                            nodes_.Data(), n_rows_);
      // evaluate the correct child indices of non-missing values next
      nBlks = dh::DivRoundUp(n_vals_, BlkDim * ItemsPerThread);
      AssignNodeIds<<<nBlks, BlkDim>>>(
          node_assigns_per_inst_.Data(), nodeLocations_.Current(),
          nodeAssigns_.Current(), instIds_.Current(), nodes_.Data(),
          colOffsets_.Data(), vals_.Current(), n_vals_, n_cols_);
      // gather the node assignments across all other columns too
      dh::Gather(param_.gpu_id, nodeAssigns_.Current(),
                 node_assigns_per_inst_.Data(), instIds_.Current(), n_vals_);
      SortKeys(level);
    }
  }

  void SortKeys(int level) {
    // segmented-sort the arrays based on node-id's
    // but we don't need more than level+1 bits for sorting!
    SegmentedSort(&tmp_mem_, &nodeAssigns_, &nodeLocations_, n_vals_, n_cols_,
                  colOffsets_, 0, level + 1);
    dh::Gather<float, int>(param_.gpu_id, vals_.other(),
                           vals_.Current(), instIds_.other(), instIds_.Current(),
                           nodeLocations_.Current(), n_vals_);
    vals_.buff().selector ^= 1;
    instIds_.buff().selector ^= 1;
  }

  void MarkLeaves() {
    const int BlkDim = 128;
    int nBlks = dh::DivRoundUp(maxNodes_, BlkDim);
    MarkLeavesKernel<<<nBlks, BlkDim>>>(nodes_.Data(), maxNodes_);
  }
};

XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker(); });

}  // namespace tree

int gpu_double_fast_compute_capable() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
  } catch(const dmlc::Error &except) {
    return 0;
  } catch(const std::exception& e) {
    return 0;
  } catch(const std::string& e) {
    return 0;
  } catch(...) {
    return 0;
  }
  for (int d_idx = 0; d_idx < n_visgpus; ++d_idx) {
    cudaDeviceProp prop;
    dh::safe_cuda(cudaGetDeviceProperties(&prop, d_idx));
    if(prop.major < 6) return 0;
  }
  return 1;
}
}  // namespace xgboost
