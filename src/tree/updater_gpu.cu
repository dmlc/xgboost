/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include <utility>
#include <vector>
#include "../common/common.h"
#include "param.h"
#include "updater_gpu_common.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu);

/**
 * @brief Absolute BFS order IDs to col-wise unique IDs based on user input
 * @param tid the index of the element that this thread should access
 * @param abs the array of absolute IDs
 * @param colIds the array of column IDs for each element
 * @param nodeStart the start of the node ID at this level
 * @param nKeys number of nodes at this level.
 * @return the uniq key
 */

static HOST_DEV_INLINE NodeIdT abs2uniqKey(int tid, const NodeIdT* abs,
                                             const int* colIds,
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
HOST_DEV_INLINE GradientPair get(int id, const GradientPair* vals,
                              const int* instIds) {
  id = instIds[id];
  return vals[id];
}

template <int BLKDIM_L1L3>
__global__ void cubScanByKeyL1(GradientPair* scans, const GradientPair* vals,
                               const int* instIds, GradientPair* mScans,
                               int* mKeys, const NodeIdT* keys, int nUniqKeys,
                               const int* colIds, NodeIdT nodeStart,
                               const int size) {
  Pair rootPair = {kNoneKey, GradientPair(0.f, 0.f)};
  int myKey;
  GradientPair myValue;
  typedef cub::BlockScan<Pair, BLKDIM_L1L3> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  Pair threadData;
  int tid = blockIdx.x * BLKDIM_L1L3 + threadIdx.x;
  if (tid < size) {
    myKey = abs2uniqKey(tid, keys, colIds, nodeStart, nUniqKeys);
    myValue = get(tid, vals, instIds);
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
  int previousKey = __shfl_up(myKey, 1);
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
__global__ void cubScanByKeyL2(GradientPair* mScans, int* mKeys, int mLength) {
  typedef cub::BlockScan<Pair, BLKSIZE, cub::BLOCK_SCAN_WARP_SCANS> BlockScan;
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
__global__ void cubScanByKeyL3(GradientPair* sums, GradientPair* scans,
                               const GradientPair* vals, const int* instIds,
                               const GradientPair* mScans, const int* mKeys,
                               const NodeIdT* keys, int nUniqKeys,
                               const int* colIds, NodeIdT nodeStart,
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
  int myKey = abs2uniqKey(tid, keys, colIds, nodeStart, nUniqKeys);
  int previousKey =
      tid == 0 ? kNoneKey
               : abs2uniqKey(tid - 1, keys, colIds, nodeStart, nUniqKeys);
  GradientPair myValue = scans[tid];
  __syncthreads();
  if (blockIdx.x > 0 && s_mKeys == previousKey) {
    myValue += s_mScans[0];
  }
  if (tid == size - 1) {
    sums[previousKey] = myValue + get(tid, vals, instIds);
  }
  if ((previousKey != myKey) && (previousKey >= 0)) {
    sums[previousKey] = myValue;
    myValue = GradientPair(0.0f, 0.0f);
  }
  scans[tid] = myValue;
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
void reduceScanByKey(GradientPair* sums, GradientPair* scans, const GradientPair* vals,
                     const int* instIds, const NodeIdT* keys, int size,
                     int nUniqKeys, int nCols, GradientPair* tmpScans,
                     int* tmpKeys, const int* colIds, NodeIdT nodeStart) {
  int nBlks = dh::DivRoundUp(size, BLKDIM_L1L3);
  cudaMemset(sums, 0, nUniqKeys * nCols * sizeof(GradientPair));
  cubScanByKeyL1<BLKDIM_L1L3>
      <<<nBlks, BLKDIM_L1L3>>>(scans, vals, instIds, tmpScans, tmpKeys, keys,
                               nUniqKeys, colIds, nodeStart, size);
  cubScanByKeyL2<BLKDIM_L2><<<1, BLKDIM_L2>>>(tmpScans, tmpKeys, nBlks);
  cubScanByKeyL3<BLKDIM_L1L3>
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

  HOST_DEV_INLINE ExactSplitCandidate() : score(-FLT_MAX), index(INT_MAX) {}

  /**
   * @brief Whether the split info is valid to be used to create a new child
   * @param minSplitLoss minimum score above which decision to split is made
   * @return true if splittable, else false
   */
  HOST_DEV_INLINE bool isSplittable(float minSplitLoss) const {
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

HOST_DEV_INLINE ExactSplitCandidate maxSplit(ExactSplitCandidate a,
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

DEV_INLINE void atomicArgMax(ExactSplitCandidate* address,
                             ExactSplitCandidate val) {
  unsigned long long* intAddress = (unsigned long long*)address;  // NOLINT
  unsigned long long old = *intAddress;                           // NOLINT
  unsigned long long assumed;                                     // NOLINT
  do {
    assumed = old;
    ExactSplitCandidate res =
        maxSplit(val, *reinterpret_cast<ExactSplitCandidate*>(&assumed));
    old = atomicCAS(intAddress, assumed, *reinterpret_cast<uint64_t*>(&res));
  } while (assumed != old);
}

DEV_INLINE void argMaxWithAtomics(
    int id, ExactSplitCandidate* nodeSplits, const GradientPair* gradScans,
    const GradientPair* gradSums, const float* vals, const int* colIds,
    const NodeIdT* nodeAssigns, const DeviceNodeStats* nodes, int nUniqKeys,
    NodeIdT nodeStart, int len, const GPUTrainingParam& param) {
  int nodeId = nodeAssigns[id];
  // @todo: this is really a bad check! but will be fixed when we move
  //  to key-based reduction
  if ((id == 0) ||
      !((nodeId == nodeAssigns[id - 1]) && (colIds[id] == colIds[id - 1]) &&
        (vals[id] == vals[id - 1]))) {
    if (nodeId != kUnusedNode) {
      int sumId = abs2uniqKey(id, nodeAssigns, colIds, nodeStart, nUniqKeys);
      GradientPair colSum = gradSums[sumId];
      int uid = nodeId - nodeStart;
      DeviceNodeStats n = nodes[nodeId];
      GradientPair parentSum = n.sum_gradients;
      float parentGain = n.root_gain;
      bool tmp;
      ExactSplitCandidate s;
      GradientPair missing = parentSum - colSum;
      s.score = LossChangeMissing(gradScans[id], missing, parentSum, parentGain,
                                 param, tmp);
      s.index = id;
      atomicArgMax(nodeSplits + uid, s);
    }  // end if nodeId != UNUSED_NODE
  }    // end if id == 0 ...
}

__global__ void atomicArgMaxByKeyGmem(
    ExactSplitCandidate* nodeSplits, const GradientPair* gradScans,
    const GradientPair* gradSums, const float* vals, const int* colIds,
    const NodeIdT* nodeAssigns, const DeviceNodeStats* nodes, int nUniqKeys,
    NodeIdT nodeStart, int len, const TrainParam param) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    argMaxWithAtomics(id, nodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len,
                      GPUTrainingParam(param));
  }
}

__global__ void atomicArgMaxByKeySmem(
    ExactSplitCandidate* nodeSplits, const GradientPair* gradScans,
    const GradientPair* gradSums, const float* vals, const int* colIds,
    const NodeIdT* nodeAssigns, const DeviceNodeStats* nodes, int nUniqKeys,
    NodeIdT nodeStart, int len, const GPUTrainingParam param) {
  extern __shared__ char sArr[];
  ExactSplitCandidate* sNodeSplits =
      reinterpret_cast<ExactSplitCandidate*>(sArr);
  int tid = threadIdx.x;
  ExactSplitCandidate defVal;
#pragma unroll 1
  for (int i = tid; i < nUniqKeys; i += blockDim.x) {
    sNodeSplits[i] = defVal;
  }
  __syncthreads();
  int id = tid + (blockIdx.x * blockDim.x);
  const int stride = blockDim.x * gridDim.x;
  for (; id < len; id += stride) {
    argMaxWithAtomics(id, sNodeSplits, gradScans, gradSums, vals, colIds,
                      nodeAssigns, nodes, nUniqKeys, nodeStart, len, param);
  }
  __syncthreads();
  for (int i = tid; i < nUniqKeys; i += blockDim.x) {
    ExactSplitCandidate s = sNodeSplits[i];
    atomicArgMax(nodeSplits + i, s);
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
void argMaxByKey(ExactSplitCandidate* nodeSplits, const GradientPair* gradScans,
                 const GradientPair* gradSums, const float* vals,
                 const int* colIds, const NodeIdT* nodeAssigns,
                 const DeviceNodeStats* nodes, int nUniqKeys,
                 NodeIdT nodeStart, int len, const TrainParam param,
                 ArgMaxByKeyAlgo algo) {
  dh::FillConst<ExactSplitCandidate, BLKDIM, ITEMS_PER_THREAD>(
      GPUSet::GetDeviceIdx(param.gpu_id), nodeSplits, nUniqKeys,
      ExactSplitCandidate());
  int nBlks = dh::DivRoundUp(len, ITEMS_PER_THREAD * BLKDIM);
  switch (algo) {
    case kAbkGmem:
      atomicArgMaxByKeyGmem<<<nBlks, BLKDIM>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, param);
      break;
    case kAbkSmem:
      atomicArgMaxByKeySmem<<<nBlks, BLKDIM,
                              sizeof(ExactSplitCandidate) * nUniqKeys>>>(
          nodeSplits, gradScans, gradSums, vals, colIds, nodeAssigns, nodes,
          nUniqKeys, nodeStart, len, GPUTrainingParam(param));
      break;
    default:
      throw std::runtime_error("argMaxByKey: Bad algo passed!");
  }
}

__global__ void assignColIds(int* colIds, const int* colOffsets) {
  int myId = blockIdx.x;
  int start = colOffsets[myId];
  int end = colOffsets[myId + 1];
  for (int id = start + threadIdx.x; id < end; id += blockDim.x) {
    colIds[id] = myId;
  }
}

__global__ void fillDefaultNodeIds(NodeIdT* nodeIdsPerInst,
                                   const DeviceNodeStats* nodes, int nRows) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id >= nRows) {
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

__global__ void assignNodeIds(NodeIdT* nodeIdsPerInst, int* nodeLocations,
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

__global__ void markLeavesKernel(DeviceNodeStats* nodes, int len) {
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
  TrainParam param;
  /** whether we have initialized memory already (so as not to repeat!) */
  bool allocated;
  /** feature values stored in column-major compressed format */
  dh::DVec2<float> vals;
  dh::DVec<float> vals_cached;
  /** corresponding instance id's of these featutre values */
  dh::DVec2<int> instIds;
  dh::DVec<int> instIds_cached;
  /** column offsets for these feature values */
  dh::DVec<int> colOffsets;
  dh::DVec<GradientPair> gradsInst;
  dh::DVec2<NodeIdT> nodeAssigns;
  dh::DVec2<int> nodeLocations;
  dh::DVec<DeviceNodeStats> nodes;
  dh::DVec<NodeIdT> nodeAssignsPerInst;
  dh::DVec<GradientPair> gradSums;
  dh::DVec<GradientPair> gradScans;
  dh::DVec<ExactSplitCandidate> nodeSplits;
  int nVals;
  int nRows;
  int nCols;
  int maxNodes;
  int maxLeaves;

  // devices are only used for resharding the HostDeviceVector passed as a parameter;
  // the algorithm works with a single GPU only
  GPUSet devices_;

  dh::CubMemory tmp_mem;
  dh::DVec<GradientPair> tmpScanGradBuff;
  dh::DVec<int> tmpScanKeyBuff;
  dh::DVec<int> colIds;
  dh::BulkAllocator<dh::MemoryType::kDevice> ba;

 public:
  GPUMaker() : allocated(false) {}
  ~GPUMaker() {}

  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    maxNodes = (1 << (param.max_depth + 1)) - 1;
    maxLeaves = 1 << param.max_depth;

    devices_ = GPUSet::All(param.n_gpus).Normalised(param.gpu_id);
  }

  void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    GradStats::CheckInfo(dmat->Info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();

    gpair->Reshard(devices_);

    try {
      // build tree
      for (size_t i = 0; i < trees.size(); ++i) {
        UpdateTree(gpair, dmat, trees[i]);
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
    }
    param.learning_rate = lr;
  }
  /// @note: Update should be only after Init!!
  void UpdateTree(HostDeviceVector<GradientPair>* gpair, DMatrix* dmat,
                  RegTree* hTree) {
    if (!allocated) {
      setupOneTimeData(dmat);
    }
    for (int i = 0; i < param.max_depth; ++i) {
      if (i == 0) {
        // make sure to start on a fresh tree with sorted values!
        vals.CurrentDVec() = vals_cached;
        instIds.CurrentDVec() = instIds_cached;
        transferGrads(gpair);
      }
      int nNodes = 1 << i;
      NodeIdT nodeStart = nNodes - 1;
      initNodeData(i, nodeStart, nNodes);
      findSplit(i, nodeStart, nNodes);
    }
    // mark all the used nodes with unused children as leaf nodes
    markLeaves();
    Dense2SparseTree(hTree, nodes, param);
  }

  void split2node(int nNodes, NodeIdT nodeStart) {
    auto d_nodes = nodes.Data();
    auto d_gradScans = gradScans.Data();
    auto d_gradSums = gradSums.Data();
    auto d_nodeAssigns = nodeAssigns.Current();
    auto d_colIds = colIds.Data();
    auto d_vals = vals.Current();
    auto d_nodeSplits = nodeSplits.Data();
    int nUniqKeys = nNodes;
    float min_split_loss = param.min_split_loss;
    auto gpu_param = GPUTrainingParam(param);

    dh::LaunchN(param.gpu_id, nNodes, [=] __device__(int uid) {
      int absNodeId = uid + nodeStart;
      ExactSplitCandidate s = d_nodeSplits[uid];
      if (s.isSplittable(min_split_loss)) {
        int idx = s.index;
        int nodeInstId =
            abs2uniqKey(idx, d_nodeAssigns, d_colIds, nodeStart, nUniqKeys);
        bool missingLeft = true;
        const DeviceNodeStats& n = d_nodes[absNodeId];
        GradientPair gradScan = d_gradScans[idx];
        GradientPair gradSum = d_gradSums[nodeInstId];
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

  void findSplit(int level, NodeIdT nodeStart, int nNodes) {
    reduceScanByKey(gradSums.Data(), gradScans.Data(), gradsInst.Data(),
                    instIds.Current(), nodeAssigns.Current(), nVals, nNodes,
                    nCols, tmpScanGradBuff.Data(), tmpScanKeyBuff.Data(),
                    colIds.Data(), nodeStart);
    argMaxByKey(nodeSplits.Data(), gradScans.Data(), gradSums.Data(),
                vals.Current(), colIds.Data(), nodeAssigns.Current(),
                nodes.Data(), nNodes, nodeStart, nVals, param,
                level <= kMaxAbkLevels ? kAbkSmem : kAbkGmem);
    split2node(nNodes, nodeStart);
  }

  void allocateAllData(int offsetSize) {
    int tmpBuffSize = ScanTempBufferSize(nVals);
    ba.Allocate(GPUSet::GetDeviceIdx(param.gpu_id), param.silent, &vals, nVals,
                &vals_cached, nVals, &instIds, nVals, &instIds_cached, nVals,
                &colOffsets, offsetSize, &gradsInst, nRows, &nodeAssigns, nVals,
                &nodeLocations, nVals, &nodes, maxNodes, &nodeAssignsPerInst,
                nRows, &gradSums, maxLeaves * nCols, &gradScans, nVals,
                &nodeSplits, maxLeaves, &tmpScanGradBuff, tmpBuffSize,
                &tmpScanKeyBuff, tmpBuffSize, &colIds, nVals);
  }

  void setupOneTimeData(DMatrix* dmat) {
    size_t free_memory = dh::AvailableMemory(GPUSet::GetDeviceIdx(param.gpu_id));
    if (!dmat->SingleColBlock()) {
      throw std::runtime_error("exact::GPUBuilder - must have 1 column block");
    }
    std::vector<float> fval;
    std::vector<int> fId;
    std::vector<size_t> offset;
    convertToCsc(dmat, &fval, &fId, &offset);
    allocateAllData(static_cast<int>(offset.size()));
    transferAndSortData(fval, fId, offset);
    allocated = true;
  }

  void convertToCsc(DMatrix* dmat, std::vector<float>* fval,
                    std::vector<int>* fId, std::vector<size_t>* offset) {
    const MetaInfo& info = dmat->Info();
    CHECK(info.num_col_ < std::numeric_limits<int>::max());
    CHECK(info.num_row_ < std::numeric_limits<int>::max());
    nRows = static_cast<int>(info.num_row_);
    nCols = static_cast<int>(info.num_col_);
    offset->reserve(nCols + 1);
    offset->push_back(0);
    fval->reserve(nCols * nRows);
    fId->reserve(nCols * nRows);
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
    nVals = static_cast<int>(fval->size());
  }

  void transferAndSortData(const std::vector<float>& fval,
                           const std::vector<int>& fId,
                           const std::vector<size_t>& offset) {
    vals.CurrentDVec() = fval;
    instIds.CurrentDVec() = fId;
    colOffsets = offset;
    dh::SegmentedSort<float, int>(&tmp_mem, &vals, &instIds, nVals, nCols,
                                  colOffsets);
    vals_cached = vals.CurrentDVec();
    instIds_cached = instIds.CurrentDVec();
    assignColIds<<<nCols, 512>>>(colIds.Data(), colOffsets.Data());
  }

  void transferGrads(HostDeviceVector<GradientPair>* gpair) {
    gpair->GatherTo(gradsInst.tbegin(), gradsInst.tend());
    // evaluate the full-grad reduction for the root node
    dh::SumReduction<GradientPair>(tmp_mem, gradsInst, gradSums, nRows);
  }

  void initNodeData(int level, NodeIdT nodeStart, int nNodes) {
    // all instances belong to root node at the beginning!
    if (level == 0) {
      nodes.Fill(DeviceNodeStats());
      nodeAssigns.CurrentDVec().Fill(0);
      nodeAssignsPerInst.Fill(0);
      // for root node, just update the gradient/score/weight/id info
      // before splitting it! Currently all data is on GPU, hence this
      // stupid little kernel
      auto d_nodes = nodes.Data();
      auto d_sums = gradSums.Data();
      auto gpu_params = GPUTrainingParam(param);
      dh::LaunchN(param.gpu_id, 1, [=] __device__(int idx) {
        d_nodes[0] = DeviceNodeStats(d_sums[0], 0, gpu_params);
      });
    } else {
      const int BlkDim = 256;
      const int ItemsPerThread = 4;
      // assign default node ids first
      int nBlks = dh::DivRoundUp(nRows, BlkDim);
      fillDefaultNodeIds<<<nBlks, BlkDim>>>(nodeAssignsPerInst.Data(),
                                            nodes.Data(), nRows);
      // evaluate the correct child indices of non-missing values next
      nBlks = dh::DivRoundUp(nVals, BlkDim * ItemsPerThread);
      assignNodeIds<<<nBlks, BlkDim>>>(
          nodeAssignsPerInst.Data(), nodeLocations.Current(),
          nodeAssigns.Current(), instIds.Current(), nodes.Data(),
          colOffsets.Data(), vals.Current(), nVals, nCols);
      // gather the node assignments across all other columns too
      dh::Gather(GPUSet::GetDeviceIdx(param.gpu_id), nodeAssigns.Current(),
                 nodeAssignsPerInst.Data(), instIds.Current(), nVals);
      sortKeys(level);
    }
  }

  void sortKeys(int level) {
    // segmented-sort the arrays based on node-id's
    // but we don't need more than level+1 bits for sorting!
    SegmentedSort(&tmp_mem, &nodeAssigns, &nodeLocations, nVals, nCols,
                  colOffsets, 0, level + 1);
    dh::Gather<float, int>(GPUSet::GetDeviceIdx(param.gpu_id), vals.other(),
                           vals.Current(), instIds.other(), instIds.Current(),
                           nodeLocations.Current(), nVals);
    vals.buff().selector ^= 1;
    instIds.buff().selector ^= 1;
  }

  void markLeaves() {
    const int BlkDim = 128;
    int nBlks = dh::DivRoundUp(maxNodes, BlkDim);
    markLeavesKernel<<<nBlks, BlkDim>>>(nodes.Data(), maxNodes);
  }
};

XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker(); });

}  // namespace tree
}  // namespace xgboost
