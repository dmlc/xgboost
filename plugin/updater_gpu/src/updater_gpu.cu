/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include <utility>
#include <vector>
#include "../../../src/common/sync.h"
#include "../../../src/tree/param.h"
#include "exact/gpu_builder.cuh"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_gpu);

__global__ void assignColIds(int* colIds, const int* colOffsets) {
  int myId = blockIdx.x;
  int start = colOffsets[myId];
  int end = colOffsets[myId + 1];
  for (int id = start + threadIdx.x; id < end; id += blockDim.x) {
    colIds[id] = myId;
  }
}

__global__ void fillDefaultNodeIds(node_id_t* nodeIdsPerInst,
                                   const DeviceDenseNode* nodes, int nRows) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  if (id >= nRows) {
    return;
  }
  // if this element belongs to none of the currently active node-id's
  node_id_t nId = nodeIdsPerInst[id];
  if (nId == UNUSED_NODE) {
    return;
  }
  const DeviceDenseNode n = nodes[nId];
  node_id_t result;
  if (n.IsLeaf() || n.IsUnused()) {
    result = UNUSED_NODE;
  } else if (n.dir == LeftDir) {
    result = (2 * n.idx) + 1;
  } else {
    result = (2 * n.idx) + 2;
  }
  nodeIdsPerInst[id] = result;
}

__global__ void assignNodeIds(node_id_t* nodeIdsPerInst, int* nodeLocations,
                              const node_id_t* nodeIds, const int* instId,
                              const DeviceDenseNode* nodes,
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
    if (nId != UNUSED_NODE) {
      const DeviceDenseNode n = nodes[nId];
      int colId = n.fidx;
      // printf("nid=%d colId=%d id=%d\n", nId, colId, id);
      int start = colOffsets[colId];
      int end = colOffsets[colId + 1];
      ///@todo: too much wasteful threads!!
      if ((id >= start) && (id < end) && !(n.IsLeaf() || n.IsUnused())) {
        node_id_t result = (2 * n.idx) + 1 + (vals[id] >= n.fvalue);
        nodeIdsPerInst[instId[id]] = result;
      }
    }
  }
}

__global__ void markLeavesKernel(DeviceDenseNode* nodes, int len) {
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
  dh::dvec2<float> vals;
  dh::dvec<float> vals_cached;
  /** corresponding instance id's of these featutre values */
  dh::dvec2<int> instIds;
  dh::dvec<int> instIds_cached;
  /** column offsets for these feature values */
  dh::dvec<int> colOffsets;
  dh::dvec<bst_gpair> gradsInst;
  dh::dvec2<node_id_t> nodeAssigns;
  dh::dvec2<int> nodeLocations;
  dh::dvec<DeviceDenseNode> nodes;
  dh::dvec<node_id_t> nodeAssignsPerInst;
  dh::dvec<bst_gpair> gradSums;
  dh::dvec<bst_gpair> gradScans;
  dh::dvec<Split> nodeSplits;
  int nVals;
  int nRows;
  int nCols;
  int maxNodes;
  int maxLeaves;
  dh::CubMemory tmp_mem;
  dh::dvec<bst_gpair> tmpScanGradBuff;
  dh::dvec<int> tmpScanKeyBuff;
  dh::dvec<int> colIds;
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;

 public:
  GPUMaker() : allocated(false) {}
  ~GPUMaker() {}

  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override {
    param.InitAllowUnknown(args);
    maxNodes = (1 << (param.max_depth + 1)) - 1;
    maxLeaves = 1 << param.max_depth;
  }

  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override {
    GradStats::CheckInfo(dmat->info());
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();

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
  void UpdateTree(const std::vector<bst_gpair>& gpair, DMatrix* hMat,
                  RegTree* hTree) {
    if (!allocated) {
      setupOneTimeData(*hMat);
    }
    for (int i = 0; i < param.max_depth; ++i) {
      if (i == 0) {
        // make sure to start on a fresh tree with sorted values!
        vals.current_dvec() = vals_cached;
        instIds.current_dvec() = instIds_cached;
        transferGrads(gpair);
      }
      int nNodes = 1 << i;
      node_id_t nodeStart = nNodes - 1;
      initNodeData(i, nodeStart, nNodes);
      findSplit(i, nodeStart, nNodes);
    }
    // mark all the used nodes with unused children as leaf nodes
    markLeaves();
    dense2sparse_tree(hTree, nodes, param);
  }

  void findSplit(int level, node_id_t nodeStart, int nNodes) {
    reduceScanByKey(gradSums.data(), gradScans.data(), gradsInst.data(),
                    instIds.current(), nodeAssigns.current(), nVals, nNodes,
                    nCols, tmpScanGradBuff.data(), tmpScanKeyBuff.data(),
                    colIds.data(), nodeStart);
    argMaxByKey(nodeSplits.data(), gradScans.data(), gradSums.data(),
                vals.current(), colIds.data(), nodeAssigns.current(),
                nodes.data(), nNodes, nodeStart, nVals, param,
                level <= MAX_ABK_LEVELS ? ABK_SMEM : ABK_GMEM);
    split2node(nodes.data(), nodeSplits.data(), gradScans.data(),
               gradSums.data(), vals.current(), colIds.data(),
               colOffsets.data(), nodeAssigns.current(), nNodes, nodeStart,
               nCols, param);
  }

  void allocateAllData(int offsetSize) {
    int tmpBuffSize = scanTempBufferSize(nVals);
    ba.allocate(dh::get_device_idx(param.gpu_id), param.silent, &vals, nVals,
                &vals_cached, nVals, &instIds, nVals, &instIds_cached, nVals,
                &colOffsets, offsetSize, &gradsInst, nRows, &nodeAssigns, nVals,
                &nodeLocations, nVals, &nodes, maxNodes, &nodeAssignsPerInst,
                nRows, &gradSums, maxLeaves * nCols, &gradScans, nVals,
                &nodeSplits, maxLeaves, &tmpScanGradBuff, tmpBuffSize,
                &tmpScanKeyBuff, tmpBuffSize, &colIds, nVals);
  }

  void setupOneTimeData(DMatrix& hMat) {
    size_t free_memory = dh::available_memory(dh::get_device_idx(param.gpu_id));
    if (!hMat.SingleColBlock()) {
      throw std::runtime_error("exact::GPUBuilder - must have 1 column block");
    }
    std::vector<float> fval;
    std::vector<int> fId, offset;
    convertToCsc(hMat, fval, fId, offset);
    allocateAllData((int)offset.size());
    transferAndSortData(fval, fId, offset);
    allocated = true;
  }

  void convertToCsc(DMatrix& hMat, std::vector<float>& fval,
                    std::vector<int>& fId, std::vector<int>& offset) {
    MetaInfo info = hMat.info();
    nRows = info.num_row;
    nCols = info.num_col;
    offset.reserve(nCols + 1);
    offset.push_back(0);
    fval.reserve(nCols * nRows);
    fId.reserve(nCols * nRows);
    // in case you end up with a DMatrix having no column access
    // then make sure to enable that before copying the data!
    if (!hMat.HaveColAccess()) {
      const std::vector<bool> enable(nCols, true);
      hMat.InitColAccess(enable, 1, nRows);
    }
    dmlc::DataIter<ColBatch>* iter = hMat.ColIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const ColBatch& batch = iter->Value();
      for (int i = 0; i < batch.size; i++) {
        const ColBatch::Inst& col = batch[i];
        for (const ColBatch::Entry* it = col.data; it != col.data + col.length;
             it++) {
          int inst_id = static_cast<int>(it->index);
          fval.push_back(it->fvalue);
          fId.push_back(inst_id);
        }
        offset.push_back(fval.size());
      }
    }
    nVals = fval.size();
  }

  void transferAndSortData(const std::vector<float>& fval,
                           const std::vector<int>& fId,
                           const std::vector<int>& offset) {
    vals.current_dvec() = fval;
    instIds.current_dvec() = fId;
    colOffsets = offset;
    segmentedSort<float, int>(&tmp_mem, &vals, &instIds, nVals, nCols,
                              colOffsets);
    vals_cached = vals.current_dvec();
    instIds_cached = instIds.current_dvec();
    assignColIds<<<nCols, 512>>>(colIds.data(), colOffsets.data());
  }

  void transferGrads(const std::vector<bst_gpair>& gpair) {
    // HACK
    dh::safe_cuda(cudaMemcpy(gradsInst.data(), &(gpair[0]),
                             sizeof(bst_gpair) * nRows,
                             cudaMemcpyHostToDevice));
    // evaluate the full-grad reduction for the root node
    sumReduction<bst_gpair>(tmp_mem, gradsInst, gradSums, nRows);
  }

  void initNodeData(int level, node_id_t nodeStart, int nNodes) {
    // all instances belong to root node at the beginning!
    if (level == 0) {
      nodes.fill(DeviceDenseNode());
      nodeAssigns.current_dvec().fill(0);
      nodeAssignsPerInst.fill(0);
      // for root node, just update the gradient/score/weight/id info
      // before splitting it! Currently all data is on GPU, hence this
      // stupid little kernel
      auto d_nodes = nodes.data();
      auto d_sums = gradSums.data();
      auto gpu_params = GPUTrainingParam(param);
      dh::launch_n(param.gpu_id, 1, [=]__device__(int idx)
      {
        d_nodes[0] = DeviceDenseNode(d_sums[0], 0, gpu_params);
      });
    } else {
      const int BlkDim = 256;
      const int ItemsPerThread = 4;
      // assign default node ids first
      int nBlks = dh::div_round_up(nRows, BlkDim);
      fillDefaultNodeIds<<<nBlks, BlkDim>>>(nodeAssignsPerInst.data(),
                                            nodes.data(), nRows);
      // evaluate the correct child indices of non-missing values next
      nBlks = dh::div_round_up(nVals, BlkDim * ItemsPerThread);
      assignNodeIds<<<nBlks, BlkDim>>>(
          nodeAssignsPerInst.data(), nodeLocations.current(),
          nodeAssigns.current(), instIds.current(), nodes.data(),
          colOffsets.data(), vals.current(), nVals, nCols);
      // gather the node assignments across all other columns too
      gather(dh::get_device_idx(param.gpu_id), nodeAssigns.current(),
             nodeAssignsPerInst.data(), instIds.current(), nVals);
      sortKeys(level);
    }
  }

  void sortKeys(int level) {
    // segmented-sort the arrays based on node-id's
    // but we don't need more than level+1 bits for sorting!
    segmentedSort(&tmp_mem, &nodeAssigns, &nodeLocations, nVals, nCols,
                  colOffsets, 0, level + 1);
    gather<float, int>(dh::get_device_idx(param.gpu_id), vals.other(),
                       vals.current(), instIds.other(), instIds.current(),
                       nodeLocations.current(), nVals);
    vals.buff().selector ^= 1;
    instIds.buff().selector ^= 1;
  }

  void markLeaves() {
    const int BlkDim = 128;
    int nBlks = dh::div_round_up(maxNodes, BlkDim);
    markLeavesKernel<<<nBlks, BlkDim>>>(nodes.data(), maxNodes);
  }
};

XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker(); });

}  // namespace tree
}  // namespace xgboost
