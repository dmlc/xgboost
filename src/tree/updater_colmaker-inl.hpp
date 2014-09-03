#ifndef XGBOOST_TREE_UPDATER_COLMAKER_INL_HPP_
#define XGBOOST_TREE_UPDATER_COLMAKER_INL_HPP_
/*!
 * \file updater_colmaker-inl.hpp
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <vector>
#include <algorithm>
#include "./param.h"
#include "./updater.h"
#include "../utils/omp.h"
#include "../utils/random.h"

namespace xgboost {
namespace tree {
/*! \brief pruner that prunes a tree after growing finishs */
template<typename TStats>
class ColMaker: public IUpdater {
 public:
  virtual ~ColMaker(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    TStats::CheckInfo(info);
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    // build tree
    for (size_t i = 0; i < trees.size(); ++i) {
      Builder builder(param);
      builder.Update(gpair, p_fmat, info, trees[i]);
    }
    param.learning_rate = lr;
  }

 private:
  // training parameter
  TrainParam param;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data*/
    TStats stats;
    /*! \brief last feature value scanned */
    float  last_fvalue;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit ThreadEntry(const TrainParam &param)
        : stats(param) {
    }
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    TStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    float weight;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit NodeEntry(const TrainParam &param)
        : stats(param), root_gain(0.0f), weight(0.0f){
    }
  };
  // actual builder that runs the algorithm
  struct Builder{
   public:
    // constructor
    explicit Builder(const TrainParam &param) : param(param) {}
    // update one tree, growing
    virtual void Update(const std::vector<bst_gpair> &gpair,
                        IFMatrix *p_fmat,
                        const BoosterInfo &info,
                        RegTree *p_tree) {
      this->InitData(gpair, *p_fmat, info.root_index, *p_tree);
      this->InitNewNode(qexpand_, gpair, *p_fmat, info, *p_tree);
      for (int depth = 0; depth < param.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, info, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, &qexpand_);
        this->InitNewNode(qexpand_, gpair, *p_fmat, info, *p_tree);
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (size_t i = 0; i < qexpand_.size(); ++i) {
        const int nid = qexpand_[i];
        (*p_tree)[nid].set_leaf(snode[nid].weight * param.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->stat(nid).loss_chg = snode[nid].best.loss_chg;
        p_tree->stat(nid).base_weight = snode[nid].weight;
        p_tree->stat(nid).sum_hess = static_cast<float>(snode[nid].stats.sum_hess);
        snode[nid].stats.SetLeafVec(param, p_tree->leafvec(nid));
      }
    }

   private:
    // initialize temp data structure
    inline void InitData(const std::vector<bst_gpair> &gpair,
                         const IFMatrix &fmat,
                         const std::vector<unsigned> &root_index, const RegTree &tree) {
      utils::Assert(tree.param.num_nodes == tree.param.num_roots, "ColMaker: can only grow new tree");
      const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
      {// setup position
        position.resize(gpair.size());
        if (root_index.size() == 0) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            position[rowset[i]] = 0;
          }
        } else {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            position[ridx] = root_index[ridx];
            utils::Assert(root_index[ridx] < (unsigned)tree.param.num_roots, "root index exceed setting");
          }
        }
        // mark delete for the deleted datas
        for (size_t i = 0; i < rowset.size(); ++i) {
          const bst_uint ridx = rowset[i];
          if (gpair[ridx].hess < 0.0f) position[ridx] = -1;
        }
        // mark subsample
        if (param.subsample < 1.0f) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            if (gpair[ridx].hess < 0.0f) continue;
            if (random::SampleBinary(param.subsample) == 0) position[ridx] = -1;
          }
        }
      }    
      {
        // initialize feature index
        unsigned ncol = static_cast<unsigned>(fmat.NumCol());
        for (unsigned i = 0; i < ncol; ++i) {
          if (fmat.GetColSize(i) != 0) {
            feat_index.push_back(i);
          }
        }
        unsigned n = static_cast<unsigned>(param.colsample_bytree * feat_index.size());
        random::Shuffle(feat_index);
        utils::Check(n > 0, "colsample_bytree is too small that no feature can be included");
        feat_index.resize(n);
      }
      {// setup temp space for each thread
        #pragma omp parallel
        {
          this->nthread = omp_get_num_threads();
        }
        // reserve a small space
        stemp.clear();
        stemp.resize(this->nthread, std::vector<ThreadEntry>());
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].clear(); stemp[i].reserve(256);
        }
        snode.reserve(256);
      }
      {// expand query
        qexpand_.reserve(256); qexpand_.clear();
        for (int i = 0; i < tree.param.num_roots; ++i) {
          qexpand_.push_back(i);
        }
      }
    }
    /*! \brief initialize the base_weight, root_gain, and NodeEntry for all the new nodes in qexpand */
    inline void InitNewNode(const std::vector<int> &qexpand,
                            const std::vector<bst_gpair> &gpair,
                            const IFMatrix &fmat,
                            const BoosterInfo &info,
                            const RegTree &tree) {
      {// setup statistics space for each tree node
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].resize(tree.param.num_nodes, ThreadEntry(param));
        }
        snode.resize(tree.param.num_nodes, NodeEntry(param));
      }
      const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
      // setup position
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int tid = omp_get_thread_num();
        if (position[ridx] < 0) continue;
        stemp[tid][position[ridx]].stats.Add(gpair, info, ridx);
      }
      // sum the per thread statistics together
      for (size_t j = 0; j < qexpand.size(); ++j) {
        const int nid = qexpand[j];
        TStats stats(param);
        for (size_t tid = 0; tid < stemp.size(); ++tid) {
          stats.Add(stemp[tid][nid].stats);
        }
        // update node statistics
        snode[nid].stats = stats;
        snode[nid].root_gain = static_cast<float>(stats.CalcGain(param));
        snode[nid].weight = static_cast<float>(stats.CalcWeight(param));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree &tree, std::vector<int> *p_qexpand) {
      std::vector<int> &qexpand = *p_qexpand;
      std::vector<int> newnodes;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[ nid ].is_leaf()) {
          newnodes.push_back(tree[nid].cleft());
          newnodes.push_back(tree[nid].cright());
        }
      }
      // use new nodes for qexpand
      qexpand = newnodes;
    }
    // enumerate the split values of specific feature
    inline void EnumerateSplit(const ColBatch::Entry *begin,
                               const ColBatch::Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<bst_gpair> &gpair,
                               const BoosterInfo &info,
                               std::vector<ThreadEntry> &temp) {
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (size_t j = 0; j < qexpand.size(); ++j) {
        temp[qexpand[j]].stats.Clear();
      }
      // left statistics
      TStats c(param);
      for(const ColBatch::Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position[ridx];
        if (nid < 0) continue;
        // start working
        const float fvalue = it->fvalue;
        // get the statistics of nid
        ThreadEntry &e = temp[nid];
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) {
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        } else {
          // try to find a split
          if (std::abs(fvalue - e.last_fvalue) > rt_2eps && e.stats.sum_hess >= param.min_child_weight) {
            c.SetSubstract(snode[nid].stats, e.stats);
            if (c.sum_hess >= param.min_child_weight) {
              bst_float loss_chg = static_cast<bst_float>(e.stats.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
              e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
            }
          }
          // update the statistics
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        }
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode[nid].stats, e.stats);
        if (e.stats.sum_hess >= param.min_child_weight && c.sum_hess >= param.min_child_weight) {
          bst_float loss_chg = static_cast<bst_float>(e.stats.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
          const float delta = d_step == +1 ? rt_eps : -rt_eps;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }
    // find splits at current level, do split per level
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<bst_gpair> &gpair,
                          IFMatrix *p_fmat,
                          const BoosterInfo &info,
                          RegTree *p_tree) {
      std::vector<bst_uint> feat_set = feat_index;
      if (param.colsample_bylevel != 1.0f) {
        random::Shuffle(feat_set);
        unsigned n = static_cast<unsigned>(param.colsample_bylevel * feat_index.size());
        utils::Check(n > 0, "colsample_bylevel is too small that no feature can be included");
        feat_set.resize(n);
      }
      utils::IIterator<ColBatch> *iter = p_fmat->ColIterator(feat_set);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        // start enumeration
        const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
        #if defined(_OPENMP)                                                                
        const int batch_size = std::max(static_cast<int>(nsize / this->nthread / 32), 1);
        #endif
        #pragma omp parallel for schedule(dynamic, batch_size)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          const bst_uint fid = batch.col_index[i];
          const int tid = omp_get_thread_num();
          const ColBatch::Inst c = batch[i];
          if (param.need_forward_search(p_fmat->GetColDensity(fid))) {            
            this->EnumerateSplit(c.data, c.data + c.length, +1, 
                                 fid, gpair, info, stemp[tid]);
          }
          if (param.need_backward_search(p_fmat->GetColDensity(fid))) {
            this->EnumerateSplit(c.data + c.length - 1, c.data - 1, -1, 
                                 fid, gpair, info, stemp[tid]);
          }
        }
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        NodeEntry &e = snode[nid];
        for (int tid = 0; tid < this->nthread; ++tid) {
          e.best.Update(stemp[tid][nid].best);
        }
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > rt_eps) {
          p_tree->AddChilds(nid);
          (*p_tree)[nid].set_split(e.best.split_index(), e.best.split_value, e.best.default_left());
        } else {
          (*p_tree)[nid].set_leaf(e.weight * param.learning_rate);
        }
      }
    }
    // reset position of each data points after split is created in the tree
    inline void ResetPosition(const std::vector<int> &qexpand, IFMatrix *p_fmat, const RegTree &tree) {
      const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
      // step 1, set default direct nodes to default, and leaf nodes to -1
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int nid = position[ridx];
        if (nid >= 0) {
          if (tree[nid].is_leaf()) {
            position[ridx] = -1;
          } else {
            // push to default branch, correct latter
            position[ridx] = tree[nid].default_left() ? tree[nid].cleft(): tree[nid].cright();
          }
        }
      }
      // step 2, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[nid].is_leaf()) fsplits.push_back(tree[nid].split_index());
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());

      utils::IIterator<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const float fvalue = col[j].fvalue;
            int nid = position[ridx];
            if (nid == -1) continue;
            // go back to parent, correct those who are not default
            nid = tree[nid].parent();
            if (tree[nid].split_index() == fid) {
              if (fvalue < tree[nid].split_cond()) {
                position[ridx] = tree[nid].cleft();
              } else {
                position[ridx] = tree[nid].cright();
              }
            }
          }
        }
      }
    }
    //--data fields--
    const TrainParam &param;
    // number of omp thread used during training
    int nthread;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
  };
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_COLMAKER_INL_HPP_
