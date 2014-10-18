#ifndef XGBOOST_TREE_UPDATER_HISTMAKER_INL_HPP_
#define XGBOOST_TREE_UPDATER_HISTMAKER_INL_HPP_
/*!
 * \file updater_histmaker-inl.hpp
 * \brief use histogram counting to construct a tree
 * \author Tianqi Chen
 */
#include <vector>
#include <algorithm>
#include "../sync/sync.h"

namespace xgboost {
namespace tree {
template<typename TStats>
class HistMaker: public IUpdater {
 public:
  virtual ~HistMaker(void) {}
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
      // TODO
    }
    param.learning_rate = lr;
  }

 protected:
  /*! \brief a single histogram */
  struct HistUnit {
    /*! \brief cutting point of histogram, contains maximum point */
    const bst_float *cut;
    /*! \brief content of statistics data */    
    TStats *data;
    /*! \brief size of histogram */
    const unsigned size;
    // constructor
    HistUnit(const bst_float *cut, TStats *data, unsigned size)
        : cut(cut), data(data), size(size) {}
    /*! \brief add a histogram to data */
    inline void Add(bst_float fv, 
                    const std::vector<bst_gpair> &gpair,
                    const BoosterInfo &info,
                    const bst_uint ridx) {
      unsigned i = std::lower_bound(cut, cut + size, fv) - cut;
      utils::Assert(i < size, "maximum value must be in cut");
      data[i].Add(gpair, info, ridx);
    }
  };
  /*! \brief a set of histograms from different index */
  struct HistSet {
    /*! \brief the index pointer of each histunit */
    const unsigned *rptr;
    /*! \brief cutting points in each histunit */
    const bst_float *cut;
    /*! \brief data in different hist unit */
    std::vector<TStats> data;
    /*! \brief */
    inline HistUnit operator[](bst_uint fid) {
      return HistUnit(cut + rptr[fid],
                      &data[0] + rptr[fid],
                      rptr[fid+1] - rptr[fid]);
    }
  };
  // thread workspace 
  struct ThreadWSpace {
    /*! \brief actual unit pointer */
    std::vector<unsigned> rptr;
    /*! \brief cut field */
    std::vector<bst_float> cut;
    // per thread histset
    std::vector<HistSet> hset;    
    // initialize the hist set
    inline void Init(const TrainParam &param) {
      int nthread;
      #pragma omp parallel
      {
        nthread = omp_get_num_threads();
      }
      hset.resize(nthread);
      // cleanup statistics
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        for (size_t i = 0; i < hset[tid].data.size(); ++i) {
          hset[tid].data[i].Clear();
        }
      }
      for (int i = 0; i < nthread; ++i) {
        hset[i].rptr = BeginPtr(rptr);
        hset[i].cut = BeginPtr(cut);
        hset[i].data.resize(cut.size(), TStats(param));        
      }
    }
    // aggregate all statistics to hset[0]
    inline void Aggregate(void) {
      bst_omp_uint nsize = static_cast<bst_omp_uint>(cut.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        for (size_t tid = 1; tid < hset.size(); ++tid) {
          hset[0].data[i].Add(hset[tid].data[i]);
        }
      }
    }
    /*! \brief clear the workspace */
    inline void Clear(void) {
      cut.clear(); rptr.resize(1); rptr[0] = 0;
    }
    /*! \brief total size */
    inline size_t Size(void) const {
      return rptr.size() - 1;
    }
  };
  // training parameter
  TrainParam param;
  // workspace of thread
  ThreadWSpace wspace;
  // position of each data
  std::vector<int> position;
  /*! \brief queue of nodes to be expanded */
  std::vector<int> qexpand;
  /*! \brief map active node to is working index offset in qexpand*/
  std::vector<int> node2workindex;
  // reducer for histogram
  sync::Reducer<TStats> histred;
 private:
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      RegTree *p_tree) {
    //this->InitData(gpair, *p_fmat, info.root_index, *p_tree);
    //this->InitNewNode(qexpand_, gpair, *p_fmat, info, *p_tree);
    for (int depth = 0; depth < param.max_depth; ++depth) {
      this->FindSplit(depth, gpair, p_fmat, info, p_tree);
      //this->ResetPosition(qexpand_, p_fmat, *p_tree);
      //this->UpdateQueueExpand(*p_tree, &qexpand_);
      //this->InitNewNode(qexpand_, gpair, *p_fmat, info, *p_tree);
      // if nothing left to be expand, break
      if (qexpand.size() == 0) break;
    }   
  }
  // create histogram for a setup histset
  inline void CreateHist(const std::vector<bst_gpair> &gpair,
                         IFMatrix *p_fmat,
                         const BoosterInfo &info,
                         const RegTree &tree) {
    bst_uint num_feature = tree.param.num_feature;
    // intialize work space
    wspace.Init(param);
    // start accumulating statistics
    utils::IIterator<RowBatch> *iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      utils::Check(batch.size < std::numeric_limits<unsigned>::max(),
                   "too large batch size ");
      const bst_omp_uint nbatch = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nbatch; ++i) {
        RowBatch::Inst inst = batch[i];
        const int tid = omp_get_thread_num();
        HistSet &hset = wspace.hset[tid];
        const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        const int nid = position[ridx];
        if (nid >= 0) {
          utils::Assert(tree[nid].is_leaf(), "CreateHist happens in leaf");
          const int wid = node2workindex[nid];
          for (bst_uint i = 0; i < inst.length; ++i) {
            utils::Assert(inst[i].index < num_feature, "feature index exceed bound");
            // feature histogram
            hset[inst[i].index + wid * (num_feature+1)]
                .Add(inst[i].fvalue, gpair, info, ridx);
          }
          // node histogram, use num_feature to borrow space
          hset[num_feature + wid * (num_feature + 1)]
              .data[0].Add(gpair, info, ridx);
        }
      }
    }
    // accumulating statistics together
    wspace.Aggregate();
    // sync the histogram
    histred.AllReduce(BeginPtr(wspace.hset[0].data), wspace.hset[0].data.size());
  }
  inline void EnumerateSplit(const HistUnit &hist, 
                             const TStats &node_sum,
                             bst_uint fid,
                             SplitEntry *best) {
    double root_gain = node_sum.CalcGain(param);
    TStats s(param), c(param);
    for (bst_uint i = 0; i < hist.size; ++i) {
      s.Add(hist.data[i]);
      if (s.sum_hess >= param.min_child_weight) {
        c.SetSubstract(node_sum, s);
        if (c.sum_hess >= param.min_child_weight) {
          double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
          best->Update(loss_chg, fid, hist.cut[i], false);
        }
      }
    }
    s.Clear();
    for (bst_uint i = hist.size - 1; i != 0; --i) {
      s.Add(hist.data[i]);
      if (s.sum_hess >= param.min_child_weight) {
        c.SetSubstract(node_sum, s);
        if (c.sum_hess >= param.min_child_weight) {
          double loss_chg = s.CalcGain(param) + c.CalcGain(param) - root_gain;
          best->Update(loss_chg, fid, hist.cut[i-1], true);
        }
      }
    }
  }
  inline void FindSplit(int depth,
                        const std::vector<bst_gpair> &gpair,
                        IFMatrix *p_fmat,
                        const BoosterInfo &info,
                        RegTree *p_tree) {
    const bst_uint num_feature = p_tree->param.num_feature;
    // create histogram
    this->CreateHist(gpair, p_fmat, info, *p_tree);
    // get the best split condition for each node
    std::vector<SplitEntry> sol(qexpand.size());
    bst_omp_uint nexpand = static_cast<bst_omp_uint>(qexpand.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (bst_omp_uint wid = 0; wid < nexpand; ++ wid) {
      const int nid = qexpand[wid];
      utils::Assert(node2workindex[nid] == static_cast<int>(wid), "node2workindex inconsistent");
      SplitEntry &best = sol[wid];     
      TStats &node_sum = wspace.hset[0][num_feature + wid * (num_feature + 1)].data[0];
      for (bst_uint fid = 0; fid < num_feature; ++ fid) {
        EnumerateSplit(wspace.hset[0][fid + wid * (num_feature+1)],
                       node_sum, fid, &best);
      }
    }    
    // get the best result, we can synchronize the solution
    for (bst_omp_uint wid = 0; wid < nexpand; ++ wid) {
      const int nid = qexpand[wid];
      const SplitEntry &best = sol[wid];
      const TStats &node_sum = wspace.hset[0][num_feature + wid * (num_feature + 1)].data[0];
      bst_float weight = node_sum.CalcWeight(param);
      // set up the values
      p_tree->stat(nid).loss_chg = best.loss_chg;
      p_tree->stat(nid).base_weight = weight;
      p_tree->stat(nid).sum_hess = static_cast<float>(node_sum.sum_hess);
      node_sum.SetLeafVec(param, p_tree->leafvec(nid));
      // now we know the solution in snode[nid], set split
      if (best.loss_chg > rt_eps) {
        p_tree->AddChilds(nid);
        (*p_tree)[nid].set_split(best.split_index(), best.split_value, best.default_left());
        // mark right child as 0, to indicate fresh leaf
        (*p_tree)[(*p_tree)[nid].cleft()].set_leaf(0.0f, 0);
        (*p_tree)[(*p_tree)[nid].cright()].set_leaf(0.0f, 0);
      } else {
        (*p_tree)[nid].set_leaf(weight * param.learning_rate);
      }
    }
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_HISTMAKER_INL_HPP_
