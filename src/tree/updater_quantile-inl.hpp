#ifndef XGBOOST_TREE_UPDATER_QUANTILE_INL_HPP_
#define XGBOOST_TREE_UPDATER_QUANTILE_INL_HPP_
/*!
 * \file updater_quantile_colmaker-inl.hpp
 * \brief use columnwise update to construct a quantile regression tree
 */
#include <vector>
#include <algorithm>
#include <iostream>
#include "./param.h"
#include "./updater.h"
#include "./updater_colmaker-inl.hpp"
#include "./updater_prune-inl.hpp"


namespace xgboost {
namespace tree {

  void SetQuantileGPair(const std::vector<bst_gpair> & gpair,std::vector<bst_gpair> & quantile_gpair) {

    for (unsigned i = 0; i < gpair.size(); i++) {
      if (gpair[i].grad > 0) {
	quantile_gpair[i] = bst_gpair(1.0,1.0);
      } else {
	quantile_gpair[i] = bst_gpair(-1.0,1.0);
      }
    }
}
/*! \brief colunwise update to construct a tree */

template<typename TStats>
class QuantileColMaker: public ColMaker<TStats> {
 public:
  virtual ~QuantileColMaker(void) {}

  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    //create new set of gradient pairs for quantile regression
    std::vector<bst_gpair> quantile_gpairs (gpair.size());
    SetQuantileGPair(gpair,quantile_gpairs);
    ColMaker<TStats>::Update(quantile_gpairs,p_fmat,info,trees);
  }

};


/*! \brief pruner that prunes a tree after growing finishs */
class QuantileTreePruner: public TreePruner {
 public:
  virtual ~QuantileTreePruner(void) {}

  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    // rescale learning rate according to size of trees
    std::vector<bst_gpair> quantile_gpairs (gpair.size());
    SetQuantileGPair(gpair,quantile_gpairs);
    TreePruner::Update(quantile_gpairs,p_fmat,info,trees);
  }

};


/*! \brief core statistics used for tree construction */
struct QuantileStats {
  /*! \brief sum gradient statistics */
  std::vector<double> gradients;
  double quantile;
  /*! \brief constructor, the object must be cleared during construction */
  QuantileStats() {
    this->Clear();
  }

  inline double CalculateLeafValue(double quantile)  {
    if (gradients.size() == 0) return 0;
    std::sort(gradients.begin(),gradients.end());
    unsigned index = (unsigned) (quantile * gradients.size());
    if (index >= gradients.size()) {
      index = gradients.size()-1;
    }
    return gradients[index];
  }


  /*! \brief clear the statistics */
  inline void Clear(void) {
    gradients.clear();
  }
  /*! \brief check if necessary information is ready */
  inline static void CheckInfo(const BoosterInfo &info) {
  }
  /*!
   * \brief accumulate statistics,
   * \param gpair the vector storing the gradient statistics
   * \param info the additional information 
   * \param ridx instance index of this instance
   */
  inline void Add(const std::vector<bst_gpair> &gpair,
                  const BoosterInfo &info,
                  bst_uint ridx) {
    const bst_gpair &b = gpair[ridx];
    this->Add(b.grad);
  }


  /*! \brief add statistics to the data */
  inline void Add(const QuantileStats &b) {
    for (unsigned i = 0; i < b.gradients.size(); i++) {
      this->Add(b.gradients[i]);
    }
  }
  /*! \brief same as add, reduce is used in All Reduce */
  inline static void Reduce(QuantileStats &a, const QuantileStats &b) {
    a.Add(b);
  }
  /*! \brief set current value to a - b */
  inline void SetSubstract(const QuantileStats &a, const QuantileStats &b) {
    //BUGBUG make this more efficient later
    std::vector<double> acopy = a.gradients;
    std::vector<double> bcopy = b.gradients;
    std::sort(acopy.begin(),acopy.end());
    std::sort(bcopy.begin(),bcopy.end());
    std::vector<double> new_gradients;
    unsigned aindex=0;
    unsigned bindex=0;
    while (aindex < acopy.size() && bindex < bcopy.size()) {
      if (acopy[aindex] == bcopy[bindex]) {
	aindex++;
	bindex++;
	continue;
      }
      if (acopy[aindex] < bcopy[bindex]) {
	new_gradients.push_back(acopy[aindex]);
	aindex++;
	continue;
      }
      if (acopy[aindex] > bcopy[bindex]) {
	bindex++;
      }
    }
    while (aindex < acopy.size()) {
      new_gradients.push_back(acopy[aindex]);
      aindex++;
    }
    gradients = new_gradients;
  }

  /*! \return whether the statistics is not used yet */
  inline bool Empty(void) const {
    return gradients.size() == 0.0;
  }


  /*! \brief add statistics to the data */
  inline void Add(double grad) {
    gradients.push_back(grad);
  }
};


class QuantileScorer: public IUpdater {
 public:
  virtual ~QuantileScorer(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    // sync-names                                                                                                                                                                                  
    if (!strcmp(name, "quantile")) quantile = static_cast<float>(atof(val));
    else {
      param.SetParam(name, val);
    }    
  }
  // update the tree, do pruning
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {        
    if (trees.size() == 0) return;
    // number of threads
    // thread temporal space
    std::vector< std::vector<QuantileStats> > stemp;
    std::vector<RegTree::FVec> fvec_temp;
    // setup temp space for each thread
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    fvec_temp.resize(nthread, RegTree::FVec());
    stemp.resize(nthread, std::vector<QuantileStats>());
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int num_nodes = 0;
      for (size_t i = 0; i < trees.size(); ++i) {
        num_nodes += trees[i]->param.num_nodes;
      }
      stemp[tid].resize(num_nodes, QuantileStats());
      std::fill(stemp[tid].begin(), stemp[tid].end(), QuantileStats());
      fvec_temp[tid].Init(trees[0]->param.num_feature);
    }
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery
#if __cplusplus >= 201103L
    auto lazy_get_stats = [&]()
#endif
    {
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
          const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
          RegTree::FVec &feats = fvec_temp[tid];
          feats.Fill(inst);
          int offset = 0;
          for (size_t j = 0; j < trees.size(); ++j) {
            AddStats(*trees[j], feats, gpair, info, ridx,
                     BeginPtr(stemp[tid]) + offset);
            offset += trees[j]->param.num_nodes;
          }
          feats.Drop(inst);
        }
      }
      // aggregate the statistics
      int num_nodes = static_cast<int>(stemp[0].size());
      #pragma omp parallel for schedule(static)
      for (int nid = 0; nid < num_nodes; ++nid) {
        for (int tid = 1; tid < nthread; ++tid) {
          stemp[0][nid].Add(stemp[tid][nid]);
        }
      }
    };
#if __cplusplus >= 201103L
    reducer.Allreduce(BeginPtr(stemp[0]), stemp[0].size(), lazy_get_stats);
#else
    reducer.Allreduce(BeginPtr(stemp[0]), stemp[0].size());
#endif
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    int offset = 0;
    for (size_t i = 0; i < trees.size(); ++i) {      
      for (int rid = 0; rid < trees[i]->param.num_roots; ++rid) {
        this->Refresh(BeginPtr(stemp[0]) + offset, rid, trees[i]);
      }
      offset += trees[i]->param.num_nodes;
    }
    // set learning rate back
    param.learning_rate = lr;
  }

 private:
  inline static void AddStats(const RegTree &tree,
                              const RegTree::FVec &feat,
                              const std::vector<bst_gpair> &gpair,
                              const BoosterInfo &info,
                              const bst_uint ridx,
                              QuantileStats *gstats) {
    // start from groups that belongs to current data
    int pid = static_cast<int>(info.GetRoot(ridx));
    // tranverse tree
    while (!tree[pid].is_leaf()) {
      unsigned split_index = tree[pid].split_index();
      pid = tree.GetNext(pid, feat.fvalue(split_index), feat.is_missing(split_index));
    }
    gstats[pid].Add(gpair, info, ridx);
  }
  inline void Refresh(QuantileStats *gstats,
                      int nid, RegTree *p_tree) {
    RegTree &tree = *p_tree;
    if (tree[nid].is_leaf()) {
	tree[nid].set_leaf(gstats[nid].CalculateLeafValue(quantile) * param.learning_rate);
    } else {
      this->Refresh(gstats, tree[nid].cleft(), p_tree);
      this->Refresh(gstats, tree[nid].cright(), p_tree);
    }
  }
  // training parameter
  TrainParam param;
  //quantile
  float quantile;
  // reducer
  rabit::Reducer<QuantileStats, QuantileStats::Reduce> reducer;  
};



}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_QUANTILE_INL_HPP_
