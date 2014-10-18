#ifndef XGBOOST_TREE_UPDATER_HISTMAKER_INL_HPP_
#define XGBOOST_TREE_UPDATER_HISTMAKER_INL_HPP_
/*!
 * \file updater_histmaker-inl.hpp
 * \brief use histogram counting to construct a tree
 * \author Tianqi Chen
 */
#include <vector>
#include <algorithm>

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
    std::vector<unsigned> cut;    
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
          hset[0][i].Add(hset[tid][i]);
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
 private:
  // create histogram for a setup histset
  inline void CreateHist(const std::vector<bst_gpair> &gpair,
                         IFMatrix *p_fmat,
                         const BoosterInfo &info,
                         unsigned num_feature) {
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
        int nid = position[ridx];
        if (nid >= 0) {
          for (bst_uint i = 0; i < inst.length; ++i) {
            utils::Assert(inst[i].index < num_feature, "feature index exceed bound");
            hset[inst[i].index + nid * num_feature]
                .Add(inst[i].fvalue, gpair, info, ridx);
          }
        }
      }
    }
    // accumulating statistics together
    wspace.Aggregate();
    // get the split solution    
  }  
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_HISTMAKER_INL_HPP_
