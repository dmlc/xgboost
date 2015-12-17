/*!
 * Copyright 2014 by Contributors
 * \file updater_basemaker-inl.hpp
 * \brief implement a common tree constructor
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_BASEMAKER_INL_HPP_
#define XGBOOST_TREE_UPDATER_BASEMAKER_INL_HPP_
#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include "../sync/sync.h"
#include "../utils/random.h"
#include "../utils/quantile.h"

namespace xgboost {
namespace tree {
/*!
 * \brief base tree maker class that defines common operation
 *  needed in tree making
 */
class BaseMaker: public IUpdater {
 public:
  // destructor
  virtual ~BaseMaker(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }

 protected:
  // helper to collect and query feature meta information
  struct FMetaHelper {
   public:
    /*! \brief find type of each feature, use column format */
    inline void InitByCol(IFMatrix *p_fmat,
                          const RegTree &tree) {
      fminmax.resize(tree.param.num_feature * 2);
      std::fill(fminmax.begin(), fminmax.end(),
                -std::numeric_limits<bst_float>::max());
      // start accumulating statistics
      utils::IIterator<ColBatch> *iter = p_fmat->ColIterator();
      iter->BeforeFirst();
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (bst_uint i = 0; i < batch.size; ++i) {
          const bst_uint fid = batch.col_index[i];
          const ColBatch::Inst &c = batch[i];
          if (c.length != 0) {
            fminmax[fid * 2 + 0] = std::max(-c[0].fvalue, fminmax[fid * 2 + 0]);
            fminmax[fid * 2 + 1] = std::max(c[c.length - 1].fvalue, fminmax[fid * 2 + 1]);
          }
        }
      }
      rabit::Allreduce<rabit::op::Max>(BeginPtr(fminmax), fminmax.size());
    }
    // get feature type, 0:empty 1:binary 2:real
    inline int Type(bst_uint fid) const {
      utils::Assert(fid * 2 + 1 < fminmax.size(),
                    "FeatHelper fid exceed query bound ");
      bst_float a = fminmax[fid * 2];
      bst_float b = fminmax[fid * 2 + 1];
      if (a == -std::numeric_limits<bst_float>::max()) return 0;
      if (-a == b) {
        return 1;
      } else {
        return 2;
      }
    }
    inline bst_float MaxValue(bst_uint fid) const {
      return fminmax[fid *2 + 1];
    }
    inline void SampleCol(float p, std::vector<bst_uint> *p_findex) const {
      std::vector<bst_uint> &findex = *p_findex;
      findex.clear();
      for (size_t i = 0; i < fminmax.size(); i += 2) {
        const bst_uint fid = static_cast<bst_uint>(i / 2);
        if (this->Type(fid) != 0) findex.push_back(fid);
      }
      unsigned n = static_cast<unsigned>(p * findex.size());
      random::Shuffle(findex);
      findex.resize(n);
      // sync the findex if it is subsample
      std::string s_cache;
      utils::MemoryBufferStream fc(&s_cache);
      utils::IStream &fs = fc;
      if (rabit::GetRank() == 0) {
        fs.Write(findex);
      }
      rabit::Broadcast(&s_cache, 0);
      fs.Read(&findex);
    }

   private:
    std::vector<bst_float> fminmax;
  };
  // ------static helper functions ------
  // helper function to get to next level of the tree
  /*! \brief this is  helper function for row based data*/
  inline static int NextLevel(const RowBatch::Inst &inst, const RegTree &tree, int nid) {
    const RegTree::Node &n = tree[nid];
    bst_uint findex = n.split_index();
    for (unsigned i = 0; i < inst.length; ++i) {
      if (findex == inst[i].index) {
        if (inst[i].fvalue < n.split_cond()) {
          return n.cleft();
        } else {
          return n.cright();
        }
      }
    }
    return n.cdefault();
  }
  /*! \brief get number of omp thread in current context */
  inline static int get_nthread(void) {
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    return nthread;
  }
  //  ------class member helpers---------
  /*! \brief initialize temp data structure */
  inline void InitData(const std::vector<bst_gpair> &gpair,
                       const IFMatrix &fmat,
                       const std::vector<unsigned> &root_index,
                       const RegTree &tree) {
    utils::Assert(tree.param.num_nodes == tree.param.num_roots,
                  "TreeMaker: can only grow new tree");
    {
      // setup position
      position.resize(gpair.size());
      if (root_index.size() == 0) {
        std::fill(position.begin(), position.end(), 0);
      } else {
        for (size_t i = 0; i < position.size(); ++i) {
          position[i] = root_index[i];
          utils::Assert(root_index[i] < (unsigned)tree.param.num_roots,
                        "root index exceed setting");
        }
      }
      // mark delete for the deleted datas
      for (size_t i = 0; i < position.size(); ++i) {
        if (gpair[i].hess < 0.0f) position[i] = ~position[i];
      }
      // mark subsample
      if (param.subsample < 1.0f) {
        for (size_t i = 0; i < position.size(); ++i) {
          if (gpair[i].hess < 0.0f) continue;
          if (random::SampleBinary(param.subsample) == 0) position[i] = ~position[i];
        }
      }
    }
    {
      // expand query
      qexpand.reserve(256); qexpand.clear();
      for (int i = 0; i < tree.param.num_roots; ++i) {
        qexpand.push_back(i);
      }
      this->UpdateNode2WorkIndex(tree);
    }
  }
  /*! \brief update queue expand add in new leaves */
  inline void UpdateQueueExpand(const RegTree &tree) {
    std::vector<int> newnodes;
    for (size_t i = 0; i < qexpand.size(); ++i) {
      const int nid = qexpand[i];
      if (!tree[nid].is_leaf()) {
        newnodes.push_back(tree[nid].cleft());
        newnodes.push_back(tree[nid].cright());
      }
    }
    // use new nodes for qexpand
    qexpand = newnodes;
    this->UpdateNode2WorkIndex(tree);
  }
  // return decoded position
  inline int DecodePosition(bst_uint ridx) const {
    const int pid = position[ridx];
    return pid < 0 ? ~pid : pid;
  }
  // encode the encoded position value for ridx
  inline void SetEncodePosition(bst_uint ridx, int nid) {
    if (position[ridx] < 0) {
      position[ridx] = ~nid;
    } else {
      position[ridx] = nid;
    }
  }
  /*!
   * \brief this is helper function uses column based data structure,
   *        reset the positions to the lastest one
   * \param nodes the set of nodes that contains the split to be used
   * \param p_fmat feature matrix needed for tree construction
   * \param tree the regression tree structure
   */
  inline void ResetPositionCol(const std::vector<int> &nodes,
                               IFMatrix *p_fmat, const RegTree &tree) {
    // set the positions in the nondefault
    this->SetNonDefaultPositionCol(nodes, p_fmat, tree);
    // set rest of instances to default position
    const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
    // set default direct nodes to default
    // for leaf nodes that are not fresh, mark then to ~nid,
    // so that they are ignored in future statistics collection
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());

    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const bst_uint ridx = rowset[i];
      const int nid = this->DecodePosition(ridx);
      if (tree[nid].is_leaf()) {
        // mark finish when it is not a fresh leaf
        if (tree[nid].cright() == -1) {
          position[ridx] = ~nid;
        }
        } else {
        // push to default branch
        if (tree[nid].default_left()) {
          this->SetEncodePosition(ridx, tree[nid].cleft());
        } else {
          this->SetEncodePosition(ridx, tree[nid].cright());
        }
      }
    }
  }
  /*!
   * \brief this is helper function uses column based data structure,
   *        update all positions into nondefault branch, if any, ignore the default branch
   * \param nodes the set of nodes that contains the split to be used
   * \param p_fmat feature matrix needed for tree construction
   * \param tree the regression tree structure
   */
  virtual void SetNonDefaultPositionCol(const std::vector<int> &nodes,
                                        IFMatrix *p_fmat, const RegTree &tree) {
    // step 1, classify the non-default data into right places
    std::vector<unsigned> fsplits;
    for (size_t i = 0; i < nodes.size(); ++i) {
      const int nid = nodes[i];
      if (!tree[nid].is_leaf()) {
        fsplits.push_back(tree[nid].split_index());
      }
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
          const int nid = this->DecodePosition(ridx);
          // go back to parent, correct those who are not default
          if (!tree[nid].is_leaf() && tree[nid].split_index() == fid) {
            if (fvalue < tree[nid].split_cond()) {
              this->SetEncodePosition(ridx, tree[nid].cleft());
            } else {
              this->SetEncodePosition(ridx, tree[nid].cright());
            }
          }
        }
      }
    }
  }
  /*! \brief helper function to get statistics from a tree */
  template<typename TStats>
  inline void GetNodeStats(const std::vector<bst_gpair> &gpair,
                           const IFMatrix &fmat,
                           const RegTree &tree,
                           const BoosterInfo &info,
                           std::vector< std::vector<TStats> > *p_thread_temp,
                           std::vector<TStats> *p_node_stats) {
    std::vector< std::vector<TStats> > &thread_temp = *p_thread_temp;
    thread_temp.resize(this->get_nthread());
    p_node_stats->resize(tree.param.num_nodes);
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      thread_temp[tid].resize(tree.param.num_nodes, TStats(param));
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const unsigned nid = qexpand[i];
        thread_temp[tid][nid].Clear();
      }
    }
    const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
    // setup position
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const bst_uint ridx = rowset[i];
      const int nid = position[ridx];
      const int tid = omp_get_thread_num();
      if (nid >= 0) {
        thread_temp[tid][nid].Add(gpair, info, ridx);
      }
    }
    // sum the per thread statistics together
    for (size_t j = 0; j < qexpand.size(); ++j) {
      const int nid = qexpand[j];
      TStats &s = (*p_node_stats)[nid];
      s.Clear();
      for (size_t tid = 0; tid < thread_temp.size(); ++tid) {
        s.Add(thread_temp[tid][nid]);
      }
    }
  }
  /*! \brief common helper data structure to build sketch */
  struct SketchEntry {
    /*! \brief total sum of amount to be met */
    double sum_total;
    /*! \brief statistics used in the sketch */
    double rmin, wmin;
    /*! \brief last seen feature value */
    bst_float last_fvalue;
    /*! \brief current size of sketch */
    double next_goal;
    // pointer to the sketch to put things in
    utils::WXQuantileSketch<bst_float, bst_float> *sketch;
    // initialize the space
    inline void Init(unsigned max_size) {
      next_goal = -1.0f;
      rmin = wmin = 0.0f;
      sketch->temp.Reserve(max_size + 1);
      sketch->temp.size = 0;
    }
    /*!
     * \brief push a new element to sketch
     * \param fvalue feature value, comes in sorted ascending order
     * \param w weight
     * \param max_size
     */
    inline void Push(bst_float fvalue, bst_float w, unsigned max_size) {
      if (next_goal == -1.0f) {
        next_goal = 0.0f;
        last_fvalue = fvalue;
        wmin = w;
        return;
      }
      if (last_fvalue != fvalue) {
        double rmax = rmin + wmin;
        if (rmax >= next_goal && sketch->temp.size != max_size) {
          if (sketch->temp.size == 0 ||
              last_fvalue > sketch->temp.data[sketch->temp.size-1].value) {
            // push to sketch
            sketch->temp.data[sketch->temp.size] =
                utils::WXQuantileSketch<bst_float, bst_float>::
                Entry(static_cast<bst_float>(rmin),
                      static_cast<bst_float>(rmax),
                      static_cast<bst_float>(wmin), last_fvalue);
            utils::Assert(sketch->temp.size < max_size,
                          "invalid maximum size max_size=%u, stemp.size=%lu\n",
                          max_size, sketch->temp.size);
            ++sketch->temp.size;
          }
          if (sketch->temp.size == max_size) {
            next_goal = sum_total * 2.0f + 1e-5f;
          } else {
            next_goal = static_cast<bst_float>(sketch->temp.size * sum_total / max_size);
          }
        } else {
          if (rmax >= next_goal) {
            rabit::TrackerPrintf("INFO: rmax=%g, sum_total=%g, next_goal=%g, size=%lu\n",
                                 rmax, sum_total, next_goal, sketch->temp.size);
          }
        }
        rmin = rmax;
        wmin = w;
        last_fvalue = fvalue;
      } else {
        wmin += w;
      }
    }
    /*! \brief push final unfinished value to the sketch */
    inline void Finalize(unsigned max_size) {
      double rmax = rmin + wmin;
      if (sketch->temp.size == 0 || last_fvalue > sketch->temp.data[sketch->temp.size-1].value) {
        utils::Assert(sketch->temp.size <= max_size,
                      "Finalize: invalid maximum size, max_size=%u, stemp.size=%lu",
                      sketch->temp.size, max_size);
        // push to sketch
        sketch->temp.data[sketch->temp.size] =
            utils::WXQuantileSketch<bst_float, bst_float>::
            Entry(static_cast<bst_float>(rmin),
                  static_cast<bst_float>(rmax),
                  static_cast<bst_float>(wmin), last_fvalue);
        ++sketch->temp.size;
      }
      sketch->PushTemp();
    }
  };
  /*! \brief training parameter of tree grower */
  TrainParam param;
  /*! \brief queue of nodes to be expanded */
  std::vector<int> qexpand;
  /*!
   * \brief map active node to is working index offset in qexpand,
   *   can be -1, which means the node is node actively expanding
   */
  std::vector<int> node2workindex;
  /*!
   * \brief position of each instance in the tree
   *   can be negative, which means this position is no longer expanding
   *   see also Decode/EncodePosition
   */
  std::vector<int> position;

 private:
  inline void UpdateNode2WorkIndex(const RegTree &tree) {
    // update the node2workindex
    std::fill(node2workindex.begin(), node2workindex.end(), -1);
    node2workindex.resize(tree.param.num_nodes);
    for (size_t i = 0; i < qexpand.size(); ++i) {
      node2workindex[qexpand[i]] = static_cast<int>(i);
    }
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_BASEMAKER_INL_HPP_
