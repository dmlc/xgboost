#ifndef XGBOOST_GBM_GBTREE_INL_HPP_
#define XGBOOST_GBM_GBTREE_INL_HPP_
/*!
 * \file gbtree-inl.hpp
 * \brief gradient boosted tree implementation
 * \author Tianqi Chen
 */
#include <vector>
#include <utility>
#include <string>
#include "./gbm.h"
#include "../tree/updater.h"

namespace xgboost {
namespace gbm {
/*!
 * \brief gradient boosted tree
 * \tparam FMatrix the data type updater taking
 */
template<typename FMatrix>
class GBTree : public IGradBooster<FMatrix> {
 public:
  virtual ~GBTree(void) {
    this->Clear();
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strncmp(name, "bst:", 4)) {
      cfg.push_back(std::make_pair(std::string(name+4), std::string(val)));
      // set into updaters, if already intialized
      for (size_t i = 0; i < updaters.size(); ++i) {
        updaters[i]->SetParam(name+4, val);
      }
    }
    if (!strcmp(name, "silent")) {
      this->SetParam("bst:silent", val);
    }
    tparam.SetParam(name, val);
    if (trees.size() == 0) mparam.SetParam(name, val);
  }
  virtual void LoadModel(utils::IStream &fi) {
    this->Clear();
    utils::Check(fi.Read(&mparam, sizeof(ModelParam)) != 0,
                 "GBTree: invalid model file");
    trees.resize(mparam.num_trees);
    for (size_t i = 0; i < trees.size(); ++i) {
      trees[i] = new tree::RegTree();
      trees[i]->LoadModel(fi);
    }
    tree_info.resize(mparam.num_trees);
    if (mparam.num_trees != 0) {
      utils::Check(fi.Read(&tree_info[0], sizeof(int) * mparam.num_trees) != 0,
                   "GBTree: invalid model file");
    }
    if (mparam.num_pbuffer != 0) {
      pred_buffer.resize(mparam.PredBufferSize());
      pred_counter.resize(mparam.PredBufferSize());
      utils::Check(fi.Read(&pred_buffer[0], pred_buffer.size() * sizeof(float)) != 0,
                   "GBTree: invalid model file");
      utils::Check(fi.Read(&pred_counter[0], pred_counter.size() * sizeof(unsigned)) != 0,
                   "GBTree: invalid model file");
    }
  }
  virtual void SaveModel(utils::IStream &fo) const {
    utils::Assert(mparam.num_trees == static_cast<int>(trees.size()), "GBTree");
    fo.Write(&mparam, sizeof(ModelParam));
    for (size_t i = 0; i < trees.size(); ++i) {
      trees[i]->SaveModel(fo);
    }
    if (tree_info.size() != 0) {
      fo.Write(&tree_info[0], sizeof(int) * tree_info.size());
    }
    if (mparam.num_pbuffer != 0) {
      fo.Write(&pred_buffer[0], pred_buffer.size() * sizeof(float));
      fo.Write(&pred_counter[0], pred_counter.size() * sizeof(unsigned));
    }
  }
  // initialize the predic buffer
  virtual void InitModel(void) {
    pred_buffer.clear(); pred_counter.clear();
    pred_buffer.resize(mparam.PredBufferSize(), 0.0f);
    pred_counter.resize(mparam.PredBufferSize(), 0);
    utils::Assert(mparam.num_trees == 0, "GBTree: model already initialized");
    utils::Assert(trees.size() == 0, "GBTree: model already initialized");
  }
  virtual void DoBoost(const std::vector<bst_gpair> &gpair,
                       const FMatrix &fmat,
                       const std::vector<unsigned> &root_index) {
    if (mparam.num_output_group == 1) {
      this->BoostNewTrees(gpair, fmat, root_index, 0);
    } else {
      const int ngroup = mparam.num_output_group;
      utils::Check(gpair.size() % ngroup == 0,
                   "must have exactly ngroup*nrow gpairs");
      std::vector<bst_gpair> tmp(gpair.size()/ngroup);
      for (int gid = 0; gid < ngroup; ++gid) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tmp.size(); ++i) {
          tmp[i] = gpair[i * ngroup + gid];
        }
        this->BoostNewTrees(tmp, fmat, root_index, gid);
      }
    }
  }
  virtual void Predict(const FMatrix &fmat,
                       int64_t buffer_offset,
                       const std::vector<unsigned> &root_index,
                       std::vector<float> *out_preds) {
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    this->InitThreadTemp(nthread);
    std::vector<float> &preds = *out_preds;
    preds.resize(0);
    // start collecting the prediction
    utils::IIterator<SparseBatch> *iter = fmat.RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const SparseBatch &batch = iter->Value();
      utils::Assert(batch.base_rowid * mparam.num_output_group == preds.size(),
                    "base_rowid is not set correctly");
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      preds.resize(preds.size() + batch.size * mparam.num_output_group);
      // parallel over local batch
      const unsigned nsize = static_cast<unsigned>(batch.size);
      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        std::vector<float> &feats = thread_temp[tid];
        const size_t ridx = batch.base_rowid + i;
        const unsigned root_idx = root_index.size() == 0 ? 0 : root_index[ridx];
        // loop over output groups
        for (int gid = 0; gid < mparam.num_output_group; ++gid) {
          preds[ridx * mparam.num_output_group + gid] =
              this->Pred(batch[i],
                         buffer_offset < 0 ? -1 : buffer_offset+ridx,
                         gid, root_idx, &feats);
        }
      }
    }
  }
  virtual std::vector<std::string> DumpModel(const utils::FeatMap& fmap, int option) {
    std::vector<std::string> dump;
    for (size_t i = 0; i < trees.size(); i++) {
      dump.push_back(trees[i]->DumpModel(fmap, option&1));
    }
    return dump;
  }

 protected:
  // clear the model
  inline void Clear(void) {
    for (size_t i = 0; i < trees.size(); ++i) {
      delete trees[i];
    }
    trees.clear();
    pred_buffer.clear();
    pred_counter.clear();
  }
  // initialize updater before using them
  inline void InitUpdater(void) {
    if (tparam.updater_initialized != 0) return;
    for (size_t i = 0; i < updaters.size(); ++i) {
      delete updaters[i];
    }
    updaters.clear();
    std::string tval = tparam.updater_seq;
    char *saveptr, *pstr;
    pstr = strtok_r(&tval[0], ",", &saveptr);
    while (pstr != NULL) {
      updaters.push_back(tree::CreateUpdater<FMatrix>(pstr));
      for (size_t j = 0; j < cfg.size(); ++j) {
        // set parameters
        updaters.back()->SetParam(cfg[j].first.c_str(), cfg[j].second.c_str());
      }
      pstr = strtok_r(NULL, ",", &saveptr);
    }
    tparam.updater_initialized = 1;
  }
  // do group specific group
  inline void BoostNewTrees(const std::vector<bst_gpair> &gpair,
                            const FMatrix &fmat,
                            const std::vector<unsigned> &root_index,
                            int bst_group) {
    this->InitUpdater();
    // create the trees
    std::vector<tree::RegTree *> new_trees;
    for (int i = 0; i < tparam.num_parallel_tree; ++i) {
      new_trees.push_back(new tree::RegTree());
      for (size_t j = 0; j < cfg.size(); ++j) {
        new_trees.back()->param.SetParam(cfg[j].first.c_str(), cfg[j].second.c_str());
      }
      new_trees.back()->InitModel();
    }
    // update the trees
    for (size_t i = 0; i < updaters.size(); ++i) {
      updaters[i]->Update(gpair, fmat, root_index, new_trees);
    }
    // push back to model
    for (size_t i = 0; i < new_trees.size(); ++i) {
      trees.push_back(new_trees[i]);
      tree_info.push_back(bst_group);
    }
    mparam.num_trees += tparam.num_parallel_tree;
  }
  // make a prediction for a single instance
  inline float Pred(const SparseBatch::Inst &inst,
                    int64_t buffer_index,
                    int bst_group,
                    unsigned root_index,
                    std::vector<float> *p_feats) {
    size_t itop = 0;
    float  psum = 0.0f;
    const int bid = mparam.BufferOffset(buffer_index, bst_group);
    // load buffered results if any
    if (bid >= 0) {
      itop = pred_counter[bid];
      psum = pred_buffer[bid];
    }
    if (itop != trees.size()) {
      FillThreadTemp(inst, p_feats);
      for (size_t i = itop; i < trees.size(); ++i) {
        if (tree_info[i] == bst_group) {
          psum += trees[i]->Predict(*p_feats, root_index);
        }
      }
      DropThreadTemp(inst, p_feats);
    }
    // updated the buffered results
    if (bid >= 0) {
      pred_counter[bid] = static_cast<unsigned>(trees.size());
      pred_buffer[bid] = psum;
    }
    return psum;
  }
  // initialize thread local space for prediction
  inline void InitThreadTemp(int nthread) {
    thread_temp.resize(nthread);
    for (size_t i = 0; i < thread_temp.size(); ++i) {
      thread_temp[i].resize(mparam.num_feature);
      std::fill(thread_temp[i].begin(), thread_temp[i].end(), NAN);
    }
  }
  // fill in a thread local dense vector using a sparse instance
  inline static void FillThreadTemp(const SparseBatch::Inst &inst,
                                    std::vector<float> *p_feats) {
    std::vector<float> &feats = *p_feats;
    for (bst_uint i = 0; i < inst.length; ++i) {
      feats[inst[i].findex] = inst[i].fvalue;
    }
  }
  // clear up a thread local dense vector
  inline static void DropThreadTemp(const SparseBatch::Inst &inst,
                                    std::vector<float> *p_feats) {
    std::vector<float> &feats = *p_feats;
    for (bst_uint i = 0; i < inst.length; ++i) {
      feats[inst[i].findex] = NAN;
    }
  }
  // --- data structure ---
  /*! \brief training parameters */
  struct TrainParam {
    /*! \brief number of threads */
    int nthread;
    /*!
     * \brief number of parallel trees constructed each iteration
     *  use this option to support boosted random forest
     */
    int num_parallel_tree;
    /*! \brief whether updater is already initialized */
    int updater_initialized;
    /*! \brief tree updater sequence */
    std::string updater_seq;
    // construction
    TrainParam(void) {
      nthread = 0;
      updater_seq = "grow_colmaker,prune";
      num_parallel_tree = 1;
      updater_initialized = 0;
    }
    inline void SetParam(const char *name, const char *val){
      if (!strcmp(name, "updater") &&
          strcmp(updater_seq.c_str(), val) != 0) {
        updater_seq = val;
        updater_initialized = 0;
      }
      if (!strcmp(name, "nthread")) {
        omp_set_num_threads(nthread = atoi(val));
      }
      if (!strcmp(name, "num_parallel_tree")) {
        num_parallel_tree = atoi(val);
      }
    }
  };
  /*! \brief model parameters */
  struct ModelParam {
    /*! \brief number of trees */
    int num_trees;
    /*! \brief number of root: default 0, means single tree */
    int num_roots;
    /*! \brief number of features to be used by trees */
    int num_feature;
    /*! \brief size of predicton buffer allocated used for buffering */
    int64_t num_pbuffer;
    /*! 
     * \brief how many output group a single instance can produce
     *  this affects the behavior of number of output we have:
     *    suppose we have n instance and k group, output will be k*n 
     */
    int num_output_group;
    /*! \brief reserved parameters */
    int reserved[32];
    /*! \brief constructor */
    ModelParam(void) {
      num_trees = 0;
      num_roots = num_feature = 0;
      num_pbuffer = 0;
      num_output_group = 1;
      memset(reserved, 0, sizeof(reserved));
    }
    /*!
     * \brief set parameters from outside
     * \param name name of the parameter
     * \param val  value of the parameter
     */
    inline void SetParam(const char *name, const char *val) {
      if (!strcmp("num_pbuffer", name)) num_pbuffer = atol(val);
      if (!strcmp("num_output_group", name)) num_output_group = atol(val);
      if (!strcmp("bst:num_roots", name)) num_roots = atoi(val);
      if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
    }
    /*! \return size of prediction buffer actually needed */
    inline size_t PredBufferSize(void) const {
      return num_output_group * num_pbuffer;
    }
    /*! 
     * \brief get the buffer offset given a buffer index and group id  
     * \return calculated buffer offset
     */
    inline size_t BufferOffset(int64_t buffer_index, int bst_group) const {
      if (buffer_index < 0) return -1;
      utils::Check(buffer_index < num_pbuffer, "buffer_index exceed num_pbuffer");
      return buffer_index + num_pbuffer * bst_group;
    }
  };
  // training parameter
  TrainParam tparam;
  // model parameter
  ModelParam mparam;
  /*! \brief vector of trees stored in the model */
  std::vector<tree::RegTree*> trees;
  /*! \brief some information indicator of the tree, reserved */
  std::vector<int> tree_info;
  /*! \brief prediction buffer */
  std::vector<float>  pred_buffer;
  /*! \brief prediction buffer counter, remember the prediction */
  std::vector<unsigned> pred_counter;
  // ----training fields----
  // configurations for tree
  std::vector< std::pair<std::string, std::string> > cfg;
  // temporal storage for per thread
  std::vector< std::vector<float> > thread_temp;
  // the updaters that can be applied to each of tree
  std::vector< tree::IUpdater<FMatrix>* > updaters;
};

}  // namespace gbm
}  // namespace xgboost
#endif  // XGBOOST_GBM_GBTREE_INL_HPP_
