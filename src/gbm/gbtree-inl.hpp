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
#include "../utils/omp.h"
#include "../tree/updater.h"

namespace xgboost {
namespace gbm {
/*!
 * \brief gradient boosted tree
 */
class GBTree : public IGradBooster {
 public:
  virtual ~GBTree(void) {
    this->Clear();
  }
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
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
  virtual void DoBoost(IFMatrix *p_fmat,
                       const BoosterInfo &info,
                       std::vector<bst_gpair> *in_gpair) {
    const std::vector<bst_gpair> &gpair = *in_gpair;
    if (mparam.num_output_group == 1) {
      this->BoostNewTrees(gpair, p_fmat, info, 0);
    } else {
      const int ngroup = mparam.num_output_group;
      utils::Check(gpair.size() % ngroup == 0,
                   "must have exactly ngroup*nrow gpairs");
      std::vector<bst_gpair> tmp(gpair.size()/ngroup);
      for (int gid = 0; gid < ngroup; ++gid) {
        bst_omp_uint nsize = static_cast<bst_omp_uint>(tmp.size());
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          tmp[i] = gpair[i * ngroup + gid];
        }
        this->BoostNewTrees(tmp, p_fmat, info, gid);
      }
    }
  }
  virtual void Predict(IFMatrix *p_fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit = 0) {
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    thread_temp.resize(nthread, tree::RegTree::FVec());
    for (int i = 0; i < nthread; ++i) {
      thread_temp[i].Init(mparam.num_feature);
    }

    std::vector<float> &preds = *out_preds;
    const size_t stride = info.num_row * mparam.num_output_group;
    preds.resize(stride * (mparam.size_leaf_vector+1));
    // start collecting the prediction
    utils::IIterator<RowBatch> *iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        tree::RegTree::FVec &feats = thread_temp[tid];
        int64_t ridx = static_cast<int64_t>(batch.base_rowid + i);
        utils::Assert(static_cast<size_t>(ridx) < info.num_row, "data row index exceed bound");
        // loop over output groups
        for (int gid = 0; gid < mparam.num_output_group; ++gid) {
          this->Pred(batch[i],
                     buffer_offset < 0 ? -1 : buffer_offset + ridx,
                     gid, info.GetRoot(ridx), &feats,
                     &preds[ridx * mparam.num_output_group + gid], stride, 
                     ntree_limit);
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
    char *pstr;
    pstr = std::strtok(&tval[0], ",");
    while (pstr != NULL) {
      updaters.push_back(tree::CreateUpdater(pstr));
      for (size_t j = 0; j < cfg.size(); ++j) {
        // set parameters
        updaters.back()->SetParam(cfg[j].first.c_str(), cfg[j].second.c_str());
      }
      pstr = std::strtok(NULL, ",");
    }
    tparam.updater_initialized = 1;
  }
  // do group specific group
  inline void BoostNewTrees(const std::vector<bst_gpair> &gpair,
                            IFMatrix *p_fmat,
                            const BoosterInfo &info,
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
      updaters[i]->Update(gpair, p_fmat, info, new_trees);
    }
    // push back to model
    for (size_t i = 0; i < new_trees.size(); ++i) {
      trees.push_back(new_trees[i]);
      tree_info.push_back(bst_group);
    }
    mparam.num_trees += tparam.num_parallel_tree;
  }
  // make a prediction for a single instance
  inline void Pred(const RowBatch::Inst &inst,
                   int64_t buffer_index,
                   int bst_group,
                   unsigned root_index,
                   tree::RegTree::FVec *p_feats,
                   float *out_pred, size_t stride, unsigned ntree_limit) {
    size_t itop = 0;
    float  psum = 0.0f;
    // sum of leaf vector 
    std::vector<float> vec_psum(mparam.size_leaf_vector, 0.0f);
    const int64_t bid = mparam.BufferOffset(buffer_index, bst_group);
    // number of valid trees
    unsigned treeleft = ntree_limit == 0 ? std::numeric_limits<unsigned>::max() : ntree_limit;
    // load buffered results if any
    if (bid >= 0 && ntree_limit == 0) {
      itop = pred_counter[bid];
      psum = pred_buffer[bid];
      for (int i = 0; i < mparam.size_leaf_vector; ++i) {
        vec_psum[i] = pred_buffer[bid + i + 1];
      }
    }
    if (itop != trees.size()) {
      p_feats->Fill(inst);
      for (size_t i = itop; i < trees.size(); ++i) {
        if (tree_info[i] == bst_group) {
          int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
          psum += (*trees[i])[tid].leaf_value();
          for (int j = 0; j < mparam.size_leaf_vector; ++j) {
            vec_psum[j] += trees[i]->leafvec(tid)[j];
          }
          if(--treeleft == 0) break;
        }
      }
      p_feats->Drop(inst);
    }
    // updated the buffered results
    if (bid >= 0 && ntree_limit == 0) {
      pred_counter[bid] = static_cast<unsigned>(trees.size());
      pred_buffer[bid] = psum;
      for (int i = 0; i < mparam.size_leaf_vector; ++i) {
        pred_buffer[bid + i + 1] = vec_psum[i];
      }
    }
    out_pred[0] = psum;
    for (int i = 0; i < mparam.size_leaf_vector; ++i) {
      out_pred[stride * (i + 1)] = vec_psum[i];
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
      using namespace std;
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
    /*! \brief size of leaf vector needed in tree */
    int size_leaf_vector;
    /*! \brief reserved parameters */
    int reserved[31];
    /*! \brief constructor */
    ModelParam(void) {
      num_trees = 0;
      num_roots = num_feature = 0;
      num_pbuffer = 0;
      num_output_group = 1;
      size_leaf_vector = 0;
      std::memset(reserved, 0, sizeof(reserved));
    }
    /*!
     * \brief set parameters from outside
     * \param name name of the parameter
     * \param val  value of the parameter
     */
    inline void SetParam(const char *name, const char *val) {
      using namespace std;
      if (!strcmp("num_pbuffer", name)) num_pbuffer = atol(val);
      if (!strcmp("num_output_group", name)) num_output_group = atol(val);
      if (!strcmp("bst:num_roots", name)) num_roots = atoi(val);
      if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
      if (!strcmp("bst:size_leaf_vector", name)) size_leaf_vector = atoi(val);
    }
    /*! \return size of prediction buffer actually needed */
    inline size_t PredBufferSize(void) const {
      return num_output_group * num_pbuffer * (size_leaf_vector + 1);
    }
    /*! 
     * \brief get the buffer offset given a buffer index and group id  
     * \return calculated buffer offset
     */
    inline int64_t BufferOffset(int64_t buffer_index, int bst_group) const {
      if (buffer_index < 0) return -1;
      utils::Check(buffer_index < num_pbuffer, "buffer_index exceed num_pbuffer");
      return (buffer_index + num_pbuffer * bst_group) * (size_leaf_vector + 1);
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
  std::vector<tree::RegTree::FVec> thread_temp;
  // the updaters that can be applied to each of tree
  std::vector<tree::IUpdater*> updaters;
};

}  // namespace gbm
}  // namespace xgboost
#endif  // XGBOOST_GBM_GBTREE_INL_HPP_
