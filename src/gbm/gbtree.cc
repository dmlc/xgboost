/*!
 * Copyright 2014 by Contributors
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <xgboost/gbm.h>
#include <xgboost/tree_updater.h>

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include "../common/common.h"

#include "../common/random.h"

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gbtree);

// boosting process types
enum TreeProcessType {
  kDefault,
  kUpdate
};

/*! \brief training parameters */
struct GBTreeTrainParam : public dmlc::Parameter<GBTreeTrainParam> {
  /*!
   * \brief number of parallel trees constructed each iteration
   *  use this option to support boosted random forest
   */
  int num_parallel_tree;
  /*! \brief tree updater sequence */
  std::string updater_seq;
  /*! \brief type of boosting process to run */
  int process_type;
  // flag to print out detailed breakdown of runtime
  int debug_verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GBTreeTrainParam) {
    DMLC_DECLARE_FIELD(num_parallel_tree)
        .set_default(1)
        .set_lower_bound(1)
        .describe("Number of parallel trees constructed during each iteration."\
                  " This option is used to support boosted random forest");
    DMLC_DECLARE_FIELD(updater_seq)
        .set_default("grow_colmaker,prune")
        .describe("Tree updater sequence.");
    DMLC_DECLARE_FIELD(process_type)
        .set_default(kDefault)
        .add_enum("default", kDefault)
        .add_enum("update", kUpdate)
        .describe("Whether to run the normal boosting process that creates new trees,"\
                  " or to update the trees in an existing model.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    // add alias
    DMLC_DECLARE_ALIAS(updater_seq, updater);
  }
};

/*! \brief training parameters */
struct DartTrainParam : public dmlc::Parameter<DartTrainParam> {
  /*! \brief whether to not print info during training */
  bool silent;
  /*! \brief type of sampling algorithm */
  int sample_type;
  /*! \brief type of normalization algorithm */
  int normalize_type;
  /*! \brief fraction of trees to drop during the dropout */
  float rate_drop;
  /*! \brief whether at least one tree should always be dropped during the dropout */
  bool one_drop;
  /*! \brief probability of skipping the dropout during an iteration */
  float skip_drop;
  /*! \brief learning step size for a time */
  float learning_rate;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DartTrainParam) {
    DMLC_DECLARE_FIELD(silent)
        .set_default(false)
        .describe("Not print information during training.");
    DMLC_DECLARE_FIELD(sample_type)
        .set_default(0)
        .add_enum("uniform", 0)
        .add_enum("weighted", 1)
        .describe("Different types of sampling algorithm.");
    DMLC_DECLARE_FIELD(normalize_type)
        .set_default(0)
        .add_enum("tree", 0)
        .add_enum("forest", 1)
        .describe("Different types of normalization algorithm.");
    DMLC_DECLARE_FIELD(rate_drop)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe("Fraction of trees to drop during the dropout.");
    DMLC_DECLARE_FIELD(one_drop)
        .set_default(false)
        .describe("Whether at least one tree should always be dropped during the dropout.");
    DMLC_DECLARE_FIELD(skip_drop)
        .set_range(0.0f, 1.0f)
        .set_default(0.0f)
        .describe("Probability of skipping the dropout during a boosting iteration.");
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.3f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }
};

/*! \brief model parameters */
struct GBTreeModelParam : public dmlc::Parameter<GBTreeModelParam> {
  /*! \brief number of trees */
  int num_trees;
  /*! \brief number of roots */
  int num_roots;
  /*! \brief number of features to be used by trees */
  int num_feature;
  /*! \brief pad this space, for backward compatibility reason.*/
  int pad_32bit;
  /*! \brief deprecated padding space. */
  int64_t num_pbuffer_deprecated;
  /*!
   * \brief how many output group a single instance can produce
   *  this affects the behavior of number of output we have:
   *    suppose we have n instance and k group, output will be k * n
   */
  int num_output_group;
  /*! \brief size of leaf vector needed in tree */
  int size_leaf_vector;
  /*! \brief reserved parameters */
  int reserved[32];
  /*! \brief constructor */
  GBTreeModelParam() {
    std::memset(this, 0, sizeof(GBTreeModelParam));
    static_assert(sizeof(GBTreeModelParam) == (4 + 2 + 2 + 32) * sizeof(int),
                  "64/32 bit compatibility issue");
  }
  // declare parameters, only declare those that need to be set.
  DMLC_DECLARE_PARAMETER(GBTreeModelParam) {
    DMLC_DECLARE_FIELD(num_output_group).set_lower_bound(1).set_default(1)
        .describe("Number of output groups to be predicted,"\
                  " used for multi-class classification.");
    DMLC_DECLARE_FIELD(num_roots).set_lower_bound(1).set_default(1)
        .describe("Tree updater sequence.");
    DMLC_DECLARE_FIELD(num_feature).set_lower_bound(0)
        .describe("Number of features used for training and prediction.");
    DMLC_DECLARE_FIELD(size_leaf_vector).set_lower_bound(0).set_default(0)
        .describe("Reserved option for vector tree.");
  }
};

// cache entry
struct CacheEntry {
  std::shared_ptr<DMatrix> data;
  std::vector<bst_float> predictions;
};

// gradient boosted trees
class GBTree : public GradientBooster {
 public:
  explicit GBTree(bst_float base_margin) : base_margin_(base_margin) {}

  void InitCache(const std::vector<std::shared_ptr<DMatrix> > &cache) {
    for (const std::shared_ptr<DMatrix>& d : cache) {
      CacheEntry e;
      e.data = d;
      cache_[d.get()] = std::move(e);
    }
  }

  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    this->cfg = cfg;
    // initialize model parameters if not yet been initialized.
    if (trees.size() == 0) {
      mparam.InitAllowUnknown(cfg);
    }
    // initialize the updaters only when needed.
    std::string updater_seq = tparam.updater_seq;
    tparam.InitAllowUnknown(cfg);
    if (updater_seq != tparam.updater_seq) updaters.clear();
    for (const auto& up : updaters) {
      up->Init(cfg);
    }
    // for the 'update' process_type, move trees into trees_to_update
    if (tparam.process_type == kUpdate && trees_to_update.size() == 0u) {
      for (size_t i = 0; i < trees.size(); ++i) {
        trees_to_update.push_back(std::move(trees[i]));
      }
      trees.clear();
      mparam.num_trees = 0;
    }
  }

  void Load(dmlc::Stream* fi) override {
    CHECK_EQ(fi->Read(&mparam, sizeof(mparam)), sizeof(mparam))
        << "GBTree: invalid model file";
    trees.clear();
    trees_to_update.clear();
    for (int i = 0; i < mparam.num_trees; ++i) {
      std::unique_ptr<RegTree> ptr(new RegTree());
      ptr->Load(fi);
      trees.push_back(std::move(ptr));
    }
    tree_info.resize(mparam.num_trees);
    if (mparam.num_trees != 0) {
      CHECK_EQ(fi->Read(dmlc::BeginPtr(tree_info), sizeof(int) * mparam.num_trees),
               sizeof(int) * mparam.num_trees);
    }
    this->cfg.clear();
    this->cfg.push_back(std::make_pair(std::string("num_feature"),
                                       common::ToString(mparam.num_feature)));
  }

  void Save(dmlc::Stream* fo) const override {
    CHECK_EQ(mparam.num_trees, static_cast<int>(trees.size()));
    fo->Write(&mparam, sizeof(mparam));
    for (size_t i = 0; i < trees.size(); ++i) {
      trees[i]->Save(fo);
    }
    if (tree_info.size() != 0) {
      fo->Write(dmlc::BeginPtr(tree_info), sizeof(int) * tree_info.size());
    }
  }

  bool AllowLazyCheckPoint() const override {
    return mparam.num_output_group == 1 ||
        tparam.updater_seq.find("distcol") != std::string::npos;
  }

  void DoBoost(DMatrix* p_fmat,
               std::vector<bst_gpair>* in_gpair,
               ObjFunction* obj) override {
    const std::vector<bst_gpair>& gpair = *in_gpair;
    std::vector<std::vector<std::unique_ptr<RegTree> > > new_trees;
    const int ngroup = mparam.num_output_group;
    if (ngroup == 1) {
      std::vector<std::unique_ptr<RegTree> > ret;
      BoostNewTrees(gpair, p_fmat, 0, &ret);
      new_trees.push_back(std::move(ret));
    } else {
      CHECK_EQ(gpair.size() % ngroup, 0U)
          << "must have exactly ngroup*nrow gpairs";
      std::vector<bst_gpair> tmp(gpair.size() / ngroup);
      for (int gid = 0; gid < ngroup; ++gid) {
        bst_omp_uint nsize = static_cast<bst_omp_uint>(tmp.size());
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          tmp[i] = gpair[i * ngroup + gid];
        }
        std::vector<std::unique_ptr<RegTree> > ret;
        BoostNewTrees(tmp, p_fmat, gid, &ret);
        new_trees.push_back(std::move(ret));
      }
    }
    double tstart = dmlc::GetTime();
    for (int gid = 0; gid < ngroup; ++gid) {
      this->CommitModel(std::move(new_trees[gid]), gid);
    }
    if (tparam.debug_verbose > 0) {
      LOG(INFO) << "CommitModel(): " << dmlc::GetTime() - tstart << " sec";
    }
  }

  void Predict(DMatrix* p_fmat,
               std::vector<bst_float>* out_preds,
               unsigned ntree_limit) override {
    if (ntree_limit == 0 ||
        ntree_limit * mparam.num_output_group >= trees.size()) {
      auto it = cache_.find(p_fmat);
      if (it != cache_.end()) {
        std::vector<bst_float>& y = it->second.predictions;
        if (y.size() != 0) {
          out_preds->resize(y.size());
          std::copy(y.begin(), y.end(), out_preds->begin());
          return;
        }
      }
    }
    PredLoopInternal<GBTree>(p_fmat, out_preds, 0, ntree_limit, true);
  }

  void Predict(const SparseBatch::Inst& inst,
               std::vector<bst_float>* out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    if (thread_temp.size() == 0) {
      thread_temp.resize(1, RegTree::FVec());
      thread_temp[0].Init(mparam.num_feature);
    }
    ntree_limit *= mparam.num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    out_preds->resize(mparam.num_output_group * (mparam.size_leaf_vector+1));
    // loop over output groups
    for (int gid = 0; gid < mparam.num_output_group; ++gid) {
      (*out_preds)[gid] =
          PredValue(inst, gid, root_index,
                    &thread_temp[0], 0, ntree_limit) + base_margin_;
    }
  }

  void PredictLeaf(DMatrix* p_fmat,
                   std::vector<bst_float>* out_preds,
                   unsigned ntree_limit) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread);
    this->PredPath(p_fmat, out_preds, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread);
    this->PredContrib(p_fmat, out_contribs, ntree_limit);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    std::vector<std::string> dump;
    for (size_t i = 0; i < trees.size(); i++) {
      dump.push_back(trees[i]->DumpModel(fmap, with_stats, format));
    }
    return dump;
  }

 protected:
  // internal prediction loop
  // add predictions to out_preds
  template<typename Derived>
  inline void PredLoopInternal(
      DMatrix* p_fmat,
      std::vector<bst_float>* out_preds,
      unsigned tree_begin,
      unsigned ntree_limit,
      bool init_out_preds) {
    int num_group = mparam.num_output_group;
    ntree_limit *= num_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }

    if (init_out_preds) {
      size_t n = num_group * p_fmat->info().num_row;
      const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
      out_preds->resize(n);
      if (base_margin.size() != 0) {
        CHECK_EQ(out_preds->size(), n);
        std::copy(base_margin.begin(), base_margin.end(), out_preds->begin());
      } else {
        std::fill(out_preds->begin(), out_preds->end(), base_margin_);
      }
    }

    if (num_group == 1) {
      PredLoopSpecalize<Derived>(p_fmat, out_preds, 1,
                                 tree_begin, ntree_limit);
    } else {
      PredLoopSpecalize<Derived>(p_fmat, out_preds, num_group,
                                 tree_begin, ntree_limit);
    }
  }

  template<typename Derived>
  inline void PredLoopSpecalize(
      DMatrix* p_fmat,
      std::vector<bst_float>* out_preds,
      int num_group,
      unsigned tree_begin,
      unsigned tree_end) {
    const MetaInfo& info = p_fmat->info();
    const int nthread = omp_get_max_threads();
    CHECK_EQ(num_group, mparam.num_output_group);
    InitThreadTemp(nthread);
    std::vector<bst_float>& preds = *out_preds;
    CHECK_EQ(mparam.size_leaf_vector, 0)
        << "size_leaf_vector is enforced to 0 so far";
    CHECK_EQ(preds.size(), p_fmat->info().num_row * num_group);
    // start collecting the prediction
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    Derived* self = static_cast<Derived*>(this);
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      // parallel over local batch
      const int K = 8;
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      const bst_omp_uint rest = nsize % K;
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize - rest; i += K) {
        const int tid = omp_get_thread_num();
        RegTree::FVec& feats = thread_temp[tid];
        int64_t ridx[K];
        RowBatch::Inst inst[K];
        for (int k = 0; k < K; ++k) {
          ridx[k] = static_cast<int64_t>(batch.base_rowid + i + k);
        }
        for (int k = 0; k < K; ++k) {
          inst[k] = batch[i + k];
        }
        for (int k = 0; k < K; ++k) {
          for (int gid = 0; gid < num_group; ++gid) {
            const size_t offset = ridx[k] * num_group + gid;
            preds[offset] +=
                self->PredValue(inst[k], gid, info.GetRoot(ridx[k]),
                                &feats, tree_begin, tree_end);
          }
        }
      }
      for (bst_omp_uint i = nsize - rest; i < nsize; ++i) {
        RegTree::FVec& feats = thread_temp[0];
        const int64_t ridx = static_cast<int64_t>(batch.base_rowid + i);
        const RowBatch::Inst inst = batch[i];
        for (int gid = 0; gid < num_group; ++gid) {
          const size_t offset = ridx * num_group + gid;
          preds[offset] +=
              self->PredValue(inst, gid, info.GetRoot(ridx),
                              &feats, tree_begin, tree_end);
        }
      }
    }
  }
  // initialize updater before using them
  inline void InitUpdater() {
    if (updaters.size() != 0) return;
    std::string tval = tparam.updater_seq;
    std::vector<std::string> ups = common::Split(tval, ',');
    for (const std::string& pstr : ups) {
      std::unique_ptr<TreeUpdater> up(TreeUpdater::Create(pstr.c_str()));
      up->Init(this->cfg);
      updaters.push_back(std::move(up));
    }
  }
  // do group specific group
  inline void
  BoostNewTrees(const std::vector<bst_gpair> &gpair,
                DMatrix *p_fmat,
                int bst_group,
                std::vector<std::unique_ptr<RegTree> >* ret) {
    this->InitUpdater();
    std::vector<RegTree*> new_trees;
    ret->clear();
    // create the trees
    for (int i = 0; i < tparam.num_parallel_tree; ++i) {
      if (tparam.process_type == kDefault) {
        // create new tree
        std::unique_ptr<RegTree> ptr(new RegTree());
        ptr->param.InitAllowUnknown(this->cfg);
        ptr->InitModel();
        new_trees.push_back(ptr.get());
        ret->push_back(std::move(ptr));
      } else if (tparam.process_type == kUpdate) {
        CHECK_LT(trees.size(), trees_to_update.size());
        // move an existing tree from trees_to_update
        auto t = std::move(trees_to_update[trees.size() +
                           bst_group * tparam.num_parallel_tree + i]);
        new_trees.push_back(t.get());
        ret->push_back(std::move(t));
      }
    }
    // update the trees
    for (auto& up : updaters) {
      up->Update(gpair, p_fmat, new_trees);
    }
  }
  // commit new trees all at once
  virtual void
  CommitModel(std::vector<std::unique_ptr<RegTree> >&& new_trees,
              int bst_group) {
    size_t old_ntree = trees.size();
    for (size_t i = 0; i < new_trees.size(); ++i) {
      trees.push_back(std::move(new_trees[i]));
      tree_info.push_back(bst_group);
    }
    mparam.num_trees += static_cast<int>(new_trees.size());

    // update cache entry
    for (auto &kv : cache_) {
      CacheEntry& e = kv.second;

      if (e.predictions.size() == 0) {
        PredLoopInternal<GBTree>(
            e.data.get(), &(e.predictions),
            0, trees.size(), true);
      } else {
        if (mparam.num_output_group == 1 && updaters.size() > 0 && new_trees.size() == 1
          && updaters.back()->UpdatePredictionCache(e.data.get(), &(e.predictions)) ) {
          {}  // do nothing
        } else {
          PredLoopInternal<GBTree>(
              e.data.get(), &(e.predictions),
              old_ntree, trees.size(), false);
        }
      }
    }
  }

  // make a prediction for a single instance
  inline bst_float PredValue(const RowBatch::Inst &inst,
                             int bst_group,
                             unsigned root_index,
                             RegTree::FVec *p_feats,
                             unsigned tree_begin,
                             unsigned tree_end) {
    bst_float psum = 0.0f;
    p_feats->Fill(inst);
    for (size_t i = tree_begin; i < tree_end; ++i) {
      if (tree_info[i] == bst_group) {
        int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
        psum += (*trees[i])[tid].leaf_value();
      }
    }
    p_feats->Drop(inst);
    return psum;
  }
  // predict independent leaf index
  inline void PredPath(DMatrix *p_fmat,
                       std::vector<bst_float> *out_preds,
                       unsigned ntree_limit) {
    const MetaInfo& info = p_fmat->info();
    // number of valid trees
    ntree_limit *= mparam.num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    std::vector<bst_float>& preds = *out_preds;
    preds.resize(info.num_row * ntree_limit);
    // start collecting the prediction
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        size_t ridx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec &feats = thread_temp[tid];
        feats.Fill(batch[i]);
        for (unsigned j = 0; j < ntree_limit; ++j) {
          int tid = trees[j]->GetLeafIndex(feats, info.GetRoot(ridx));
          preds[ridx * ntree_limit + j] = static_cast<bst_float>(tid);
        }
        feats.Drop(batch[i]);
      }
    }
  }
  // predict contributions
  inline void PredContrib(DMatrix *p_fmat,
                          std::vector<bst_float> *out_contribs,
                          unsigned ntree_limit) {
    const MetaInfo& info = p_fmat->info();
    // number of valid trees
    ntree_limit *= mparam.num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    const int ngroup = mparam.num_output_group;
    size_t ncolumns = mparam.num_feature + 1;
    // allocate space for (number of features + bias) times the number of rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(info.num_row * ncolumns * mparam.num_output_group);
    // make sure contributions is zeroed, we could be reusing a previously allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // initialize tree node mean values
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i=0; i < ntree_limit; ++i) {
      trees[i]->FillNodeMeanValues();
    }
    // start collecting the contributions
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    const std::vector<bst_float>& base_margin = info.base_margin;
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        size_t row_idx = static_cast<size_t>(batch.base_rowid + i);
        unsigned root_id = info.GetRoot(row_idx);
        RegTree::FVec &feats = thread_temp[omp_get_thread_num()];
        // loop over all classes
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float *p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          feats.Fill(batch[i]);
          // calculate contributions
          for (unsigned j = 0; j < ntree_limit; ++j) {
            if (tree_info[j] != gid) {
              continue;
            }
            trees[j]->CalculateContributions(feats, root_id, p_contribs);
          }
          feats.Drop(batch[i]);
          // add base margin to BIAS
          if (base_margin.size() != 0) {
            p_contribs[ncolumns - 1] += base_margin[row_idx * ngroup + gid];
          } else {
            p_contribs[ncolumns - 1] += base_margin_;
          }
        }
      }
    }
  }
  // init thread buffers
  inline void InitThreadTemp(int nthread) {
    int prev_thread_temp_size = thread_temp.size();
    if (prev_thread_temp_size < nthread) {
      thread_temp.resize(nthread, RegTree::FVec());
      for (int i = prev_thread_temp_size; i < nthread; ++i) {
        thread_temp[i].Init(mparam.num_feature);
      }
    }
  }
  // --- data structure ---
  // base margin
  bst_float base_margin_;
  // training parameter
  GBTreeTrainParam tparam;
  // model parameter
  GBTreeModelParam mparam;
  /*! \brief vector of trees stored in the model */
  std::vector<std::unique_ptr<RegTree> > trees;
  /*! \brief for the update process, a place to keep the initial trees */
  std::vector<std::unique_ptr<RegTree> > trees_to_update;
  /*! \brief some information indicator of the tree, reserved */
  std::vector<int> tree_info;
  // ----training fields----
  std::unordered_map<DMatrix*, CacheEntry> cache_;
  // configurations for tree
  std::vector<std::pair<std::string, std::string> > cfg;
  // temporal storage for per thread
  std::vector<RegTree::FVec> thread_temp;
  // the updaters that can be applied to each of tree
  std::vector<std::unique_ptr<TreeUpdater> > updaters;
};

// dart
class Dart : public GBTree {
 public:
  explicit Dart(bst_float base_margin) : GBTree(base_margin) {}

  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    GBTree::Configure(cfg);
    if (trees.size() == 0) {
      dparam.InitAllowUnknown(cfg);
    }
  }

  void Load(dmlc::Stream* fi) override {
    GBTree::Load(fi);
    weight_drop.resize(mparam.num_trees);
    if (mparam.num_trees != 0) {
      fi->Read(&weight_drop);
    }
  }

  void Save(dmlc::Stream* fo) const override {
    GBTree::Save(fo);
    if (weight_drop.size() != 0) {
      fo->Write(weight_drop);
    }
  }

  // predict the leaf scores with dropout if ntree_limit = 0
  void Predict(DMatrix* p_fmat,
               std::vector<bst_float>* out_preds,
               unsigned ntree_limit) override {
    DropTrees(ntree_limit);
    PredLoopInternal<Dart>(p_fmat, out_preds, 0, ntree_limit, true);
  }

  void Predict(const SparseBatch::Inst& inst,
               std::vector<bst_float>* out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    DropTrees(1);
    if (thread_temp.size() == 0) {
      thread_temp.resize(1, RegTree::FVec());
      thread_temp[0].Init(mparam.num_feature);
    }
    out_preds->resize(mparam.num_output_group);
    ntree_limit *= mparam.num_output_group;
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    // loop over output groups
    for (int gid = 0; gid < mparam.num_output_group; ++gid) {
      (*out_preds)[gid]
          = PredValue(inst, gid, root_index,
                      &thread_temp[0], 0, ntree_limit) + base_margin_;
    }
  }

 protected:
  friend class GBTree;
  // commit new trees all at once
  void CommitModel(std::vector<std::unique_ptr<RegTree> >&& new_trees,
                   int bst_group) override {
    for (size_t i = 0; i < new_trees.size(); ++i) {
      trees.push_back(std::move(new_trees[i]));
      tree_info.push_back(bst_group);
    }
    mparam.num_trees += static_cast<int>(new_trees.size());
    size_t num_drop = NormalizeTrees(new_trees.size());
    if (dparam.silent != 1) {
      LOG(INFO) << "drop " << num_drop << " trees, "
                << "weight = " << weight_drop.back();
    }
  }
  // predict the leaf scores without dropped trees
  inline bst_float PredValue(const RowBatch::Inst &inst,
                             int bst_group,
                             unsigned root_index,
                             RegTree::FVec *p_feats,
                             unsigned tree_begin,
                             unsigned tree_end) {
    bst_float psum = 0.0f;
    p_feats->Fill(inst);
    for (size_t i = tree_begin; i < tree_end; ++i) {
      if (tree_info[i] == bst_group) {
        bool drop = (std::binary_search(idx_drop.begin(), idx_drop.end(), i));
        if (!drop) {
          int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
          psum += weight_drop[i] * (*trees[i])[tid].leaf_value();
        }
      }
    }
    p_feats->Drop(inst);
    return psum;
  }

  // select dropped trees
  inline void DropTrees(unsigned ntree_limit_drop) {
    std::uniform_real_distribution<> runif(0.0, 1.0);
    auto& rnd = common::GlobalRandom();
    // reset
    idx_drop.clear();
    // sample dropped trees
    bool skip = false;
    if (dparam.skip_drop > 0.0) skip = (runif(rnd) < dparam.skip_drop);
    if (ntree_limit_drop == 0 && !skip) {
      if (dparam.sample_type == 1) {
        bst_float sum_weight = 0.0;
        for (size_t i = 0; i < weight_drop.size(); ++i) {
          sum_weight += weight_drop[i];
        }
        for (size_t i = 0; i < weight_drop.size(); ++i) {
          if (runif(rnd) < dparam.rate_drop * weight_drop.size() * weight_drop[i] / sum_weight) {
            idx_drop.push_back(i);
          }
        }
        if (dparam.one_drop && idx_drop.empty() && !weight_drop.empty()) {
          // the expression below is an ugly but MSVC2013-friendly equivalent of
          // size_t i = std::discrete_distribution<size_t>(weight_drop.begin(),
          //                                               weight_drop.end())(rnd);
          size_t i = std::discrete_distribution<size_t>(
            weight_drop.size(), 0., static_cast<double>(weight_drop.size()),
            [this](double x) -> double {
              return weight_drop[static_cast<size_t>(x)];
            })(rnd);
          idx_drop.push_back(i);
        }
      } else {
        for (size_t i = 0; i < weight_drop.size(); ++i) {
          if (runif(rnd) < dparam.rate_drop) {
            idx_drop.push_back(i);
          }
        }
        if (dparam.one_drop && idx_drop.empty() && !weight_drop.empty()) {
          size_t i = std::uniform_int_distribution<size_t>(0, weight_drop.size() - 1)(rnd);
          idx_drop.push_back(i);
        }
      }
    }
  }
  // set normalization factors
  inline size_t NormalizeTrees(size_t size_new_trees) {
    float lr = 1.0 * dparam.learning_rate / size_new_trees;
    size_t num_drop = idx_drop.size();
    if (num_drop == 0) {
      for (size_t i = 0; i < size_new_trees; ++i) {
        weight_drop.push_back(1.0);
      }
    } else {
      if (dparam.normalize_type == 1) {
        // normalize_type 1
        float factor = 1.0 / (1.0 + lr);
        for (size_t i = 0; i < idx_drop.size(); ++i) {
          weight_drop[idx_drop[i]] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop.push_back(factor);
        }
      } else {
        // normalize_type 0
        float factor = 1.0 * num_drop / (num_drop + lr);
        for (size_t i = 0; i < idx_drop.size(); ++i) {
          weight_drop[idx_drop[i]] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop.push_back(1.0 / (num_drop + lr));
        }
      }
    }
    // reset
    idx_drop.clear();
    return num_drop;
  }

  // --- data structure ---
  // training parameter
  DartTrainParam dparam;
  /*! \brief prediction buffer */
  std::vector<bst_float> weight_drop;
  // indexes of dropped trees
  std::vector<size_t> idx_drop;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBTreeModelParam);
DMLC_REGISTER_PARAMETER(GBTreeTrainParam);
DMLC_REGISTER_PARAMETER(DartTrainParam);

XGBOOST_REGISTER_GBM(GBTree, "gbtree")
.describe("Tree booster, gradient boosted trees.")
.set_body([](const std::vector<std::shared_ptr<DMatrix> >& cached_mats, bst_float base_margin) {
    GBTree* p = new GBTree(base_margin);
    p->InitCache(cached_mats);
    return p;
  });
XGBOOST_REGISTER_GBM(Dart, "dart")
.describe("Tree booster, dart.")
.set_body([](const std::vector<std::shared_ptr<DMatrix> >& cached_mats, bst_float base_margin) {
    GBTree* p = new Dart(base_margin);
    return p;
  });
}  // namespace gbm
}  // namespace xgboost
