/*!
 * Copyright 2014 by Contributors
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/logging.h>
#include <xgboost/gbm.h>
#include <xgboost/tree_updater.h>

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <limits>
#include "../common/common.h"

#include "../common/random.h"

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gbtree);

/*! \brief training parameters */
struct GBTreeTrainParam : public dmlc::Parameter<GBTreeTrainParam> {
  /*!
   * \brief number of parallel trees constructed each iteration
   *  use this option to support boosted random forest
   */
  int num_parallel_tree;
  /*! \brief tree updater sequence */
  std::string updater_seq;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GBTreeTrainParam) {
    DMLC_DECLARE_FIELD(num_parallel_tree).set_lower_bound(1).set_default(1)
        .describe("Number of parallel trees constructed during each iteration."\
                  " This option is used to support boosted random forest");
    DMLC_DECLARE_FIELD(updater_seq).set_default("grow_colmaker,prune")
        .describe("Tree updater sequence.");
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
  /*! \brief how many trees are dropped */
  float rate_drop;
  /*! \brief whether to drop trees */
  float skip_drop;
  /*! \brief learning step size for a time */
  float learning_rate;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DartTrainParam) {
    DMLC_DECLARE_FIELD(silent).set_default(false)
        .describe("Not print information during trainig.");
    DMLC_DECLARE_FIELD(sample_type).set_default(0)
        .add_enum("uniform", 0)
        .add_enum("weighted", 1)
        .describe("Different types of sampling algorithm.");
    DMLC_DECLARE_FIELD(normalize_type).set_default(0)
        .add_enum("tree", 0)
        .add_enum("forest", 1)
        .describe("Different types of normalization algorithm.");
    DMLC_DECLARE_FIELD(rate_drop).set_range(0.0f, 1.0f).set_default(0.0f)
        .describe("Parameter of how many trees are dropped.");
    DMLC_DECLARE_FIELD(skip_drop).set_range(0.0f, 1.0f).set_default(0.0f)
        .describe("Parameter of whether to drop trees.");
    DMLC_DECLARE_FIELD(learning_rate).set_lower_bound(0.0f).set_default(0.3f)
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
  /*! \brief pad this space, for backward compatiblity reason.*/
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

// gradient boosted trees
class GBTree : public GradientBooster {
 public:
  GBTree() : num_pbuffer(0) {}

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
  }

  void Load(dmlc::Stream* fi) override {
    CHECK_EQ(fi->Read(&mparam, sizeof(mparam)), sizeof(mparam))
        << "GBTree: invalid model file";
    trees.clear();
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
    // clear the predict buffer.
    this->ResetPredBuffer(num_pbuffer);
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

  void ResetPredBuffer(size_t num_pbuffer) override {
    this->num_pbuffer = num_pbuffer;
    pred_buffer.clear();
    pred_counter.clear();
    pred_buffer.resize(this->PredBufferSize(), 0.0f);
    pred_counter.resize(this->PredBufferSize(), 0);
  }

  bool AllowLazyCheckPoint() const override {
    return mparam.num_output_group == 1 ||
        tparam.updater_seq.find("distcol") != std::string::npos;
  }

  void DoBoost(DMatrix* p_fmat,
               int64_t buffer_offset,
               std::vector<bst_gpair>* in_gpair) override {
    const std::vector<bst_gpair>& gpair = *in_gpair;
    std::vector<std::vector<std::unique_ptr<RegTree> > > new_trees;
    if (mparam.num_output_group == 1) {
      std::vector<std::unique_ptr<RegTree> > ret;
      BoostNewTrees(gpair, p_fmat, buffer_offset, 0, &ret);
      new_trees.push_back(std::move(ret));
    } else {
      const int ngroup = mparam.num_output_group;
      CHECK_EQ(gpair.size() % ngroup, 0)
          << "must have exactly ngroup*nrow gpairs";
      std::vector<bst_gpair> tmp(gpair.size() / ngroup);
      for (int gid = 0; gid < ngroup; ++gid) {
        bst_omp_uint nsize = static_cast<bst_omp_uint>(tmp.size());
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          tmp[i] = gpair[i * ngroup + gid];
        }
        std::vector<std::unique_ptr<RegTree> > ret;
        BoostNewTrees(tmp, p_fmat, buffer_offset, gid, &ret);
        new_trees.push_back(std::move(ret));
      }
    }
    for (int gid = 0; gid < mparam.num_output_group; ++gid) {
      this->CommitModel(std::move(new_trees[gid]), gid);
    }
  }

  void Predict(DMatrix* p_fmat,
               int64_t buffer_offset,
               std::vector<float>* out_preds,
               unsigned ntree_limit) override {
    const MetaInfo& info = p_fmat->info();
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    InitThreadTemp(nthread);
    std::vector<float> &preds = *out_preds;
    const size_t stride = p_fmat->info().num_row * mparam.num_output_group;
    preds.resize(stride * (mparam.size_leaf_vector+1));
    // start collecting the prediction
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();

    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      int ridx_error = 0;
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        RegTree::FVec &feats = thread_temp[tid];
        int64_t ridx = static_cast<int64_t>(batch.base_rowid + i);
        if (static_cast<size_t>(ridx) >= info.num_row) {
          ridx_error = 1;
          continue;
        }
        // loop over output groups
        for (int gid = 0; gid < mparam.num_output_group; ++gid) {
          this->Pred(batch[i],
                     buffer_offset < 0 ? -1 : buffer_offset + ridx,
                     gid, info.GetRoot(ridx), &feats,
                     &preds[ridx * mparam.num_output_group + gid], stride,
                     ntree_limit);
        }
      }
      CHECK(!ridx_error) << "ridx out of bounds";
    }
  }

  void Predict(const SparseBatch::Inst& inst,
               std::vector<float>* out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    if (thread_temp.size() == 0) {
      thread_temp.resize(1, RegTree::FVec());
      thread_temp[0].Init(mparam.num_feature);
    }
    out_preds->resize(mparam.num_output_group * (mparam.size_leaf_vector+1));
    // loop over output groups
    for (int gid = 0; gid < mparam.num_output_group; ++gid) {
      this->Pred(inst, -1, gid, root_index, &thread_temp[0],
                 &(*out_preds)[gid], mparam.num_output_group,
                 ntree_limit);
    }
  }

  void PredictLeaf(DMatrix* p_fmat,
                   std::vector<float>* out_preds,
                   unsigned ntree_limit) override {
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    InitThreadTemp(nthread);
    this->PredPath(p_fmat, out_preds, ntree_limit);
  }

  std::vector<std::string> Dump2Text(const FeatureMap& fmap, int option) const override {
    std::vector<std::string> dump;
    for (size_t i = 0; i < trees.size(); i++) {
      dump.push_back(trees[i]->Dump2Text(fmap, option & 1));
    }
    return dump;
  }

 protected:
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
                int64_t buffer_offset,
                int bst_group,
                std::vector<std::unique_ptr<RegTree> >* ret) {
    this->InitUpdater();
    std::vector<RegTree*> new_trees;
    ret->clear();
    // create the trees
    for (int i = 0; i < tparam.num_parallel_tree; ++i) {
      std::unique_ptr<RegTree> ptr(new RegTree());
      ptr->param.InitAllowUnknown(this->cfg);
      ptr->InitModel();
      new_trees.push_back(ptr.get());
      ret->push_back(std::move(ptr));
    }
    // update the trees
    for (auto& up : updaters) {
      up->Update(gpair, p_fmat, new_trees);
    }
    // optimization, update buffer, if possible
    // this is only under distributed column mode
    // for safety check of lazy checkpoint
    if (buffer_offset >= 0 &&
        new_trees.size() == 1 && updaters.size() > 0 &&
        updaters.back()->GetLeafPosition() != nullptr) {
      CHECK_EQ(p_fmat->info().num_row, p_fmat->buffered_rowset().size());
      this->UpdateBufferByPosition(p_fmat,
                                   buffer_offset,
                                   bst_group,
                                   *new_trees[0],
                                   updaters.back()->GetLeafPosition());
    }
  }
  // commit new trees all at once
  virtual void
  CommitModel(std::vector<std::unique_ptr<RegTree> >&& new_trees,
              int bst_group) {
    for (size_t i = 0; i < new_trees.size(); ++i) {
      trees.push_back(std::move(new_trees[i]));
      tree_info.push_back(bst_group);
    }
    mparam.num_trees += static_cast<int>(new_trees.size());
  }
  // update buffer by pre-cached position
  inline void UpdateBufferByPosition(DMatrix *p_fmat,
                                     int64_t buffer_offset,
                                     int bst_group,
                                     const RegTree &new_tree,
                                     const int* leaf_position) {
    const RowSet& rowset = p_fmat->buffered_rowset();
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
    int pred_counter_error = 0, tid_error = 0;
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const bst_uint ridx = rowset[i];
      const int64_t bid = this->BufferOffset(buffer_offset + ridx, bst_group);
      const int tid = leaf_position[ridx];
      if (pred_counter[bid] != trees.size()) {
        pred_counter_error = 1;
        continue;
      }
      if (tid < 0) {
        tid_error = 1;
        continue;
      }
      pred_buffer[bid] += new_tree[tid].leaf_value();
      for (int i = 0; i < mparam.size_leaf_vector; ++i) {
        pred_buffer[bid + i + 1] += new_tree.leafvec(tid)[i];
      }
      pred_counter[bid] += tparam.num_parallel_tree;
    }
    CHECK(!pred_counter_error) << "incorrect pred_counter[bid]";
    CHECK(!tid_error) << "tid cannot be negative";
  }
  // make a prediction for a single instance
  inline void Pred(const RowBatch::Inst &inst,
                   int64_t buffer_index,
                   int bst_group,
                   unsigned root_index,
                   RegTree::FVec *p_feats,
                   float *out_pred,
                   size_t stride,
                   unsigned ntree_limit) {
    size_t itop = 0;
    float  psum = 0.0f;
    // sum of leaf vector
    std::vector<float> vec_psum(mparam.size_leaf_vector, 0.0f);
    const int64_t bid = this->BufferOffset(buffer_index, bst_group);
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
          if (--treeleft == 0) break;
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
  // predict independent leaf index
  inline void PredPath(DMatrix *p_fmat,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit) {
    const MetaInfo& info = p_fmat->info();
    // number of valid trees
    if (ntree_limit == 0 || ntree_limit > trees.size()) {
      ntree_limit = static_cast<unsigned>(trees.size());
    }
    std::vector<float>& preds = *out_preds;
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
          preds[ridx * ntree_limit + j] = static_cast<float>(tid);
        }
        feats.Drop(batch[i]);
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
  /*! \return size of prediction buffer actually needed */
  inline size_t PredBufferSize() const {
    return mparam.num_output_group * num_pbuffer * (mparam.size_leaf_vector + 1);
  }
  /*!
   * \brief get the buffer offset given a buffer index and group id
   * \return calculated buffer offset
   */
  inline int64_t BufferOffset(int64_t buffer_index, int bst_group) const {
    if (buffer_index < 0) return -1;
    size_t bidx = static_cast<size_t>(buffer_index);
    CHECK_LT(bidx, num_pbuffer);
    return (bidx + num_pbuffer * bst_group) * (mparam.size_leaf_vector + 1);
  }

  // --- data structure ---
  // training parameter
  GBTreeTrainParam tparam;
  // model parameter
  GBTreeModelParam mparam;
  /*! \brief vector of trees stored in the model */
  std::vector<std::unique_ptr<RegTree> > trees;
  /*! \brief some information indicator of the tree, reserved */
  std::vector<int> tree_info;
  /*! \brief predict buffer size */
  size_t num_pbuffer;
  /*! \brief prediction buffer */
  std::vector<float> pred_buffer;
  /*! \brief prediction buffer counter, remember the prediction */
  std::vector<unsigned> pred_counter;
  // ----training fields----
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
  Dart() {}

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
               int64_t buffer_offset,
               std::vector<float>* out_preds,
               unsigned ntree_limit) override {
    DropTrees(ntree_limit);
    const MetaInfo& info = p_fmat->info();
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    InitThreadTemp(nthread);
    std::vector<float> &preds = *out_preds;
    const size_t stride = p_fmat->info().num_row * mparam.num_output_group;
    preds.resize(stride * (mparam.size_leaf_vector+1));
    // start collecting the prediction
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();

    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        RegTree::FVec &feats = thread_temp[tid];
        int64_t ridx = static_cast<int64_t>(batch.base_rowid + i);
        CHECK_LT(static_cast<size_t>(ridx), info.num_row);
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

  void Predict(const SparseBatch::Inst& inst,
               std::vector<float>* out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    DropTrees(1);
    if (thread_temp.size() == 0) {
      thread_temp.resize(1, RegTree::FVec());
      thread_temp[0].Init(mparam.num_feature);
    }
    out_preds->resize(mparam.num_output_group * (mparam.size_leaf_vector+1));
    // loop over output groups
    for (int gid = 0; gid < mparam.num_output_group; ++gid) {
      this->Pred(inst, -1, gid, root_index, &thread_temp[0],
                 &(*out_preds)[gid], mparam.num_output_group,
                 ntree_limit);
    }
  }

 protected:
  // commit new trees all at once
  virtual void
  CommitModel(std::vector<std::unique_ptr<RegTree> >&& new_trees,
              int bst_group) {
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
  inline void Pred(const RowBatch::Inst &inst,
                   int64_t buffer_index,
                   int bst_group,
                   unsigned root_index,
                   RegTree::FVec *p_feats,
                   float *out_pred,
                   size_t stride,
                   unsigned ntree_limit) {
    float  psum = 0.0f;
    // sum of leaf vector
    std::vector<float> vec_psum(mparam.size_leaf_vector, 0.0f);
    const int64_t bid = this->BufferOffset(buffer_index, bst_group);
    p_feats->Fill(inst);
    for (size_t i = 0; i < trees.size(); ++i) {
      if (tree_info[i] == bst_group) {
        bool drop = (std::find(idx_drop.begin(), idx_drop.end(), i) != idx_drop.end());
        if (!drop) {
          int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
          psum += weight_drop[i] * (*trees[i])[tid].leaf_value();
          for (int j = 0; j < mparam.size_leaf_vector; ++j) {
            vec_psum[j] += weight_drop[i] * trees[i]->leafvec(tid)[j];
          }
        }
      }
    }
    p_feats->Drop(inst);
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
        float sum_weight = 0.0;
        for (size_t i = 0; i < weight_drop.size(); ++i) {
          sum_weight += weight_drop[i];
        }
        for (size_t i = 0; i < weight_drop.size(); ++i) {
          if (runif(rnd) < dparam.rate_drop * weight_drop.size() * weight_drop[i] / sum_weight) {
            idx_drop.push_back(i);
          }
        }
      } else {
        for (size_t i = 0; i < weight_drop.size(); ++i) {
          if (runif(rnd) < dparam.rate_drop) {
            idx_drop.push_back(i);
          }
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
          weight_drop[i] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop.push_back(factor);
        }
      } else {
        // normalize_type 0
        float factor = 1.0 * num_drop / (num_drop + lr);
        for (size_t i = 0; i < idx_drop.size(); ++i) {
          weight_drop[i] *= factor;
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
  std::vector<float> weight_drop;
  // indexes of dropped trees
  std::vector<size_t> idx_drop;
};

// register the ojective functions
DMLC_REGISTER_PARAMETER(GBTreeModelParam);
DMLC_REGISTER_PARAMETER(GBTreeTrainParam);
DMLC_REGISTER_PARAMETER(DartTrainParam);

XGBOOST_REGISTER_GBM(GBTree, "gbtree")
.describe("Tree booster, gradient boosted trees.")
.set_body([]() {
    return new GBTree();
  });
XGBOOST_REGISTER_GBM(Dart, "dart")
.describe("Tree booster, dart.")
.set_body([]() {
    return new Dart();
  });
}  // namespace gbm
}  // namespace xgboost
