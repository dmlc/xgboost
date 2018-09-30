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
#include <xgboost/predictor.h>
#include <xgboost/tree_updater.h>
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <limits>
#include <algorithm>
#include "../common/common.h"
#include "../common/host_device_vector.h"
#include "../common/random.h"
#include "gbtree_model.h"
#include "../common/timer.h"

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
  std::string predictor;
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
    DMLC_DECLARE_FIELD(predictor)
      .set_default("cpu_predictor")
      .describe("Predictor algorithm type");
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


// cache entry
struct CacheEntry {
  std::shared_ptr<DMatrix> data;
  std::vector<bst_float> predictions;
};

// gradient boosted trees
class GBTree : public GradientBooster {
 public:
  explicit GBTree(bst_float base_margin) : model_(base_margin) {}

  void InitCache(const std::vector<std::shared_ptr<DMatrix> > &cache) {
    cache_ = cache;
  }

  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    this->cfg_ = cfg;
    model_.Configure(cfg);
    // initialize the updaters only when needed.
    std::string updater_seq = tparam_.updater_seq;
    tparam_.InitAllowUnknown(cfg);
    if (updater_seq != tparam_.updater_seq) updaters_.clear();
    for (const auto& up : updaters_) {
      up->Init(cfg);
    }
    // for the 'update' process_type, move trees into trees_to_update
    if (tparam_.process_type == kUpdate) {
      model_.InitTreesToUpdate();
    }

    // configure predictor
    predictor_ = std::unique_ptr<Predictor>(Predictor::Create(tparam_.predictor));
    predictor_->Init(cfg, cache_);
    monitor_.Init("GBTree", tparam_.debug_verbose);
  }

  void Load(dmlc::Stream* fi) override {
    model_.Load(fi);

    this->cfg_.clear();
    this->cfg_.emplace_back(std::string("num_feature"),
                                       common::ToString(model_.param.num_feature));
  }

  void Save(dmlc::Stream* fo) const override {
    model_.Save(fo);
  }

  bool AllowLazyCheckPoint() const override {
    return model_.param.num_output_group == 1 ||
        tparam_.updater_seq.find("distcol") != std::string::npos;
  }

  void DoBoost(DMatrix* p_fmat,
               HostDeviceVector<GradientPair>* in_gpair,
               ObjFunction* obj) override {
    std::vector<std::vector<std::unique_ptr<RegTree> > > new_trees;
    const int ngroup = model_.param.num_output_group;
    monitor_.Start("BoostNewTrees");
    if (ngroup == 1) {
      std::vector<std::unique_ptr<RegTree> > ret;
      BoostNewTrees(in_gpair, p_fmat, 0, &ret);
      new_trees.push_back(std::move(ret));
    } else {
      CHECK_EQ(in_gpair->Size() % ngroup, 0U)
          << "must have exactly ngroup*nrow gpairs";
      // TODO(canonizer): perform this on GPU if HostDeviceVector has device set.
      HostDeviceVector<GradientPair> tmp(in_gpair->Size() / ngroup,
                                      GradientPair(), in_gpair->Devices());
      std::vector<GradientPair>& gpair_h = in_gpair->HostVector();
      auto nsize = static_cast<bst_omp_uint>(tmp.Size());
      for (int gid = 0; gid < ngroup; ++gid) {
        std::vector<GradientPair>& tmp_h = tmp.HostVector();
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          tmp_h[i] = gpair_h[i * ngroup + gid];
        }
        std::vector<std::unique_ptr<RegTree> > ret;
        BoostNewTrees(&tmp, p_fmat, gid, &ret);
        new_trees.push_back(std::move(ret));
      }
    }
    monitor_.Stop("BoostNewTrees");
    monitor_.Start("CommitModel");
    this->CommitModel(std::move(new_trees));
    monitor_.Stop("CommitModel");
  }

  void PredictBatch(DMatrix* p_fmat,
               HostDeviceVector<bst_float>* out_preds,
               unsigned ntree_limit) override {
    predictor_->PredictBatch(p_fmat, out_preds, model_, 0, ntree_limit);
  }

  void PredictInstance(const SparsePage::Inst& inst,
               std::vector<bst_float>* out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    predictor_->PredictInstance(inst, out_preds, model_,
                               ntree_limit, root_index);
  }

  void PredictLeaf(DMatrix* p_fmat,
                   std::vector<bst_float>* out_preds,
                   unsigned ntree_limit) override {
    predictor_->PredictLeaf(p_fmat, out_preds, model_, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate, int condition,
                           unsigned condition_feature) override {
    predictor_->PredictContribution(p_fmat, out_contribs, model_, ntree_limit, approximate);
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       unsigned ntree_limit, bool approximate) override {
    predictor_->PredictInteractionContributions(p_fmat, out_contribs, model_,
                                               ntree_limit, approximate);
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    return model_.DumpModel(fmap, with_stats, format);
  }

 protected:
  // initialize updater before using them
  inline void InitUpdater() {
    if (updaters_.size() != 0) return;
    std::string tval = tparam_.updater_seq;
    std::vector<std::string> ups = common::Split(tval, ',');
    for (const std::string& pstr : ups) {
      std::unique_ptr<TreeUpdater> up(TreeUpdater::Create(pstr.c_str()));
      up->Init(this->cfg_);
      updaters_.push_back(std::move(up));
    }
  }

  // do group specific group
  inline void BoostNewTrees(HostDeviceVector<GradientPair>* gpair,
                            DMatrix *p_fmat,
                            int bst_group,
                            std::vector<std::unique_ptr<RegTree> >* ret) {
    this->InitUpdater();
    std::vector<RegTree*> new_trees;
    ret->clear();
    // create the trees
    for (int i = 0; i < tparam_.num_parallel_tree; ++i) {
      if (tparam_.process_type == kDefault) {
        // create new tree
        std::unique_ptr<RegTree> ptr(new RegTree());
        ptr->param.InitAllowUnknown(this->cfg_);
        ptr->InitModel();
        new_trees.push_back(ptr.get());
        ret->push_back(std::move(ptr));
      } else if (tparam_.process_type == kUpdate) {
        CHECK_LT(model_.trees.size(), model_.trees_to_update.size());
        // move an existing tree from trees_to_update
        auto t = std::move(model_.trees_to_update[model_.trees.size() +
                           bst_group * tparam_.num_parallel_tree + i]);
        new_trees.push_back(t.get());
        ret->push_back(std::move(t));
      }
    }
    // update the trees
    for (auto& up : updaters_) {
      up->Update(gpair, p_fmat, new_trees);
}
  }

  // commit new trees all at once
  virtual void
  CommitModel(std::vector<std::vector<std::unique_ptr<RegTree>>>&& new_trees) {
    int num_new_trees = 0;
    for (int gid = 0; gid < model_.param.num_output_group; ++gid) {
      num_new_trees += new_trees[gid].size();
      model_.CommitModel(std::move(new_trees[gid]), gid);
    }
    predictor_->UpdatePredictionCache(model_, &updaters_, num_new_trees);
  }

  // --- data structure ---
  GBTreeModel model_;
  // training parameter
  GBTreeTrainParam tparam_;
  // ----training fields----
  // configurations for tree
  std::vector<std::pair<std::string, std::string> > cfg_;
  // the updaters that can be applied to each of tree
  std::vector<std::unique_ptr<TreeUpdater>> updaters_;
  // Cached matrices
  std::vector<std::shared_ptr<DMatrix>> cache_;
  std::unique_ptr<Predictor> predictor_;
  common::Monitor monitor_;
};

// dart
class Dart : public GBTree {
 public:
  explicit Dart(bst_float base_margin) : GBTree(base_margin) {}

  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    GBTree::Configure(cfg);
    if (model_.trees.size() == 0) {
      dparam_.InitAllowUnknown(cfg);
    }
  }

  void Load(dmlc::Stream* fi) override {
    GBTree::Load(fi);
    weight_drop_.resize(model_.param.num_trees);
    if (model_.param.num_trees != 0) {
      fi->Read(&weight_drop_);
    }
  }

  void Save(dmlc::Stream* fo) const override {
    GBTree::Save(fo);
    if (weight_drop_.size() != 0) {
      fo->Write(weight_drop_);
    }
  }

  // predict the leaf scores with dropout if ntree_limit = 0
  void PredictBatch(DMatrix* p_fmat,
                    HostDeviceVector<bst_float>* out_preds,
                    unsigned ntree_limit) override {
    DropTrees(ntree_limit);
    PredLoopInternal<Dart>(p_fmat, &out_preds->HostVector(), 0, ntree_limit, true);
  }

  void PredictInstance(const SparsePage::Inst& inst,
               std::vector<bst_float>* out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    DropTrees(1);
    if (thread_temp_.size() == 0) {
      thread_temp_.resize(1, RegTree::FVec());
      thread_temp_[0].Init(model_.param.num_feature);
    }
    out_preds->resize(model_.param.num_output_group);
    ntree_limit *= model_.param.num_output_group;
    if (ntree_limit == 0 || ntree_limit > model_.trees.size()) {
      ntree_limit = static_cast<unsigned>(model_.trees.size());
    }
    // loop over output groups
    for (int gid = 0; gid < model_.param.num_output_group; ++gid) {
      (*out_preds)[gid]
          = PredValue(inst, gid, root_index,
                      &thread_temp_[0], 0, ntree_limit) + model_.base_margin;
    }
  }

 protected:
  friend class GBTree;
  // internal prediction loop
  // add predictions to out_preds
  template<typename Derived>
  inline void PredLoopInternal(
      DMatrix* p_fmat,
      std::vector<bst_float>* out_preds,
      unsigned tree_begin,
      unsigned ntree_limit,
      bool init_out_preds) {
    int num_group = model_.param.num_output_group;
    ntree_limit *= num_group;
    if (ntree_limit == 0 || ntree_limit > model_.trees.size()) {
      ntree_limit = static_cast<unsigned>(model_.trees.size());
    }

    if (init_out_preds) {
      size_t n = num_group * p_fmat->Info().num_row_;
      const std::vector<bst_float>& base_margin = p_fmat->Info().base_margin_;
      out_preds->resize(n);
      if (base_margin.size() != 0) {
        CHECK_EQ(out_preds->size(), n);
        std::copy(base_margin.begin(), base_margin.end(), out_preds->begin());
      } else {
        std::fill(out_preds->begin(), out_preds->end(), model_.base_margin);
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
    const MetaInfo& info = p_fmat->Info();
    const int nthread = omp_get_max_threads();
    CHECK_EQ(num_group, model_.param.num_output_group);
    InitThreadTemp(nthread);
    std::vector<bst_float>& preds = *out_preds;
    CHECK_EQ(model_.param.size_leaf_vector, 0)
        << "size_leaf_vector is enforced to 0 so far";
    CHECK_EQ(preds.size(), p_fmat->Info().num_row_ * num_group);
    // start collecting the prediction
    auto iter = p_fmat->RowIterator();
    auto* self = static_cast<Derived*>(this);
    iter->BeforeFirst();
    while (iter->Next()) {
      auto &batch = iter->Value();
      // parallel over local batch
      constexpr int kUnroll = 8;
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
      const bst_omp_uint rest = nsize % kUnroll;
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize - rest; i += kUnroll) {
        const int tid = omp_get_thread_num();
        RegTree::FVec& feats = thread_temp_[tid];
        int64_t ridx[kUnroll];
        SparsePage::Inst inst[kUnroll];
        for (int k = 0; k < kUnroll; ++k) {
          ridx[k] = static_cast<int64_t>(batch.base_rowid + i + k);
        }
        for (int k = 0; k < kUnroll; ++k) {
          inst[k] = batch[i + k];
        }
        for (int k = 0; k < kUnroll; ++k) {
          for (int gid = 0; gid < num_group; ++gid) {
            const size_t offset = ridx[k] * num_group + gid;
            preds[offset] +=
                self->PredValue(inst[k], gid, info.GetRoot(ridx[k]),
                                &feats, tree_begin, tree_end);
          }
        }
      }
      for (bst_omp_uint i = nsize - rest; i < nsize; ++i) {
        RegTree::FVec& feats = thread_temp_[0];
        const auto ridx = static_cast<int64_t>(batch.base_rowid + i);
        const SparsePage::Inst inst = batch[i];
        for (int gid = 0; gid < num_group; ++gid) {
          const size_t offset = ridx * num_group + gid;
          preds[offset] +=
              self->PredValue(inst, gid, info.GetRoot(ridx),
                              &feats, tree_begin, tree_end);
        }
      }
    }
  }

  // commit new trees all at once
  void
  CommitModel(std::vector<std::vector<std::unique_ptr<RegTree>>>&& new_trees) override {
    int num_new_trees = 0;
    for (int gid = 0; gid < model_.param.num_output_group; ++gid) {
      num_new_trees += new_trees[gid].size();
      model_.CommitModel(std::move(new_trees[gid]), gid);
    }
    size_t num_drop = NormalizeTrees(num_new_trees);
    if (dparam_.silent != 1) {
      LOG(INFO) << "drop " << num_drop << " trees, "
                << "weight = " << weight_drop_.back();
    }
  }

  // predict the leaf scores without dropped trees
  inline bst_float PredValue(const SparsePage::Inst &inst,
                             int bst_group,
                             unsigned root_index,
                             RegTree::FVec *p_feats,
                             unsigned tree_begin,
                             unsigned tree_end) {
    bst_float psum = 0.0f;
    p_feats->Fill(inst);
    for (size_t i = tree_begin; i < tree_end; ++i) {
      if (model_.tree_info[i] == bst_group) {
        bool drop = (std::binary_search(idx_drop_.begin(), idx_drop_.end(), i));
        if (!drop) {
          int tid = model_.trees[i]->GetLeafIndex(*p_feats, root_index);
          psum += weight_drop_[i] * (*model_.trees[i])[tid].LeafValue();
        }
      }
    }
    p_feats->Drop(inst);
    return psum;
  }

  // select which trees to drop
  inline void DropTrees(unsigned ntree_limit_drop) {
    idx_drop_.clear();
    if (ntree_limit_drop > 0) return;

    std::uniform_real_distribution<> runif(0.0, 1.0);
    auto& rnd = common::GlobalRandom();
    bool skip = false;
    if (dparam_.skip_drop > 0.0) skip = (runif(rnd) < dparam_.skip_drop);
    // sample some trees to drop
    if (!skip) {
      if (dparam_.sample_type == 1) {
        bst_float sum_weight = 0.0;
        for (auto elem : weight_drop_) {
          sum_weight += elem;
        }
        for (size_t i = 0; i < weight_drop_.size(); ++i) {
          if (runif(rnd) < dparam_.rate_drop * weight_drop_.size() * weight_drop_[i] / sum_weight) {
            idx_drop_.push_back(i);
          }
        }
        if (dparam_.one_drop && idx_drop_.empty() && !weight_drop_.empty()) {
          // the expression below is an ugly but MSVC2013-friendly equivalent of
          // size_t i = std::discrete_distribution<size_t>(weight_drop.begin(),
          //                                               weight_drop.end())(rnd);
          size_t i = std::discrete_distribution<size_t>(
            weight_drop_.size(), 0., static_cast<double>(weight_drop_.size()),
            [this](double x) -> double {
              return weight_drop_[static_cast<size_t>(x)];
            })(rnd);
          idx_drop_.push_back(i);
        }
      } else {
        for (size_t i = 0; i < weight_drop_.size(); ++i) {
          if (runif(rnd) < dparam_.rate_drop) {
            idx_drop_.push_back(i);
          }
        }
        if (dparam_.one_drop && idx_drop_.empty() && !weight_drop_.empty()) {
          size_t i = std::uniform_int_distribution<size_t>(0, weight_drop_.size() - 1)(rnd);
          idx_drop_.push_back(i);
        }
      }
    }
  }

  // set normalization factors
  inline size_t NormalizeTrees(size_t size_new_trees) {
    float lr = 1.0 * dparam_.learning_rate / size_new_trees;
    size_t num_drop = idx_drop_.size();
    if (num_drop == 0) {
      for (size_t i = 0; i < size_new_trees; ++i) {
        weight_drop_.push_back(1.0);
      }
    } else {
      if (dparam_.normalize_type == 1) {
        // normalize_type 1
        float factor = 1.0 / (1.0 + lr);
        for (auto i : idx_drop_) {
          weight_drop_[i] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop_.push_back(factor);
        }
      } else {
        // normalize_type 0
        float factor = 1.0 * num_drop / (num_drop + lr);
        for (auto i : idx_drop_) {
          weight_drop_[i] *= factor;
        }
        for (size_t i = 0; i < size_new_trees; ++i) {
          weight_drop_.push_back(1.0 / (num_drop + lr));
        }
      }
    }
    // reset
    idx_drop_.clear();
    return num_drop;
  }

  // init thread buffers
  inline void InitThreadTemp(int nthread) {
    int prev_thread_temp_size = thread_temp_.size();
    if (prev_thread_temp_size < nthread) {
      thread_temp_.resize(nthread, RegTree::FVec());
      for (int i = prev_thread_temp_size; i < nthread; ++i) {
        thread_temp_[i].Init(model_.param.num_feature);
      }
    }
  }

  // --- data structure ---
  // training parameter
  DartTrainParam dparam_;
  /*! \brief prediction buffer */
  std::vector<bst_float> weight_drop_;
  // indexes of dropped trees
  std::vector<size_t> idx_drop_;
  // temporal storage for per thread
  std::vector<RegTree::FVec> thread_temp_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBTreeModelParam);
DMLC_REGISTER_PARAMETER(GBTreeTrainParam);
DMLC_REGISTER_PARAMETER(DartTrainParam);

XGBOOST_REGISTER_GBM(GBTree, "gbtree")
.describe("Tree booster, gradient boosted trees.")
.set_body([](const std::vector<std::shared_ptr<DMatrix> >& cached_mats, bst_float base_margin) {
    auto* p = new GBTree(base_margin);
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
