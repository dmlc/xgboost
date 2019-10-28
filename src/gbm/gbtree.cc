/*!
 * Copyright 2014-2019 by Contributors
 * \file gbtree.cc
 * \brief gradient boosted tree implementation.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <limits>
#include <algorithm>

#include "xgboost/logging.h"
#include "xgboost/gbm.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_updater.h"
#include "xgboost/host_device_vector.h"

#include "gbtree.h"
#include "gbtree_model.h"
#include "../common/common.h"
#include "../common/random.h"
#include "../common/timer.h"


namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gbtree);

void GBTree::Configure(const Args& cfg) {
  this->cfg_ = cfg;
  tparam_.UpdateAllowUnknown(cfg);

  model_.Configure(cfg);

  // for the 'update' process_type, move trees into trees_to_update
  if (tparam_.process_type == TreeProcessType::kUpdate) {
    model_.InitTreesToUpdate();
  }

  // configure predictors
  if (!cpu_predictor_) {
    cpu_predictor_ = std::unique_ptr<Predictor>(
        Predictor::Create("cpu_predictor", this->learner_param_));
    cpu_predictor_->Configure(cfg, cache_);
  }
#if defined(XGBOOST_USE_CUDA)
  if (!gpu_predictor_) {
    gpu_predictor_ = std::unique_ptr<Predictor>(
        Predictor::Create("gpu_predictor", this->learner_param_));
    gpu_predictor_->Configure(cfg, cache_);
  }
#endif  // defined(XGBOOST_USE_CUDA)

  monitor_.Init("GBTree");

  specified_predictor_ = std::any_of(cfg.cbegin(), cfg.cend(),
                   [](std::pair<std::string, std::string> const& arg) {
                     return arg.first == "predictor";
                   });
  if (!specified_predictor_ && tparam_.tree_method == TreeMethod::kGPUHist) {
    tparam_.predictor = "gpu_predictor";
  }

  specified_updater_ = std::any_of(cfg.cbegin(), cfg.cend(),
                   [](std::pair<std::string, std::string> const& arg) {
                     return arg.first == "updater";
                   });
  if (specified_updater_) {
    LOG(WARNING) << "DANGER AHEAD: You have manually specified `updater` "
        "parameter. The `tree_method` parameter will be ignored. "
        "Incorrect sequence of updaters will produce undefined "
        "behavior. For common uses, we recommend using "
        "`tree_method` parameter instead.";
  } else {
    this->ConfigureUpdaters();
    LOG(DEBUG) << "Using updaters: " << tparam_.updater_seq;
  }

  configured_ = true;
}

// FIXME(trivialfis): This handles updaters and predictor.  Because the choice of updaters
// depends on whether external memory is used and how large is dataset.  We can remove the
// dependency on DMatrix once `hist` tree method can handle external memory so that we can
// make it default.
void GBTree::ConfigureWithKnownData(Args const& cfg, DMatrix* fmat) {
  std::string updater_seq = tparam_.updater_seq;

  this->PerformTreeMethodHeuristic(fmat);
  this->ConfigureUpdaters();

  // initialize the updaters only when needed.
  if (updater_seq != tparam_.updater_seq) {
    LOG(DEBUG) << "Using updaters: " << tparam_.updater_seq;
    this->updaters_.clear();
  }

  this->InitUpdater(cfg);
}

void GBTree::PerformTreeMethodHeuristic(DMatrix* fmat) {
  if (specified_updater_) {
    // This method is disabled when `updater` parameter is explicitly
    // set, since only experts are expected to do so.
    return;
  }
  // tparam_ is set before calling this function.
  if (tparam_.tree_method != TreeMethod::kAuto) {
    return;
  }

  tparam_.updater_seq = "grow_histmaker,prune";
  if (rabit::IsDistributed()) {
    LOG(WARNING) <<
      "Tree method is automatically selected to be 'approx' "
      "for distributed training.";
    tparam_.tree_method = TreeMethod::kApprox;
  } else if (!fmat->SingleColBlock()) {
    LOG(WARNING) << "Tree method is automatically set to 'approx' "
                    "since external-memory data matrix is used.";
    tparam_.tree_method = TreeMethod::kApprox;
  } else if (fmat->Info().num_row_ >= (4UL << 20UL)) {
    /* Choose tree_method='approx' automatically for large data matrix */
    LOG(WARNING) << "Tree method is automatically selected to be "
        "'approx' for faster speed. To use old behavior "
        "(exact greedy algorithm on single machine), "
        "set tree_method to 'exact'.";
    tparam_.tree_method = TreeMethod::kApprox;
  } else {
    tparam_.tree_method = TreeMethod::kExact;
  }
  LOG(DEBUG) << "Using tree method: " << static_cast<int>(tparam_.tree_method);
}

void GBTree::ConfigureUpdaters() {
  // `updater` parameter was manually specified
  /* Choose updaters according to tree_method parameters */
  switch (tparam_.tree_method) {
    case TreeMethod::kAuto:
      // Use heuristic to choose between 'exact' and 'approx' This
      // choice is carried out in PerformTreeMethodHeuristic() before
      // calling this function.
      break;
    case TreeMethod::kApprox:
      tparam_.updater_seq = "grow_histmaker,prune";
      break;
    case TreeMethod::kExact:
      tparam_.updater_seq = "grow_colmaker,prune";
      break;
    case TreeMethod::kHist:
      LOG(INFO) <<
          "Tree method is selected to be 'hist', which uses a "
          "single updater grow_quantile_histmaker.";
      tparam_.updater_seq = "grow_quantile_histmaker";
      break;
    case TreeMethod::kGPUHist:
      this->AssertGPUSupport();
      tparam_.updater_seq = "grow_gpu_hist";
      if (!specified_predictor_) {
        tparam_.predictor = "gpu_predictor";
      }
      break;
    default:
      LOG(FATAL) << "Unknown tree_method ("
                 << static_cast<int>(tparam_.tree_method) << ") detected";
  }
}

void GBTree::DoBoost(DMatrix* p_fmat,
                     HostDeviceVector<GradientPair>* in_gpair,
                     ObjFunction* obj) {
  std::vector<std::vector<std::unique_ptr<RegTree> > > new_trees;
  const int ngroup = model_.param.num_output_group;
  ConfigureWithKnownData(this->cfg_, p_fmat);
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
                                       GradientPair(),
                                       in_gpair->DeviceIdx());
    const auto& gpair_h = in_gpair->ConstHostVector();
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
  this->CommitModel(std::move(new_trees));
}

void GBTree::InitUpdater(Args const& cfg) {
  std::string tval = tparam_.updater_seq;
  std::vector<std::string> ups = common::Split(tval, ',');

  if (updaters_.size() != 0) {
    // Assert we have a valid set of updaters.
    CHECK_EQ(ups.size(), updaters_.size());
    for (auto const& up : updaters_) {
      bool contains = std::any_of(ups.cbegin(), ups.cend(),
                        [&up](std::string const& name) {
                          return name == up->Name();
                        });
      if (!contains) {
        std::stringstream ss;
        ss << "Internal Error: " << " mismatched updater sequence.\n";
        ss << "Specified updaters: ";
        std::for_each(ups.cbegin(), ups.cend(),
                      [&ss](std::string const& name){
                        ss << name << " ";
                      });
        ss << "\n" << "Actual updaters: ";
        std::for_each(updaters_.cbegin(), updaters_.cend(),
                      [&ss](std::unique_ptr<TreeUpdater> const& updater){
                        ss << updater->Name() << " ";
                      });
        LOG(FATAL) << ss.str();
      }
    }
    return;
  }

  for (const std::string& pstr : ups) {
    std::unique_ptr<TreeUpdater> up(TreeUpdater::Create(pstr.c_str(), learner_param_));
    up->Configure(cfg);
    updaters_.push_back(std::move(up));
  }
}

void GBTree::BoostNewTrees(HostDeviceVector<GradientPair>* gpair,
                           DMatrix *p_fmat,
                           int bst_group,
                           std::vector<std::unique_ptr<RegTree> >* ret) {
  std::vector<RegTree*> new_trees;
  ret->clear();
  // create the trees
  for (int i = 0; i < tparam_.num_parallel_tree; ++i) {
    if (tparam_.process_type == TreeProcessType::kDefault) {
      // create new tree
      std::unique_ptr<RegTree> ptr(new RegTree());
      ptr->param.InitAllowUnknown(this->cfg_);
      new_trees.push_back(ptr.get());
      ret->push_back(std::move(ptr));
    } else if (tparam_.process_type == TreeProcessType::kUpdate) {
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

void GBTree::CommitModel(std::vector<std::vector<std::unique_ptr<RegTree>>>&& new_trees) {
  monitor_.Start("CommitModel");
  int num_new_trees = 0;
  for (int gid = 0; gid < model_.param.num_output_group; ++gid) {
    num_new_trees += new_trees[gid].size();
    model_.CommitModel(std::move(new_trees[gid]), gid);
  }
  CHECK(configured_);
  GetPredictor()->UpdatePredictionCache(model_, &updaters_, num_new_trees);
  monitor_.Stop("CommitModel");
}


// dart
class Dart : public GBTree {
 public:
  explicit Dart(bst_float base_margin) : GBTree(base_margin) {}

  void Configure(const Args& cfg) override {
    GBTree::Configure(cfg);
    if (model_.trees.size() == 0) {
      dparam_.UpdateAllowUnknown(cfg);
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

  bool UseGPU() const override {
    return GBTree::UseGPU();
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
      const auto& base_margin =
        p_fmat->Info().base_margin_.ConstHostVector();
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
    auto* self = static_cast<Derived*>(this);
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
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
    LOG(INFO) << "drop " << num_drop << " trees, "
              << "weight = " << weight_drop_.back();
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
