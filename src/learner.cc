/*!
 * Copyright 2014 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <xgboost/learner.h>
#include <xgboost/logging.h>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "./common/common.h"
#include "./common/io.h"
#include "./common/random.h"

namespace xgboost {
// implementation of base learner.
bool Learner::AllowLazyCheckPoint() const {
  return gbm_->AllowLazyCheckPoint();
}

std::vector<std::string> Learner::DumpModel(const FeatureMap& fmap,
                                            bool with_stats,
                                            std::string format) const {
  return gbm_->DumpModel(fmap, with_stats, format);
}

/*! \brief training parameter for regression */
struct LearnerModelParam : public dmlc::Parameter<LearnerModelParam> {
  /* \brief global bias */
  bst_float base_score;
  /* \brief number of features  */
  unsigned num_feature;
  /* \brief number of classes, if it is multi-class classification  */
  int num_class;
  /*! \brief Model contain additional properties */
  int contain_extra_attrs;
  /*! \brief Model contain eval metrics */
  int contain_eval_metrics;
  /*! \brief reserved field */
  int reserved[29];
  /*! \brief constructor */
  LearnerModelParam() {
    std::memset(this, 0, sizeof(LearnerModelParam));
    base_score = 0.5f;
  }
  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerModelParam) {
    DMLC_DECLARE_FIELD(base_score)
        .set_default(0.5f)
        .describe("Global bias of the model.");
    DMLC_DECLARE_FIELD(num_feature)
        .set_default(0)
        .describe(
            "Number of features in training data,"
            " this parameter will be automatically detected by learner.");
    DMLC_DECLARE_FIELD(num_class).set_default(0).set_lower_bound(0).describe(
        "Number of class option for multi-class classifier. "
        " By default equals 0 and corresponds to binary classifier.");
  }
};

struct LearnerTrainParam : public dmlc::Parameter<LearnerTrainParam> {
  // stored random seed
  int seed;
  // whether seed the PRNG each iteration
  bool seed_per_iteration;
  // data split mode, can be row, col, or none.
  int dsplit;
  // tree construction method
  int tree_method;
  // internal test flag
  std::string test_flag;
  // maximum buffered row value
  float prob_buffer_row;
  // maximum row per batch.
  size_t max_row_perbatch;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  // flag to print out detailed breakdown of runtime
  int debug_verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerTrainParam) {
    DMLC_DECLARE_FIELD(seed).set_default(0).describe(
        "Random number seed during training.");
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe(
            "Seed PRNG determnisticly via iterator number, "
            "this option will be switched on automatically on distributed "
            "mode.");
    DMLC_DECLARE_FIELD(dsplit)
        .set_default(0)
        .add_enum("auto", 0)
        .add_enum("col", 1)
        .add_enum("row", 2)
        .describe("Data split mode for distributed training.");
    DMLC_DECLARE_FIELD(tree_method)
        .set_default(0)
        .add_enum("auto", 0)
        .add_enum("approx", 1)
        .add_enum("exact", 2)
        .add_enum("hist", 3)
        .add_enum("gpu_exact", 4)
        .add_enum("gpu_hist", 5)
        .add_enum("gpu_hist_experimental", 6)
        .describe("Choice of tree construction method.");
    DMLC_DECLARE_FIELD(test_flag).set_default("").describe(
        "Internal test flag");
    DMLC_DECLARE_FIELD(prob_buffer_row)
        .set_default(1.0f)
        .set_range(0.0f, 1.0f)
        .describe("Maximum buffered row portion");
    DMLC_DECLARE_FIELD(max_row_perbatch)
        .set_default(std::numeric_limits<size_t>::max())
        .describe("maximum row per batch.");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
  }
};

DMLC_REGISTER_PARAMETER(LearnerModelParam);
DMLC_REGISTER_PARAMETER(LearnerTrainParam);

/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
class LearnerImpl : public Learner {
 public:
  explicit LearnerImpl(const std::vector<std::shared_ptr<DMatrix> >& cache)
      : cache_(cache) {
    // boosted tree
    name_obj_ = "reg:linear";
    name_gbm_ = "gbtree";
  }

  void ConfigureUpdaters() {
    if (tparam.tree_method == 0 || tparam.tree_method == 1 ||
        tparam.tree_method == 2) {
      if (cfg_.count("updater") == 0) {
        if (tparam.dsplit == 1) {
          cfg_["updater"] = "distcol";
        } else if (tparam.dsplit == 2) {
          cfg_["updater"] = "grow_histmaker,prune";
        }
        if (tparam.prob_buffer_row != 1.0f) {
          cfg_["updater"] = "grow_histmaker,refresh,prune";
        }
      }
    } else if (tparam.tree_method == 3) {
      /* histogram-based algorithm */
      LOG(CONSOLE) << "Tree method is selected to be \'hist\', which uses a "
                      "single updater "
                   << "grow_fast_histmaker.";
      cfg_["updater"] = "grow_fast_histmaker";
    } else if (tparam.tree_method == 4) {
      if (cfg_.count("updater") == 0) {
        cfg_["updater"] = "grow_gpu,prune";
      }
      if (cfg_.count("predictor") == 0) {
        cfg_["predictor"] = "gpu_predictor";
      }
    } else if (tparam.tree_method == 5) {
      if (cfg_.count("updater") == 0) {
        cfg_["updater"] = "grow_gpu_hist";
      }
      if (cfg_.count("predictor") == 0) {
        cfg_["predictor"] = "gpu_predictor";
      }
    } else if (tparam.tree_method == 6) {
      if (cfg_.count("updater") == 0) {
        cfg_["updater"] = "grow_gpu_hist_experimental,prune";
      }
      if (cfg_.count("predictor") == 0) {
        cfg_["predictor"] = "gpu_predictor";
      }
    }
  }

  void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) override {
    // add to configurations
    tparam.InitAllowUnknown(args);
    cfg_.clear();
    for (const auto& kv : args) {
      if (kv.first == "eval_metric") {
        // check duplication
        auto dup_check = [&kv](const std::unique_ptr<Metric>& m) {
          return m->Name() != kv.second;
        };
        if (std::all_of(metrics_.begin(), metrics_.end(), dup_check)) {
          metrics_.emplace_back(Metric::Create(kv.second));
          mparam.contain_eval_metrics = 1;
        }
      } else {
        cfg_[kv.first] = kv.second;
      }
    }
    if (tparam.nthread != 0) {
      omp_set_num_threads(tparam.nthread);
    }

    // add additional parameters
    // These are cosntraints that need to be satisfied.
    if (tparam.dsplit == 0 && rabit::IsDistributed()) {
      tparam.dsplit = 2;
    }

    if (cfg_.count("num_class") != 0) {
      cfg_["num_output_group"] = cfg_["num_class"];
      if (atoi(cfg_["num_class"].c_str()) > 1 && cfg_.count("objective") == 0) {
        cfg_["objective"] = "multi:softmax";
      }
    }

    if (cfg_.count("max_delta_step") == 0 && cfg_.count("objective") != 0 &&
        cfg_["objective"] == "count:poisson") {
      cfg_["max_delta_step"] = "0.7";
    }

    ConfigureUpdaters();

    if (cfg_.count("objective") == 0) {
      cfg_["objective"] = "reg:linear";
    }
    if (cfg_.count("booster") == 0) {
      cfg_["booster"] = "gbtree";
    }

    if (!this->ModelInitialized()) {
      mparam.InitAllowUnknown(args);
      name_obj_ = cfg_["objective"];
      name_gbm_ = cfg_["booster"];
      // set seed only before the model is initialized
      common::GlobalRandom().seed(tparam.seed);
    }

    // set number of features correctly.
    cfg_["num_feature"] = common::ToString(mparam.num_feature);
    cfg_["num_class"] = common::ToString(mparam.num_class);

    if (gbm_.get() != nullptr) {
      gbm_->Configure(cfg_.begin(), cfg_.end());
    }
    if (obj_.get() != nullptr) {
      obj_->Configure(cfg_.begin(), cfg_.end());
    }
  }

  void InitModel() override { this->LazyInitModel(); }

  void Load(dmlc::Stream* fi) override {
    // TODO(tqchen) mark deprecation of old format.
    common::PeekableInStream fp(fi);
    // backward compatible header check.
    std::string header;
    header.resize(4);
    if (fp.PeekRead(&header[0], 4) == 4) {
      CHECK_NE(header, "bs64")
          << "Base64 format is no longer supported in brick.";
      if (header == "binf") {
        CHECK_EQ(fp.Read(&header[0], 4), 4U);
      }
    }
    // use the peekable reader.
    fi = &fp;
    // read parameter
    CHECK_EQ(fi->Read(&mparam, sizeof(mparam)), sizeof(mparam))
        << "BoostLearner: wrong model format";
    {
      // backward compatibility code for compatible with old model type
      // for new model, Read(&name_obj_) is suffice
      uint64_t len;
      CHECK_EQ(fi->Read(&len, sizeof(len)), sizeof(len));
      if (len >= std::numeric_limits<unsigned>::max()) {
        int gap;
        CHECK_EQ(fi->Read(&gap, sizeof(gap)), sizeof(gap))
            << "BoostLearner: wrong model format";
        len = len >> static_cast<uint64_t>(32UL);
      }
      if (len != 0) {
        name_obj_.resize(len);
        CHECK_EQ(fi->Read(&name_obj_[0], len), len)
            << "BoostLearner: wrong model format";
      }
    }
    CHECK(fi->Read(&name_gbm_)) << "BoostLearner: wrong model format";
    // duplicated code with LazyInitModel
    obj_.reset(ObjFunction::Create(name_obj_));
    gbm_.reset(GradientBooster::Create(name_gbm_, cache_, mparam.base_score));
    gbm_->Load(fi);
    if (mparam.contain_extra_attrs != 0) {
      std::vector<std::pair<std::string, std::string> > attr;
      fi->Read(&attr);
      attributes_ =
          std::map<std::string, std::string>(attr.begin(), attr.end());
    }
    if (name_obj_ == "count:poisson") {
      std::string max_delta_step;
      fi->Read(&max_delta_step);
      cfg_["max_delta_step"] = max_delta_step;
    }
    if (mparam.contain_eval_metrics != 0) {
      std::vector<std::string> metr;
      fi->Read(&metr);
      for (auto name : metr) {
        metrics_.emplace_back(Metric::Create(name));
      }
    }
    cfg_["num_class"] = common::ToString(mparam.num_class);
    cfg_["num_feature"] = common::ToString(mparam.num_feature);
    obj_->Configure(cfg_.begin(), cfg_.end());
  }

  // rabit save model to rabit checkpoint
  void Save(dmlc::Stream* fo) const override {
    fo->Write(&mparam, sizeof(LearnerModelParam));
    fo->Write(name_obj_);
    fo->Write(name_gbm_);
    gbm_->Save(fo);
    if (mparam.contain_extra_attrs != 0) {
      std::vector<std::pair<std::string, std::string> > attr(
          attributes_.begin(), attributes_.end());
      fo->Write(attr);
    }
    if (name_obj_ == "count:poisson") {
      std::map<std::string, std::string>::const_iterator it =
          cfg_.find("max_delta_step");
      if (it != cfg_.end()) fo->Write(it->second);
    }
    if (mparam.contain_eval_metrics != 0) {
      std::vector<std::string> metr;
      for (auto& ev : metrics_) {
        metr.emplace_back(ev->Name());
      }
      fo->Write(metr);
    }
  }

  void UpdateOneIter(int iter, DMatrix* train) override {
    CHECK(ModelInitialized())
        << "Always call InitModel or LoadModel before update";
    if (tparam.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(tparam.seed * kRandSeedMagic + iter);
    }
    this->LazyInitDMatrix(train);
    this->PredictRaw(train, &preds_);
    obj_->GetGradient(preds_, train->info(), iter, &gpair_);
    gbm_->DoBoost(train, &gpair_, obj_.get());
  }

  void BoostOneIter(int iter, DMatrix* train,
                    std::vector<bst_gpair>* in_gpair) override {
    if (tparam.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(tparam.seed * kRandSeedMagic + iter);
    }
    this->LazyInitDMatrix(train);
    gbm_->DoBoost(train, in_gpair);
  }

  std::string EvalOneIter(int iter, const std::vector<DMatrix*>& data_sets,
                          const std::vector<std::string>& data_names) override {
    double tstart = dmlc::GetTime();
    std::ostringstream os;
    os << '[' << iter << ']' << std::setiosflags(std::ios::fixed);
    if (metrics_.size() == 0) {
      metrics_.emplace_back(Metric::Create(obj_->DefaultEvalMetric()));
    }
    for (size_t i = 0; i < data_sets.size(); ++i) {
      this->PredictRaw(data_sets[i], &preds_);
      obj_->EvalTransform(&preds_);
      for (auto& ev : metrics_) {
        os << '\t' << data_names[i] << '-' << ev->Name() << ':'
           << ev->Eval(preds_, data_sets[i]->info(), tparam.dsplit == 2);
      }
    }

    if (tparam.debug_verbose > 0) {
      LOG(INFO) << "EvalOneIter(): " << dmlc::GetTime() - tstart << " sec";
    }
    return os.str();
  }

  void SetAttr(const std::string& key, const std::string& value) override {
    attributes_[key] = value;
    mparam.contain_extra_attrs = 1;
  }

  bool GetAttr(const std::string& key, std::string* out) const override {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) return false;
    *out = it->second;
    return true;
  }

  bool DelAttr(const std::string& key) override {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) return false;
    attributes_.erase(it);
    return true;
  }

  std::vector<std::string> GetAttrNames() const override {
    std::vector<std::string> out;
    out.reserve(attributes_.size());
    for (auto& p : attributes_) {
      out.push_back(p.first);
    }
    return out;
  }

  std::pair<std::string, bst_float> Evaluate(DMatrix* data,
                                             std::string metric) {
    if (metric == "auto") metric = obj_->DefaultEvalMetric();
    std::unique_ptr<Metric> ev(Metric::Create(metric.c_str()));
    this->PredictRaw(data, &preds_);
    obj_->EvalTransform(&preds_);
    return std::make_pair(metric,
                          ev->Eval(preds_, data->info(), tparam.dsplit == 2));
  }

  void Predict(DMatrix* data, bool output_margin,
               std::vector<bst_float>* out_preds, unsigned ntree_limit,
               bool pred_leaf, bool pred_contribs, bool approx_contribs) const override {
    if (pred_contribs) {
      gbm_->PredictContribution(data, out_preds, ntree_limit, approx_contribs);
    } else if (pred_leaf) {
      gbm_->PredictLeaf(data, out_preds, ntree_limit);
    } else {
      this->PredictRaw(data, out_preds, ntree_limit);
      if (!output_margin) {
        obj_->PredTransform(out_preds);
      }
    }
  }

 protected:
  // check if p_train is ready to used by training.
  // if not, initialize the column access.
  inline void LazyInitDMatrix(DMatrix* p_train) {
    if (tparam.tree_method == 3 || tparam.tree_method == 4 ||
        tparam.tree_method == 5) {
      return;
    }

    if (!p_train->HaveColAccess()) {
      int ncol = static_cast<int>(p_train->info().num_col);
      std::vector<bool> enabled(ncol, true);
      // set max row per batch to limited value
      // in distributed mode, use safe choice otherwise
      size_t max_row_perbatch = tparam.max_row_perbatch;
      const size_t safe_max_row = static_cast<size_t>(32UL << 10UL);

      if (tparam.tree_method == 0 && p_train->info().num_row >= (4UL << 20UL)) {
        LOG(CONSOLE)
            << "Tree method is automatically selected to be \'approx\'"
            << " for faster speed."
            << " to use old behavior(exact greedy algorithm on single machine),"
            << " set tree_method to \'exact\'";
        max_row_perbatch = std::min(max_row_perbatch, safe_max_row);
      }

      if (tparam.tree_method == 1) {
        LOG(CONSOLE) << "Tree method is selected to be \'approx\'";
        max_row_perbatch = std::min(max_row_perbatch, safe_max_row);
      }

      if (tparam.test_flag == "block" || tparam.dsplit == 2) {
        max_row_perbatch = std::min(max_row_perbatch, safe_max_row);
      }
      // initialize column access
      p_train->InitColAccess(enabled, tparam.prob_buffer_row, max_row_perbatch);
    }

    if (!p_train->SingleColBlock() && cfg_.count("updater") == 0) {
      if (tparam.tree_method == 2) {
        LOG(CONSOLE) << "tree method is set to be 'exact',"
                     << " but currently we are only able to proceed with "
                        "approximate algorithm";
      }
      cfg_["updater"] = "grow_histmaker,prune";
      if (gbm_.get() != nullptr) {
        gbm_->Configure(cfg_.begin(), cfg_.end());
      }
    }
  }

  // return whether model is already initialized.
  inline bool ModelInitialized() const { return gbm_.get() != nullptr; }
  // lazily initialize the model if it haven't yet been initialized.
  inline void LazyInitModel() {
    if (this->ModelInitialized()) return;
    // estimate feature bound
    unsigned num_feature = 0;
    for (size_t i = 0; i < cache_.size(); ++i) {
      CHECK(cache_[i] != nullptr);
      num_feature = std::max(num_feature,
                             static_cast<unsigned>(cache_[i]->info().num_col));
    }
    // run allreduce on num_feature to find the maximum value
    rabit::Allreduce<rabit::op::Max>(&num_feature, 1);
    if (num_feature > mparam.num_feature) {
      mparam.num_feature = num_feature;
    }
    // setup
    cfg_["num_feature"] = common::ToString(mparam.num_feature);
    CHECK(obj_.get() == nullptr && gbm_.get() == nullptr);
    obj_.reset(ObjFunction::Create(name_obj_));
    obj_->Configure(cfg_.begin(), cfg_.end());
    // reset the base score
    mparam.base_score = obj_->ProbToMargin(mparam.base_score);
    gbm_.reset(GradientBooster::Create(name_gbm_, cache_, mparam.base_score));
    gbm_->Configure(cfg_.begin(), cfg_.end());
  }
  /*!
   * \brief get un-transformed prediction
   * \param data training data matrix
   * \param out_preds output vector that stores the prediction
   * \param ntree_limit limit number of trees used for boosted tree
   *   predictor, when it equals 0, this means we are using all the trees
   */
  inline void PredictRaw(DMatrix* data, std::vector<bst_float>* out_preds,
                         unsigned ntree_limit = 0) const {
    CHECK(gbm_.get() != nullptr)
        << "Predict must happen after Load or InitModel";
    gbm_->PredictBatch(data, out_preds, ntree_limit);
  }
  // model parameter
  LearnerModelParam mparam;
  // training parameter
  LearnerTrainParam tparam;
  // configurations
  std::map<std::string, std::string> cfg_;
  // attributes
  std::map<std::string, std::string> attributes_;
  // name of gbm
  std::string name_gbm_;
  // name of objective function
  std::string name_obj_;
  // temporal storages for prediction
  std::vector<bst_float> preds_;
  // gradient pairs
  std::vector<bst_gpair> gpair_;

 private:
  /*! \brief random number transformation seed. */
  static const int kRandSeedMagic = 127;
  // internal cached dmatrix
  std::vector<std::shared_ptr<DMatrix> > cache_;
};

Learner* Learner::Create(
    const std::vector<std::shared_ptr<DMatrix> >& cache_data) {
  return new LearnerImpl(cache_data);
}
}  // namespace xgboost
