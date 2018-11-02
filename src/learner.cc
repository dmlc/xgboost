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
#include <ios>
#include <utility>
#include <vector>
#include "./common/common.h"
#include "./common/host_device_vector.h"
#include "./common/io.h"
#include "./common/random.h"
#include "./common/enum_class_param.h"
#include "./common/timer.h"

namespace {

const char* kMaxDeltaStepDefaultValue = "0.7";

enum class TreeMethod : int {
  kAuto = 0, kApprox = 1, kExact = 2, kHist = 3,
  kGPUExact = 4, kGPUHist = 5
};

enum class DataSplitMode : int {
  kAuto = 0, kCol = 1, kRow = 2
};

inline bool IsFloat(const std::string& str) {
  std::stringstream ss(str);
  float f;
  return !((ss >> std::noskipws >> f).rdstate() ^ std::ios_base::eofbit);
}

inline bool IsInt(const std::string& str) {
  std::stringstream ss(str);
  int i;
  return !((ss >> std::noskipws >> i).rdstate() ^ std::ios_base::eofbit);
}

inline std::string RenderParamVal(const std::string& str) {
  if (IsFloat(str) || IsInt(str)) {
    return str;
  } else {
    return std::string("'") + str + "'";
  }
}

}  // anonymous namespace

DECLARE_FIELD_ENUM_CLASS(TreeMethod);
DECLARE_FIELD_ENUM_CLASS(DataSplitMode);

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
  DataSplitMode dsplit;
  // tree construction method
  TreeMethod tree_method;
  // internal test flag
  std::string test_flag;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  // flag to print out detailed breakdown of runtime
  int debug_verbose;
  // flag to disable default metric
  int disable_default_eval_metric;
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
        .set_default(DataSplitMode::kAuto)
        .add_enum("auto", DataSplitMode::kAuto)
        .add_enum("col", DataSplitMode::kCol)
        .add_enum("row", DataSplitMode::kRow)
        .describe("Data split mode for distributed training.");
    DMLC_DECLARE_FIELD(tree_method)
        .set_default(TreeMethod::kAuto)
        .add_enum("auto", TreeMethod::kAuto)
        .add_enum("approx", TreeMethod::kApprox)
        .add_enum("exact", TreeMethod::kExact)
        .add_enum("hist", TreeMethod::kHist)
        .add_enum("gpu_exact", TreeMethod::kGPUExact)
        .add_enum("gpu_hist", TreeMethod::kGPUHist)
        .describe("Choice of tree construction method.");
    DMLC_DECLARE_FIELD(test_flag).set_default("").describe(
        "Internal test flag");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    DMLC_DECLARE_FIELD(disable_default_eval_metric)
        .set_default(0)
        .describe("flag to disable default metric. Set to >0 to disable");
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
  explicit LearnerImpl(std::vector<std::shared_ptr<DMatrix> >  cache)
      : cache_(std::move(cache)) {
    // boosted tree
    name_obj_ = "reg:linear";
    name_gbm_ = "gbtree";
  }

  static void AssertGPUSupport() {
#ifndef XGBOOST_USE_CUDA
    LOG(FATAL) << "XGBoost version not compiled with GPU support.";
#endif
  }

  void ConfigureUpdaters() {
    /* Choose updaters according to tree_method parameters */
    if (cfg_.count("updater") > 0) {
      LOG(CONSOLE) << "DANGER AHEAD: You have manually specified `updater` "
                      "parameter. The `tree_method` parameter will be ignored. "
                      "Incorrect sequence of updaters will produce undefined "
                      "behavior. For common uses, we recommend using "
                      "`tree_method` parameter instead.";
      return;
    }

    switch (tparam_.tree_method) {
     case TreeMethod::kAuto:
      // Use heuristic to choose between 'exact' and 'approx'
      // This choice is deferred to PerformTreeMethodHeuristic().
      break;
     case TreeMethod::kApprox:
      cfg_["updater"] = "grow_histmaker,prune";
      break;
     case TreeMethod::kExact:
      cfg_["updater"] = "grow_colmaker,prune";
      break;
     case TreeMethod::kHist:
      LOG(CONSOLE) << "Tree method is selected to be 'hist', which uses a "
                      "single updater grow_fast_histmaker.";
      cfg_["updater"] = "grow_fast_histmaker";
      break;
     case TreeMethod::kGPUExact:
      this->AssertGPUSupport();
      cfg_["updater"] = "grow_gpu,prune";
      if (cfg_.count("predictor") == 0) {
        cfg_["predictor"] = "gpu_predictor";
      }
      break;
     case TreeMethod::kGPUHist:
      this->AssertGPUSupport();
      cfg_["updater"] = "grow_gpu_hist";
      if (cfg_.count("predictor") == 0) {
        cfg_["predictor"] = "gpu_predictor";
      }
      break;
     default:
      LOG(FATAL) << "Unknown tree_method ("
                 << static_cast<int>(tparam_.tree_method) << ") detected";
    }
  }

  void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) override {
    // add to configurations
    tparam_.InitAllowUnknown(args);
    monitor_.Init("Learner", tparam_.debug_verbose);
    cfg_.clear();
    for (const auto& kv : args) {
      if (kv.first == "eval_metric") {
        // check duplication
        auto dup_check = [&kv](const std::unique_ptr<Metric>& m) {
          return m->Name() != kv.second;
        };
        if (std::all_of(metrics_.begin(), metrics_.end(), dup_check)) {
          metrics_.emplace_back(Metric::Create(kv.second));
          mparam_.contain_eval_metrics = 1;
        }
      } else {
        cfg_[kv.first] = kv.second;
      }
    }
    if (tparam_.nthread != 0) {
      omp_set_num_threads(tparam_.nthread);
    }

    // add additional parameters
    // These are cosntraints that need to be satisfied.
    if (tparam_.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      tparam_.dsplit = DataSplitMode::kRow;
    }

    if (cfg_.count("num_class") != 0) {
      cfg_["num_output_group"] = cfg_["num_class"];
      if (atoi(cfg_["num_class"].c_str()) > 1 && cfg_.count("objective") == 0) {
        cfg_["objective"] = "multi:softmax";
      }
    }

    if (cfg_.count("max_delta_step") == 0 && cfg_.count("objective") != 0 &&
        cfg_["objective"] == "count:poisson") {
      cfg_["max_delta_step"] = kMaxDeltaStepDefaultValue;
    }

    ConfigureUpdaters();

    if (cfg_.count("objective") == 0) {
      cfg_["objective"] = "reg:linear";
    }
    if (cfg_.count("booster") == 0) {
      cfg_["booster"] = "gbtree";
    }

    if (!this->ModelInitialized()) {
      mparam_.InitAllowUnknown(args);
      name_obj_ = cfg_["objective"];
      name_gbm_ = cfg_["booster"];
      // set seed only before the model is initialized
      common::GlobalRandom().seed(tparam_.seed);
    }

    // set number of features correctly.
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);
    cfg_["num_class"] = common::ToString(mparam_.num_class);

    if (gbm_ != nullptr) {
      gbm_->Configure(cfg_.begin(), cfg_.end());
    }
    if (obj_ != nullptr) {
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
    CHECK_EQ(fi->Read(&mparam_, sizeof(mparam_)), sizeof(mparam_))
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
    gbm_.reset(GradientBooster::Create(name_gbm_, cache_, mparam_.base_score));
    gbm_->Load(fi);
    if (mparam_.contain_extra_attrs != 0) {
      std::vector<std::pair<std::string, std::string> > attr;
      fi->Read(&attr);
      for (auto& kv : attr) {
        // Load `predictor`, `n_gpus`, `gpu_id` parameters from extra attributes
        const std::string prefix = "SAVED_PARAM_";
        if (kv.first.find(prefix) == 0) {
          const std::string saved_param = kv.first.substr(prefix.length());
#ifdef XGBOOST_USE_CUDA
          if (saved_param == "predictor" || saved_param == "n_gpus"
              || saved_param == "gpu_id") {
            cfg_[saved_param] = kv.second;
            LOG(INFO)
              << "Parameter '" << saved_param << "' has been recovered from "
              << "the saved model. It will be set to "
              << RenderParamVal(kv.second) << " for prediction. To "
              << "override the predictor behavior, explicitly set '"
              << saved_param << "' parameter as follows:\n"
              << "  * Python package: bst.set_param('"
              << saved_param << "', [new value])\n"
              << "  * R package:      xgb.parameters(bst) <- list("
              << saved_param << " = [new value])\n"
              << "  * JVM packages:   bst.setParam(\""
              << saved_param << "\", [new value])";
          }
#else
          if (saved_param == "predictor" && kv.second == "gpu_predictor") {
            LOG(INFO) << "Parameter 'predictor' will be set to 'cpu_predictor' "
                      << "since XGBoots wasn't compiled with GPU support.";
            cfg_["predictor"] = "cpu_predictor";
            kv.second = "cpu_predictor";
          }
#endif
        }
      }
      attributes_ =
          std::map<std::string, std::string>(attr.begin(), attr.end());
    }
    if (name_obj_ == "count:poisson") {
      std::string max_delta_step;
      fi->Read(&max_delta_step);
      cfg_["max_delta_step"] = max_delta_step;
    }
    if (mparam_.contain_eval_metrics != 0) {
      std::vector<std::string> metr;
      fi->Read(&metr);
      for (auto name : metr) {
        metrics_.emplace_back(Metric::Create(name));
      }
    }
    cfg_["num_class"] = common::ToString(mparam_.num_class);
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);
    obj_->Configure(cfg_.begin(), cfg_.end());
  }

  // rabit save model to rabit checkpoint
  void Save(dmlc::Stream* fo) const override {
    LearnerModelParam mparam = mparam_;  // make a copy to potentially modify
    std::vector<std::pair<std::string, std::string> > extra_attr;
      // extra attributed to be added just before saving

    if (name_obj_ == "count:poisson") {
      auto it = cfg_.find("max_delta_step");
      if (it != cfg_.end()) {
        // write `max_delta_step` parameter as extra attribute of booster
        mparam.contain_extra_attrs = 1;
        extra_attr.emplace_back("count_poisson_max_delta_step", it->second);
      }
    }
    {
      // Write `predictor`, `n_gpus`, `gpu_id` parameters as extra attributes
      for (const auto& key : std::vector<std::string>{
                                   "predictor", "n_gpus", "gpu_id"}) {
        auto it = cfg_.find(key);
        if (it != cfg_.end()) {
          mparam.contain_extra_attrs = 1;
          extra_attr.emplace_back("SAVED_PARAM_" + key, it->second);
        }
      }
    }
    fo->Write(&mparam, sizeof(LearnerModelParam));
    fo->Write(name_obj_);
    fo->Write(name_gbm_);
    gbm_->Save(fo);
    if (mparam.contain_extra_attrs != 0) {
      std::map<std::string, std::string> attr(attributes_);
      for (const auto& kv : extra_attr) {
        attr[kv.first] = kv.second;
      }
      fo->Write(std::vector<std::pair<std::string, std::string>>(
                  attr.begin(), attr.end()));
    }
    if (name_obj_ == "count:poisson") {
      auto it = cfg_.find("max_delta_step");
      if (it != cfg_.end()) {
        fo->Write(it->second);
      } else {
        // recover value of max_delta_step from extra attributes
        auto it2 = attributes_.find("count_poisson_max_delta_step");
        const std::string max_delta_step
          = (it2 != attributes_.end()) ? it2->second : kMaxDeltaStepDefaultValue;
        fo->Write(max_delta_step);
      }
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
    monitor_.Start("UpdateOneIter");
    CHECK(ModelInitialized())
        << "Always call InitModel or LoadModel before update";
    if (tparam_.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(tparam_.seed * kRandSeedMagic + iter);
    }
    this->PerformTreeMethodHeuristic(train);
    monitor_.Start("PredictRaw");
    this->PredictRaw(train, &preds_);
    monitor_.Stop("PredictRaw");
    monitor_.Start("GetGradient");
    obj_->GetGradient(preds_, train->Info(), iter, &gpair_);
    monitor_.Stop("GetGradient");
    gbm_->DoBoost(train, &gpair_, obj_.get());
    monitor_.Stop("UpdateOneIter");
  }

  void BoostOneIter(int iter, DMatrix* train,
                    HostDeviceVector<GradientPair>* in_gpair) override {
    monitor_.Start("BoostOneIter");
    if (tparam_.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(tparam_.seed * kRandSeedMagic + iter);
    }
    this->PerformTreeMethodHeuristic(train);
    gbm_->DoBoost(train, in_gpair);
    monitor_.Stop("BoostOneIter");
  }

  std::string EvalOneIter(int iter, const std::vector<DMatrix*>& data_sets,
                          const std::vector<std::string>& data_names) override {
    monitor_.Start("EvalOneIter");
    std::ostringstream os;
    os << '[' << iter << ']' << std::setiosflags(std::ios::fixed);
    if (metrics_.size() == 0 && tparam_.disable_default_eval_metric <= 0) {
      metrics_.emplace_back(Metric::Create(obj_->DefaultEvalMetric()));
    }
    for (size_t i = 0; i < data_sets.size(); ++i) {
      this->PredictRaw(data_sets[i], &preds_);
      obj_->EvalTransform(&preds_);
      for (auto& ev : metrics_) {
        os << '\t' << data_names[i] << '-' << ev->Name() << ':'
           << ev->Eval(preds_.ConstHostVector(), data_sets[i]->Info(),
                       tparam_.dsplit == DataSplitMode::kRow);
      }
    }

    monitor_.Stop("EvalOneIter");
    return os.str();
  }

  void SetAttr(const std::string& key, const std::string& value) override {
    attributes_[key] = value;
    mparam_.contain_extra_attrs = 1;
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
                          ev->Eval(preds_.ConstHostVector(), data->Info(),
                                   tparam_.dsplit == DataSplitMode::kRow));
  }

  void Predict(DMatrix* data, bool output_margin,
               HostDeviceVector<bst_float>* out_preds, unsigned ntree_limit,
               bool pred_leaf, bool pred_contribs, bool approx_contribs,
               bool pred_interactions) const override {
    if (pred_contribs) {
      gbm_->PredictContribution(data, &out_preds->HostVector(), ntree_limit, approx_contribs);
    } else if (pred_interactions) {
      gbm_->PredictInteractionContributions(data, &out_preds->HostVector(), ntree_limit,
                                            approx_contribs);
    } else if (pred_leaf) {
      gbm_->PredictLeaf(data, &out_preds->HostVector(), ntree_limit);
    } else {
      this->PredictRaw(data, out_preds, ntree_limit);
      if (!output_margin) {
        obj_->PredTransform(out_preds);
      }
    }
  }

  const std::map<std::string, std::string>& GetConfigurationArguments() const override {
    return cfg_;
  }

 protected:
  // Revise `tree_method` and `updater` parameters after seeing the training
  // data matrix
  inline void PerformTreeMethodHeuristic(DMatrix* p_train) {
    if (name_gbm_ != "gbtree" || cfg_.count("updater") > 0) {
      // 1. This method is not applicable for non-tree learners
      // 2. This method is disabled when `updater` parameter is explicitly
      //    set, since only experts are expected to do so.
      return;
    }

    const TreeMethod current_tree_method = tparam_.tree_method;
    if (rabit::IsDistributed()) {
      /* Choose tree_method='approx' when distributed training is activated */
      CHECK(tparam_.dsplit != DataSplitMode::kAuto)
        << "Precondition violated; dsplit cannot be 'auto' in distributed mode";
      if (tparam_.dsplit == DataSplitMode::kCol) {
        // 'distcol' updater hidden until it becomes functional again
        // See discussion at https://github.com/dmlc/xgboost/issues/1832
        LOG(FATAL) << "Column-wise data split is currently not supported.";
      }
      switch (current_tree_method) {
       case TreeMethod::kAuto:
        LOG(CONSOLE) << "Tree method is automatically selected to be 'approx' "
                        "for distributed training.";
        break;
       case TreeMethod::kApprox:
        // things are okay, do nothing
        break;
       case TreeMethod::kExact:
       case TreeMethod::kHist:
        LOG(CONSOLE) << "Tree method was set to be '"
                     << (current_tree_method == TreeMethod::kExact ?
                        "exact" : "hist")
                     << "', but only 'approx' is available for distributed "
                        "training. The `tree_method` parameter is now being "
                        "changed to 'approx'";
        break;
       case TreeMethod::kGPUExact:
       case TreeMethod::kGPUHist:
        LOG(FATAL) << "Distributed training is not available with GPU algoritms";
        break;
       default:
        LOG(FATAL) << "Unknown tree_method ("
                   << static_cast<int>(current_tree_method) << ") detected";
      }
      tparam_.tree_method = TreeMethod::kApprox;
    } else if (!p_train->SingleColBlock()) {
      /* Some tree methods are not available for external-memory DMatrix */
      switch (current_tree_method) {
       case TreeMethod::kAuto:
        LOG(CONSOLE) << "Tree method is automatically set to 'approx' "
                        "since external-memory data matrix is used.";
        break;
       case TreeMethod::kApprox:
        // things are okay, do nothing
        break;
       case TreeMethod::kExact:
        LOG(CONSOLE) << "Tree method was set to be 'exact', "
                        "but currently we are only able to proceed with "
                        "approximate algorithm ('approx') because external-"
                        "memory data matrix is used.";
        break;
       case TreeMethod::kHist:
        // things are okay, do nothing
        break;
       case TreeMethod::kGPUExact:
       case TreeMethod::kGPUHist:
        LOG(FATAL)
          << "External-memory data matrix is not available with GPU algorithms";
        break;
       default:
        LOG(FATAL) << "Unknown tree_method ("
                   << static_cast<int>(current_tree_method) << ") detected";
      }
      tparam_.tree_method = TreeMethod::kApprox;
    } else if (p_train->Info().num_row_ >= (4UL << 20UL)
               && current_tree_method == TreeMethod::kAuto) {
      /* Choose tree_method='approx' automatically for large data matrix */
      LOG(CONSOLE) << "Tree method is automatically selected to be "
                      "'approx' for faster speed. To use old behavior "
                      "(exact greedy algorithm on single machine), "
                      "set tree_method to 'exact'.";
      tparam_.tree_method = TreeMethod::kApprox;
    }

    /* If tree_method was changed, re-configure updaters and gradient boosters */
    if (tparam_.tree_method != current_tree_method) {
      ConfigureUpdaters();
      if (gbm_ != nullptr) {
        gbm_->Configure(cfg_.begin(), cfg_.end());
      }
    }
  }

  // return whether model is already initialized.
  inline bool ModelInitialized() const { return gbm_ != nullptr; }
  // lazily initialize the model if it haven't yet been initialized.
  inline void LazyInitModel() {
    if (this->ModelInitialized()) return;
    // estimate feature bound
    unsigned num_feature = 0;
    for (auto & matrix : cache_) {
      CHECK(matrix != nullptr);
      num_feature = std::max(num_feature,
                             static_cast<unsigned>(matrix->Info().num_col_));
    }
    // run allreduce on num_feature to find the maximum value
    rabit::Allreduce<rabit::op::Max>(&num_feature, 1);
    if (num_feature > mparam_.num_feature) {
      mparam_.num_feature = num_feature;
    }
    // setup
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);
    CHECK(obj_ == nullptr && gbm_ == nullptr);
    obj_.reset(ObjFunction::Create(name_obj_));
    obj_->Configure(cfg_.begin(), cfg_.end());
    // reset the base score
    mparam_.base_score = obj_->ProbToMargin(mparam_.base_score);
    gbm_.reset(GradientBooster::Create(name_gbm_, cache_, mparam_.base_score));
    gbm_->Configure(cfg_.begin(), cfg_.end());
  }
  /*!
   * \brief get un-transformed prediction
   * \param data training data matrix
   * \param out_preds output vector that stores the prediction
   * \param ntree_limit limit number of trees used for boosted tree
   *   predictor, when it equals 0, this means we are using all the trees
   */
  inline void PredictRaw(DMatrix* data, HostDeviceVector<bst_float>* out_preds,
                         unsigned ntree_limit = 0) const {
    CHECK(gbm_ != nullptr)
        << "Predict must happen after Load or InitModel";
    gbm_->PredictBatch(data, out_preds, ntree_limit);
  }

  // model parameter
  LearnerModelParam mparam_;
  // training parameter
  LearnerTrainParam tparam_;
  // configurations
  std::map<std::string, std::string> cfg_;
  // attributes
  std::map<std::string, std::string> attributes_;
  // name of gbm
  std::string name_gbm_;
  // name of objective function
  std::string name_obj_;
  // temporal storages for prediction
  HostDeviceVector<bst_float> preds_;
  // gradient pairs
  HostDeviceVector<GradientPair> gpair_;

 private:
  /*! \brief random number transformation seed. */
  static const int kRandSeedMagic = 127;
  // internal cached dmatrix
  std::vector<std::shared_ptr<DMatrix> > cache_;

  common::Monitor monitor_;
};

Learner* Learner::Create(
    const std::vector<std::shared_ptr<DMatrix> >& cache_data) {
  return new LearnerImpl(cache_data);
}
}  // namespace xgboost
