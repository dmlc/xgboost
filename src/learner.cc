/*!
 * Copyright 2014-2019 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/any.h>
#include <xgboost/feature_map.h>
#include <xgboost/learner.h>
#include <xgboost/logging.h>
#include <xgboost/generic_parameters.h>
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
#include "./common/timer.h"

namespace {

const char* kMaxDeltaStepDefaultValue = "0.7";

inline bool IsFloat(const std::string& str) {
  std::stringstream ss(str);
  float f{};
  return !((ss >> std::noskipws >> f).rdstate() ^ std::ios_base::eofbit);
}

inline bool IsInt(const std::string& str) {
  std::stringstream ss(str);
  int i{};
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

namespace xgboost {

enum class DataSplitMode : int {
  kAuto = 0, kCol = 1, kRow = 2
};
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::DataSplitMode);

namespace xgboost {
// implementation of base learner.
bool Learner::AllowLazyCheckPoint() const {
  return gbm_->AllowLazyCheckPoint();
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
  // data split mode, can be row, col, or none.
  DataSplitMode dsplit;
  // flag to disable default metric
  int disable_default_eval_metric;

  std::string booster;
  std::string objective;

  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerTrainParam) {
    DMLC_DECLARE_FIELD(dsplit)
        .set_default(DataSplitMode::kAuto)
        .add_enum("auto", DataSplitMode::kAuto)
        .add_enum("col", DataSplitMode::kCol)
        .add_enum("row", DataSplitMode::kRow)
        .describe("Data split mode for distributed training.");
    DMLC_DECLARE_FIELD(disable_default_eval_metric)
        .set_default(0)
        .describe("flag to disable default metric. Set to >0 to disable");
    DMLC_DECLARE_FIELD(booster)
        .set_default("gbtree")
        .describe("Gradient booster used for training.");
    DMLC_DECLARE_FIELD(objective)
        .set_default("reg:squarederror")
        .describe("Objective function used for obtaining gradient.");
  }
};


DMLC_REGISTER_PARAMETER(LearnerModelParam);
DMLC_REGISTER_PARAMETER(LearnerTrainParam);
DMLC_REGISTER_PARAMETER(GenericParameter);

/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
class LearnerImpl : public Learner {
 public:
  explicit LearnerImpl(std::vector<std::shared_ptr<DMatrix> >  cache)
      : configured_{false}, cache_(std::move(cache)) {}
  // Configuration before data is known.
  void Configure() override {
    if (configured_) { return; }
    monitor_.Init("Learner");
    monitor_.Start("Configure");
    auto old_tparam = tparam_;
    Args args = {cfg_.cbegin(), cfg_.cend()};

    tparam_.InitAllowUnknown(args);
    generic_param_.InitAllowUnknown(args);
    ConsoleLogger::Configure(args);
    if (generic_param_.nthread != 0) {
      omp_set_num_threads(generic_param_.nthread);
    }

    // add additional parameters
    // These are cosntraints that need to be satisfied.
    if (tparam_.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      tparam_.dsplit = DataSplitMode::kRow;
    }

    mparam_.InitAllowUnknown(args);
    // set seed only before the model is initialized
    common::GlobalRandom().seed(generic_param_.seed);
    // must precede configure gbm since num_features is required for gbm
    this->ConfigureNumFeatures();
    args = {cfg_.cbegin(), cfg_.cend()};  // renew
    this->ConfigureObjective(old_tparam, &args);
    this->ConfigureGBM(old_tparam, args);
    this->ConfigureMetrics(args);

    this->configured_ = true;
    monitor_.Stop("Configure");
  }

  void CheckDataSplitMode() {
    if (rabit::IsDistributed()) {
      CHECK(tparam_.dsplit != DataSplitMode::kAuto)
        << "Precondition violated; dsplit cannot be 'auto' in distributed mode";
      if (tparam_.dsplit == DataSplitMode::kCol) {
        // 'distcol' updater hidden until it becomes functional again
        // See discussion at https://github.com/dmlc/xgboost/issues/1832
        LOG(FATAL) << "Column-wise data split is currently not supported.";
      }
    }
  }

  void Load(dmlc::Stream* fi) override {
    generic_param_.InitAllowUnknown(Args{});
    tparam_.Init(std::vector<std::pair<std::string, std::string>>{});
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
        tparam_.objective.resize(len);
        CHECK_EQ(fi->Read(&tparam_.objective[0], len), len)
            << "BoostLearner: wrong model format";
      }
    }
    CHECK(fi->Read(&tparam_.booster)) << "BoostLearner: wrong model format";
    // duplicated code with LazyInitModel
    obj_.reset(ObjFunction::Create(tparam_.objective, &generic_param_));
    gbm_.reset(GradientBooster::Create(tparam_.booster, &generic_param_,
                                       cache_, mparam_.base_score));
    gbm_->Load(fi);
    if (mparam_.contain_extra_attrs != 0) {
      std::vector<std::pair<std::string, std::string> > attr;
      fi->Read(&attr);
      for (auto& kv : attr) {
        // Load `predictor`, `gpu_id` parameters from extra attributes
        const std::string prefix = "SAVED_PARAM_";
        if (kv.first.find(prefix) == 0) {
          const std::string saved_param = kv.first.substr(prefix.length());
          bool is_gpu_predictor = saved_param == "predictor" && kv.second == "gpu_predictor";
#ifdef XGBOOST_USE_CUDA
          if (saved_param == "predictor" || saved_param == "gpu_id") {
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
          if (is_gpu_predictor) {
            cfg_["predictor"] = "cpu_predictor";
            kv.second = "cpu_predictor";
          }
#endif  // XGBOOST_USE_CUDA
          // NO visible GPU in current environment
          if (is_gpu_predictor && GPUSet::AllVisible().Size() == 0) {
            cfg_["predictor"] = "cpu_predictor";
            kv.second = "cpu_predictor";
            LOG(INFO) << "Switch gpu_predictor to cpu_predictor.";
          }
        }
      }
      attributes_ = std::map<std::string, std::string>(attr.begin(), attr.end());
    }
    if (tparam_.objective == "count:poisson") {
      std::string max_delta_step;
      fi->Read(&max_delta_step);
      cfg_["max_delta_step"] = max_delta_step;
    }
    if (mparam_.contain_eval_metrics != 0) {
      std::vector<std::string> metr;
      fi->Read(&metr);
      for (auto name : metr) {
        metrics_.emplace_back(Metric::Create(name, &generic_param_));
      }
    }

    cfg_["num_class"] = common::ToString(mparam_.num_class);
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);

    auto n = tparam_.__DICT__();
    cfg_.insert(n.cbegin(), n.cend());

    Args args = {cfg_.cbegin(), cfg_.cend()};
    generic_param_.InitAllowUnknown(args);
    gbm_->Configure(args);
    obj_->Configure({cfg_.begin(), cfg_.end()});

    for (auto& p_metric : metrics_) {
      p_metric->Configure({cfg_.begin(), cfg_.end()});
    }

    this->configured_ = true;
  }

  // rabit save model to rabit checkpoint
  void Save(dmlc::Stream* fo) const override {
    if (!this->configured_) {
      // Save empty model.  Calling Configure in a dummy LearnerImpl avoids violating
      // constness.
      LearnerImpl empty(std::move(this->cache_));
      empty.SetParams({this->cfg_.cbegin(), this->cfg_.cend()});
      for (auto const& kv : attributes_) {
        empty.SetAttr(kv.first, kv.second);
      }
      empty.Configure();
      empty.Save(fo);
      return;
    }

    LearnerModelParam mparam = mparam_;  // make a copy to potentially modify
    std::vector<std::pair<std::string, std::string> > extra_attr;
    // extra attributed to be added just before saving
    if (tparam_.objective == "count:poisson") {
      auto it = cfg_.find("max_delta_step");
      if (it != cfg_.end()) {
        // write `max_delta_step` parameter as extra attribute of booster
        mparam.contain_extra_attrs = 1;
        extra_attr.emplace_back("count_poisson_max_delta_step", it->second);
      }
    }
    {
      // Write `predictor`, `gpu_id` parameters as extra attributes
      for (const auto& key : std::vector<std::string>{"predictor", "gpu_id"}) {
        auto it = cfg_.find(key);
        if (it != cfg_.end()) {
          mparam.contain_extra_attrs = 1;
          extra_attr.emplace_back("SAVED_PARAM_" + key, it->second);
        }
      }
    }
    fo->Write(&mparam, sizeof(LearnerModelParam));
    fo->Write(tparam_.objective);
    fo->Write(tparam_.booster);
    gbm_->Save(fo);
    if (mparam.contain_extra_attrs != 0) {
      std::map<std::string, std::string> attr(attributes_);
      for (const auto& kv : extra_attr) {
        attr[kv.first] = kv.second;
      }
      fo->Write(std::vector<std::pair<std::string, std::string>>(
                  attr.begin(), attr.end()));
    }
    if (tparam_.objective == "count:poisson") {
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

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    return gbm_->DumpModel(fmap, with_stats, format);
  }

  void UpdateOneIter(int iter, DMatrix* train) override {
    monitor_.Start("UpdateOneIter");

    if (generic_param_.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(generic_param_.seed * kRandSeedMagic + iter);
    }
    this->Configure();
    this->CheckDataSplitMode();
    this->ValidateDMatrix(train);

    monitor_.Start("PredictRaw");
    this->PredictRaw(train, &preds_[train]);
    monitor_.Stop("PredictRaw");
    monitor_.Start("GetGradient");
    obj_->GetGradient(preds_[train], train->Info(), iter, &gpair_);
    monitor_.Stop("GetGradient");
    gbm_->DoBoost(train, &gpair_, obj_.get());
    monitor_.Stop("UpdateOneIter");
  }

  void BoostOneIter(int iter, DMatrix* train,
                    HostDeviceVector<GradientPair>* in_gpair) override {
    monitor_.Start("BoostOneIter");
    if (generic_param_.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(generic_param_.seed * kRandSeedMagic + iter);
    }
    this->Configure();
    this->CheckDataSplitMode();
    this->ValidateDMatrix(train);

    gbm_->DoBoost(train, in_gpair);
    monitor_.Stop("BoostOneIter");
  }

  std::string EvalOneIter(int iter, const std::vector<DMatrix*>& data_sets,
                          const std::vector<std::string>& data_names) override {
    monitor_.Start("EvalOneIter");
    this->Configure();

    std::ostringstream os;
    os << '[' << iter << ']' << std::setiosflags(std::ios::fixed);
    if (metrics_.size() == 0 && tparam_.disable_default_eval_metric <= 0) {
      metrics_.emplace_back(Metric::Create(obj_->DefaultEvalMetric(), &generic_param_));
      metrics_.back()->Configure({cfg_.begin(), cfg_.end()});
    }
    for (size_t i = 0; i < data_sets.size(); ++i) {
      DMatrix * dmat = data_sets[i];
      this->ValidateDMatrix(dmat);
      this->PredictRaw(data_sets[i], &preds_[dmat]);
      obj_->EvalTransform(&preds_[dmat]);
      for (auto& ev : metrics_) {
        os << '\t' << data_names[i] << '-' << ev->Name() << ':'
           << ev->Eval(preds_[dmat], data_sets[i]->Info(),
                       tparam_.dsplit == DataSplitMode::kRow);
      }
    }

    monitor_.Stop("EvalOneIter");
    return os.str();
  }

  void SetParam(const std::string& key, const std::string& value) override {
    configured_ = false;
    if (key == kEvalMetric) {
      if (std::find(metric_names_.cbegin(), metric_names_.cend(),
                    value) == metric_names_.cend()) {
        metric_names_.emplace_back(value);
      }
    } else {
      cfg_[key] = value;
    }
  }
  // Short hand for setting multiple parameters
  void SetParams(std::vector<std::pair<std::string, std::string>> const& args) override {
    configured_ = false;
    for (auto const& kv : args) {
      this->SetParam(kv.first, kv.second);
    }
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
    if (it == attributes_.end()) { return false; }
    attributes_.erase(it);
    return true;
  }

  std::vector<std::string> GetAttrNames() const override {
    std::vector<std::string> out;
    for (auto const& kv : attributes_) {
      out.emplace_back(kv.first);
    }
    return out;
  }

  GenericParameter const& GetGenericParameter() const override {
    return generic_param_;
  }

  void Predict(DMatrix* data, bool output_margin,
               HostDeviceVector<bst_float>* out_preds, unsigned ntree_limit,
               bool pred_leaf, bool pred_contribs, bool approx_contribs,
               bool pred_interactions) override {
    int multiple_predictions = static_cast<int>(pred_leaf) +
                               static_cast<int>(pred_interactions) +
                               static_cast<int>(pred_contribs);
    this->Configure();
    CHECK_LE(multiple_predictions, 1) << "Perform one kind of prediction at a time.";
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
  /*!
   * \brief get un-transformed prediction
   * \param data training data matrix
   * \param out_preds output vector that stores the prediction
   * \param ntree_limit limit number of trees used for boosted tree
   *   predictor, when it equals 0, this means we are using all the trees
   */
  void PredictRaw(DMatrix* data, HostDeviceVector<bst_float>* out_preds,
                  unsigned ntree_limit = 0) const {
    CHECK(gbm_ != nullptr)
        << "Predict must happen after Load or InitModel";
    this->ValidateDMatrix(data);
    gbm_->PredictBatch(data, out_preds, ntree_limit);
  }

  void ConfigureObjective(LearnerTrainParam const& old, Args* p_args) {
    if (cfg_.find("num_class") != cfg_.cend() && cfg_.at("num_class") != "0") {
      cfg_["num_output_group"] = cfg_["num_class"];
      if (atoi(cfg_["num_class"].c_str()) > 1 && cfg_.count("objective") == 0) {
        tparam_.objective = "multi:softmax";
      }
    }

    if (cfg_.find("max_delta_step") == cfg_.cend() &&
        cfg_.find("objective") != cfg_.cend() &&
        tparam_.objective == "count:poisson") {
      cfg_["max_delta_step"] = kMaxDeltaStepDefaultValue;
    }
    if (obj_ == nullptr || tparam_.objective != old.objective) {
      obj_.reset(ObjFunction::Create(tparam_.objective, &generic_param_));
    }
    // reset the base score
    mparam_.base_score = obj_->ProbToMargin(mparam_.base_score);
    auto& args = *p_args;
    args = {cfg_.cbegin(), cfg_.cend()};  // renew
    obj_->Configure(args);
  }

  void ConfigureMetrics(Args const& args) {
    for (auto const& name : metric_names_) {
      auto DupCheck = [&name](std::unique_ptr<Metric> const& m) {
                        return m->Name() != name;
                      };
      if (std::all_of(metrics_.begin(), metrics_.end(), DupCheck)) {
        metrics_.emplace_back(std::unique_ptr<Metric>(Metric::Create(name, &generic_param_)));
        mparam_.contain_eval_metrics = 1;
      }
    }
    for (auto& p_metric : metrics_) {
      p_metric->Configure(args);
    }
  }

  void ConfigureGBM(LearnerTrainParam const& old, Args const& args) {
    if (gbm_ == nullptr || old.booster != tparam_.booster) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &generic_param_,
                                         cache_, mparam_.base_score));
    }
    gbm_->Configure(args);

    if (this->gbm_->UseGPU()) {
      if (cfg_.find("gpu_id") == cfg_.cend()) {
        generic_param_.gpu_id = 0;
      }
    }
  }

  // set number of features correctly.
  void ConfigureNumFeatures() {
    // estimate feature bound
    // TODO(hcho3): Change num_feature to 64-bit integer
    unsigned num_feature = 0;
    for (auto & matrix : cache_) {
      CHECK(matrix != nullptr);
      const uint64_t num_col = matrix->Info().num_col_;
      CHECK_LE(num_col, static_cast<uint64_t>(std::numeric_limits<unsigned>::max()))
          << "Unfortunately, XGBoost does not support data matrices with "
          << std::numeric_limits<unsigned>::max() << " features or greater";
      num_feature = std::max(num_feature, static_cast<unsigned>(num_col));
    }
    // run allreduce on num_feature to find the maximum value
    rabit::Allreduce<rabit::op::Max>(&num_feature, 1);
    if (num_feature > mparam_.num_feature) {
      mparam_.num_feature = num_feature;
    }
    CHECK_NE(mparam_.num_feature, 0)
        << "0 feature is supplied.  Are you using raw Booster interface?";
    // setup
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);
    cfg_["num_class"] = common::ToString(mparam_.num_class);
  }

  void ValidateDMatrix(DMatrix* p_fmat) const {
    MetaInfo const& info = p_fmat->Info();
    auto const& weights = info.weights_.HostVector();
    if (info.group_ptr_.size() != 0 && weights.size() != 0) {
      CHECK(weights.size() == info.group_ptr_.size() - 1)
          << "\n"
          << "weights size: " << weights.size()            << ", "
          << "groups size: "  << info.group_ptr_.size() -1 << ", "
          << "num rows: "     << p_fmat->Info().num_row_   << "\n"
          << "Number of weights should be equal to number of groups in ranking task.";
    }
  }

  // model parameter
  LearnerModelParam mparam_;
  LearnerTrainParam tparam_;
  // configurations
  std::map<std::string, std::string> cfg_;
  // FIXME(trivialfis): Legacy field used to store extra attributes into binary model.
  std::map<std::string, std::string> attributes_;
  std::vector<std::string> metric_names_;
  static std::string const kEvalMetric;  // NOLINT
  // temporal storages for prediction
  std::map<DMatrix*, HostDeviceVector<bst_float>> preds_;
  // gradient pairs
  HostDeviceVector<GradientPair> gpair_;

  bool configured_;

 private:
  /*! \brief random number transformation seed. */
  static int32_t constexpr kRandSeedMagic = 127;
  // internal cached dmatrix
  std::vector<std::shared_ptr<DMatrix> > cache_;

  common::Monitor monitor_;
};

std::string const LearnerImpl::kEvalMetric {"eval_metric"};  // NOLINT

constexpr int32_t LearnerImpl::kRandSeedMagic;

Learner* Learner::Create(
    const std::vector<std::shared_ptr<DMatrix> >& cache_data) {
  return new LearnerImpl(cache_data);
}
}  // namespace xgboost
