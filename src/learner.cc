/*!
 * Copyright 2014-2022 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>

#include <atomic>
#include <mutex>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <stack>
#include <utility>
#include <vector>

#include "dmlc/any.h"
#include "xgboost/base.h"
#include "xgboost/c_api.h"
#include "xgboost/data.h"
#include "xgboost/model.h"
#include "xgboost/predictor.h"
#include "xgboost/feature_map.h"
#include "xgboost/gbm.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"
#include "xgboost/logging.h"
#include "xgboost/metric.h"
#include "xgboost/objective.h"
#include "xgboost/parameter.h"

#include "common/common.h"
#include "common/io.h"
#include "common/observer.h"
#include "common/random.h"
#include "common/timer.h"
#include "common/charconv.h"
#include "common/version.h"
#include "common/threading_utils.h"

namespace {

const char* kMaxDeltaStepDefaultValue = "0.7";
}  // anonymous namespace

namespace xgboost {

enum class DataSplitMode : int {
  kAuto = 0, kCol = 1, kRow = 2
};
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::DataSplitMode);

namespace xgboost {
Learner::~Learner() = default;

/*! \brief training parameter for regression
 *
 * Should be deprecated, but still used for being compatible with binary IO.
 * Once it's gone, `LearnerModelParam` should handle transforming `base_margin`
 * with objective by itself.
 */
struct LearnerModelParamLegacy : public dmlc::Parameter<LearnerModelParamLegacy> {
  /* \brief global bias */
  bst_float base_score;
  /* \brief number of features  */
  uint32_t num_feature;
  /* \brief number of classes, if it is multi-class classification  */
  int32_t num_class;
  /*! \brief Model contain additional properties */
  int32_t contain_extra_attrs;
  /*! \brief Model contain eval metrics */
  int32_t contain_eval_metrics;
  /*! \brief the version of XGBoost. */
  uint32_t major_version;
  uint32_t minor_version;

  uint32_t num_target{1};
  /*! \brief reserved field */
  int reserved[26];
  /*! \brief constructor */
  LearnerModelParamLegacy() {
    std::memset(this, 0, sizeof(LearnerModelParamLegacy));
    base_score = 0.5f;
    num_target = 1;
    major_version = std::get<0>(Version::Self());
    minor_version = std::get<1>(Version::Self());
    static_assert(sizeof(LearnerModelParamLegacy) == 136,
                  "Do not change the size of this struct, as it will break binary IO.");
  }
  // Skip other legacy fields.
  Json ToJson() const {
    Object obj;
    char floats[NumericLimits<float>::kToCharsSize];
    auto ret = to_chars(floats, floats + NumericLimits<float>::kToCharsSize, base_score);
    CHECK(ret.ec == std::errc());
    obj["base_score"] =
        std::string{floats, static_cast<size_t>(std::distance(floats, ret.ptr))};

    char integers[NumericLimits<int64_t>::kToCharsSize];
    ret = to_chars(integers, integers + NumericLimits<int64_t>::kToCharsSize,
                   static_cast<int64_t>(num_feature));
    CHECK(ret.ec == std::errc());
    obj["num_feature"] =
        std::string{integers, static_cast<size_t>(std::distance(integers, ret.ptr))};
    ret = to_chars(integers, integers + NumericLimits<int64_t>::kToCharsSize,
                   static_cast<int64_t>(num_class));
    CHECK(ret.ec == std::errc());
    obj["num_class"] =
        std::string{integers, static_cast<size_t>(std::distance(integers, ret.ptr))};

    ret = to_chars(integers, integers + NumericLimits<int64_t>::kToCharsSize,
                   static_cast<int64_t>(num_target));
    obj["num_target"] =
        std::string{integers, static_cast<size_t>(std::distance(integers, ret.ptr))};

    return Json(std::move(obj));
  }
  void FromJson(Json const& obj) {
    auto const& j_param = get<Object const>(obj);
    std::map<std::string, std::string> m;
    m["num_feature"] = get<String const>(j_param.at("num_feature"));
    m["num_class"] = get<String const>(j_param.at("num_class"));
    auto n_targets_it = j_param.find("num_target");
    if (n_targets_it != j_param.cend()) {
      m["num_target"] = get<String const>(n_targets_it->second);
    }

    this->Init(m);
    std::string str = get<String const>(j_param.at("base_score"));
    from_chars(str.c_str(), str.c_str() + str.size(), base_score);
  }
  inline LearnerModelParamLegacy ByteSwap() const {
    LearnerModelParamLegacy x = *this;
    dmlc::ByteSwap(&x.base_score, sizeof(x.base_score), 1);
    dmlc::ByteSwap(&x.num_feature, sizeof(x.num_feature), 1);
    dmlc::ByteSwap(&x.num_class, sizeof(x.num_class), 1);
    dmlc::ByteSwap(&x.contain_extra_attrs, sizeof(x.contain_extra_attrs), 1);
    dmlc::ByteSwap(&x.contain_eval_metrics, sizeof(x.contain_eval_metrics), 1);
    dmlc::ByteSwap(&x.major_version, sizeof(x.major_version), 1);
    dmlc::ByteSwap(&x.minor_version, sizeof(x.minor_version), 1);
    dmlc::ByteSwap(&x.num_target, sizeof(x.num_target), 1);
    dmlc::ByteSwap(x.reserved, sizeof(x.reserved[0]), sizeof(x.reserved) / sizeof(x.reserved[0]));
    return x;
  }

  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerModelParamLegacy) {
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
    DMLC_DECLARE_FIELD(num_target)
        .set_default(1)
        .set_lower_bound(1)
        .describe("Number of target for multi-target regression.");
  }
};

LearnerModelParam::LearnerModelParam(LearnerModelParamLegacy const& user_param, float base_margin,
                                     ObjInfo t)
    : base_score{base_margin}, num_feature{user_param.num_feature}, task{t} {
  auto n_classes = std::max(static_cast<uint32_t>(user_param.num_class), 1u);
  auto n_targets = user_param.num_target;
  num_output_group = std::max(n_classes, n_targets);
  // For version < 1.6, n_targets == 0
  CHECK(n_classes <= 1 || n_targets <= 1)
      << "Multi-class multi-output is not yet supported. n_classes:" << n_classes
      << ", n_targets:" << n_targets;
}

struct LearnerTrainParam : public XGBoostParameter<LearnerTrainParam> {
  // data split mode, can be row, col, or none.
  DataSplitMode dsplit {DataSplitMode::kAuto};
  // flag to disable default metric
  bool disable_default_eval_metric {false};
  // FIXME(trivialfis): The following parameters belong to model itself, but can be
  // specified by users.  Move them to model parameter once we can get rid of binary IO.
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
        .set_default(false)
        .describe("Flag to disable default metric. Set to >0 to disable");
    DMLC_DECLARE_FIELD(booster)
        .set_default("gbtree")
        .describe("Gradient booster used for training.");
    DMLC_DECLARE_FIELD(objective)
        .set_default("reg:squarederror")
        .describe("Objective function used for obtaining gradient.");
  }
};


DMLC_REGISTER_PARAMETER(LearnerModelParamLegacy);
DMLC_REGISTER_PARAMETER(LearnerTrainParam);
DMLC_REGISTER_PARAMETER(GenericParameter);

int constexpr GenericParameter::kCpuId;
int64_t constexpr GenericParameter::kDefaultSeed;

GenericParameter::GenericParameter() : cfs_cpu_count_{common::GetCfsCPUCount()} {}

void GenericParameter::ConfigureGpuId(bool require_gpu) {
#if defined(XGBOOST_USE_CUDA)
  if (gpu_id == kCpuId) {  // 0. User didn't specify the `gpu_id'
    if (require_gpu) {     // 1. `tree_method' or `predictor' or both are using
                           // GPU.
      // 2. Use device 0 as default.
      this->UpdateAllowUnknown(Args{{"gpu_id", "0"}});
    }
  }

  // 3. When booster is loaded from a memory image (Python pickle or R
  // raw model), number of available GPUs could be different.  Wrap around it.
  int32_t n_gpus = common::AllVisibleGPUs();
  if (n_gpus == 0) {
    if (gpu_id != kCpuId) {
      LOG(WARNING) << "No visible GPU is found, setting `gpu_id` to -1";
    }
    this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(kCpuId)}});
  } else if (fail_on_invalid_gpu_id) {
    CHECK(gpu_id == kCpuId || gpu_id < n_gpus)
      << "Only " << n_gpus << " GPUs are visible, gpu_id "
      << gpu_id << " is invalid.";
  } else if (gpu_id != kCpuId && gpu_id >= n_gpus) {
    LOG(WARNING) << "Only " << n_gpus
                 << " GPUs are visible, setting `gpu_id` to " << gpu_id % n_gpus;
    this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(gpu_id % n_gpus)}});
  }
#else
  // Just set it to CPU, don't think about it.
  this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(kCpuId)}});
#endif  // defined(XGBOOST_USE_CUDA)
}

int32_t GenericParameter::Threads() const {
  auto n_threads = common::OmpGetNumThreads(nthread);
  if (cfs_cpu_count_ > 0) {
    n_threads = std::min(n_threads, cfs_cpu_count_);
  }
  return n_threads;
}

using LearnerAPIThreadLocalStore =
    dmlc::ThreadLocalStore<std::map<Learner const *, XGBAPIThreadLocalEntry>>;

using ThreadLocalPredictionCache =
    dmlc::ThreadLocalStore<std::map<Learner const *, PredictionContainer>>;

namespace {
StringView ModelMsg() {
  return StringView{
      R"doc(
  If you are loading a serialized model (like pickle in Python, RDS in R) generated by
  older XGBoost, please export the model by calling `Booster.save_model` from that version
  first, then load it back in current version. See:

    https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

  for more details about differences between saving model and serializing.
)doc"};
}
}  // anonymous namespace

class LearnerConfiguration : public Learner {
 private:
  std::mutex config_lock_;

 protected:
  static std::string const kEvalMetric;  // NOLINT

 protected:
  std::atomic<bool> need_configuration_;
  std::map<std::string, std::string> cfg_;
  // Stores information like best-iteration for early stopping.
  std::map<std::string, std::string> attributes_;
  // Name of each feature, usually set from DMatrix.
  std::vector<std::string> feature_names_;
  // Type of each feature, usually set from DMatrix.
  std::vector<std::string> feature_types_;

  common::Monitor monitor_;
  LearnerModelParamLegacy mparam_;
  LearnerModelParam learner_model_param_;
  LearnerTrainParam tparam_;
  std::vector<std::string> metric_names_;

 public:
  explicit LearnerConfiguration(std::vector<std::shared_ptr<DMatrix> > cache)
      : need_configuration_{true} {
    monitor_.Init("Learner");
    auto& local_cache = (*ThreadLocalPredictionCache::Get())[this];
    for (std::shared_ptr<DMatrix> const& d : cache) {
      local_cache.Cache(d, GenericParameter::kCpuId);
    }
  }
  ~LearnerConfiguration() override {
    auto local_cache = ThreadLocalPredictionCache::Get();
    if (local_cache->find(this) != local_cache->cend()) {
      local_cache->erase(this);
    }
  }

  // Configuration before data is known.
  void Configure() override {
    // Varient of double checked lock
    if (!this->need_configuration_) { return; }
    std::lock_guard<std::mutex> guard(config_lock_);
    if (!this->need_configuration_) { return; }

    monitor_.Start("Configure");
    auto old_tparam = tparam_;
    Args args = {cfg_.cbegin(), cfg_.cend()};

    tparam_.UpdateAllowUnknown(args);
    auto mparam_backup = mparam_;

    mparam_.UpdateAllowUnknown(args);

    auto initialized = generic_parameters_.GetInitialised();
    auto old_seed = generic_parameters_.seed;
    generic_parameters_.UpdateAllowUnknown(args);

    ConsoleLogger::Configure(args);

    // add additional parameters
    // These are cosntraints that need to be satisfied.
    if (tparam_.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      tparam_.dsplit = DataSplitMode::kRow;
    }

    // set seed only before the model is initialized
    if (!initialized || generic_parameters_.seed != old_seed) {
      common::GlobalRandom().seed(generic_parameters_.seed);
    }

    // must precede configure gbm since num_features is required for gbm
    this->ConfigureNumFeatures();
    args = {cfg_.cbegin(), cfg_.cend()};  // renew
    this->ConfigureObjective(old_tparam, &args);

    auto task = this->ConfigureTargets();

    // Before 1.0.0, we save `base_score` into binary as a transformed value by objective.
    // After 1.0.0 we save the value provided by user and keep it immutable instead.  To
    // keep the stability, we initialize it in binary LoadModel instead of configuration.
    // Under what condition should we omit the transformation:
    //
    // - base_score is loaded from old binary model.
    //
    // What are the other possible conditions:
    //
    // - model loaded from new binary or JSON.
    // - model is created from scratch.
    // - model is configured second time due to change of parameter
    if (!learner_model_param_.Initialized() || mparam_.base_score != mparam_backup.base_score) {
      learner_model_param_ =
          LearnerModelParam(mparam_, obj_->ProbToMargin(mparam_.base_score), task);
    }

    this->ConfigureGBM(old_tparam, args);
    generic_parameters_.ConfigureGpuId(this->gbm_->UseGPU());
    this->ConfigureMetrics(args);

    this->need_configuration_ = false;
    if (generic_parameters_.validate_parameters) {
      this->ValidateParameters();
    }

    cfg_.clear();
    monitor_.Stop("Configure");
  }

  virtual PredictionContainer* GetPredictionCache() const {
    return &((*ThreadLocalPredictionCache::Get())[this]);
  }

  void LoadConfig(Json const& in) override {
    // If configuration is loaded, ensure that the model came from the same version
    CHECK(IsA<Object>(in));
    auto origin_version = Version::Load(in);

    if (!Version::Same(origin_version)) {
      LOG(WARNING) << ModelMsg();
      return;  // skip configuration if version is not matched
    }

    auto const& learner_parameters = get<Object>(in["learner"]);
    FromJson(learner_parameters.at("learner_train_param"), &tparam_);

    auto const& gradient_booster = learner_parameters.at("gradient_booster");

    auto const& objective_fn = learner_parameters.at("objective");
    if (!obj_) {
      obj_.reset(ObjFunction::Create(tparam_.objective, &generic_parameters_));
    }
    obj_->LoadConfig(objective_fn);
    learner_model_param_.task = obj_->Task();

    tparam_.booster = get<String>(gradient_booster["name"]);
    if (!gbm_) {
      gbm_.reset(GradientBooster::Create(tparam_.booster,
                                         &generic_parameters_, &learner_model_param_));
    }
    gbm_->LoadConfig(gradient_booster);

    auto const& j_metrics = learner_parameters.at("metrics");
    auto n_metrics = get<Array const>(j_metrics).size();
    metric_names_.resize(n_metrics);
    metrics_.resize(n_metrics);
    for (size_t i = 0; i < n_metrics; ++i) {
      auto old_serialization = IsA<String>(j_metrics[i]);
      if (old_serialization) {
        LOG(WARNING) << ModelMsg();
        metric_names_[i] = get<String>(j_metrics[i]);
      } else {
        metric_names_[i] = get<String>(j_metrics[i]["name"]);
      }
      metrics_[i] = std::unique_ptr<Metric>(Metric::Create(metric_names_[i], &generic_parameters_));
      if (!old_serialization) {
        metrics_[i]->LoadConfig(j_metrics[i]);
      }
    }

    FromJson(learner_parameters.at("generic_param"), &generic_parameters_);
    // make sure the GPU ID is valid in new environment before start running configure.
    generic_parameters_.ConfigureGpuId(false);

    this->need_configuration_ = true;
  }

  void SaveConfig(Json* p_out) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";
    Version::Save(p_out);
    Json& out { *p_out };
    // parameters
    out["learner"] = Object();
    auto& learner_parameters = out["learner"];

    learner_parameters["learner_train_param"] = ToJson(tparam_);
    learner_parameters["learner_model_param"] = mparam_.ToJson();
    learner_parameters["gradient_booster"] = Object();
    auto& gradient_booster = learner_parameters["gradient_booster"];
    gbm_->SaveConfig(&gradient_booster);

    learner_parameters["objective"] = Object();
    auto& objective_fn = learner_parameters["objective"];
    obj_->SaveConfig(&objective_fn);

    std::vector<Json> metrics(metrics_.size(), Json{Object{}});
    for (size_t i = 0; i < metrics_.size(); ++i) {
      metrics_[i]->SaveConfig(&metrics[i]);
    }
    learner_parameters["metrics"] = Array(std::move(metrics));

    learner_parameters["generic_param"] = ToJson(generic_parameters_);
  }

  void SetParam(const std::string& key, const std::string& value) override {
    this->need_configuration_ = true;
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
    for (auto const& kv : args) {
      this->SetParam(kv.first, kv.second);
    }
  }

  uint32_t GetNumFeature() const override {
    return learner_model_param_.num_feature;
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

  void SetFeatureNames(std::vector<std::string> const& fn) override {
    feature_names_ = fn;
  }

  void GetFeatureNames(std::vector<std::string>* fn) const override {
    *fn = feature_names_;
  }

  void SetFeatureTypes(std::vector<std::string> const& ft) override {
    this->feature_types_ = ft;
  }

  void GetFeatureTypes(std::vector<std::string>* p_ft) const override {
    auto& ft = *p_ft;
    ft = this->feature_types_;
  }

  std::vector<std::string> GetAttrNames() const override {
    std::vector<std::string> out;
    for (auto const& kv : attributes_) {
      out.emplace_back(kv.first);
    }
    return out;
  }

  const std::map<std::string, std::string>& GetConfigurationArguments() const override {
    return cfg_;
  }

  GenericParameter const* Ctx() const override { return &generic_parameters_; }

 private:
  void ValidateParameters() {
    Json config { Object() };
    this->SaveConfig(&config);
    std::stack<Json> stack;
    stack.push(config);
    std::string const postfix{"_param"};

    auto is_parameter = [&postfix](std::string const &key) {
      return key.size() > postfix.size() &&
             std::equal(postfix.rbegin(), postfix.rend(), key.rbegin());
    };

    // Extract all parameters
    std::vector<std::string> keys;
    // First global parameters
    Json const global_config{ToJson(*GlobalConfigThreadLocalStore::Get())};
    for (auto const& items : get<Object const>(global_config)) {
      keys.emplace_back(items.first);
    }
    // Parameters in various xgboost components.
    while (!stack.empty()) {
      auto j_obj = stack.top();
      stack.pop();
      auto const &obj = get<Object const>(j_obj);

      for (auto const &kv : obj) {
        if (is_parameter(kv.first)) {
          auto parameter = get<Object const>(kv.second);
          std::transform(parameter.begin(), parameter.end(), std::back_inserter(keys),
                         [](std::pair<std::string const&, Json const&> const& kv) {
                           return kv.first;
                         });
        } else if (IsA<Object>(kv.second)) {
          stack.push(kv.second);
        }
      }
    }

    // FIXME(trivialfis): Make eval_metric a training parameter.
    keys.emplace_back(kEvalMetric);
    keys.emplace_back("num_output_group");

    std::sort(keys.begin(), keys.end());

    std::vector<std::string> provided;
    for (auto const &kv : cfg_) {
      if (std::any_of(kv.first.cbegin(), kv.first.cend(),
                      [](char ch) { return std::isspace(ch); })) {
        LOG(FATAL) << "Invalid parameter \"" << kv.first << "\" contains whitespace.";
      }
      provided.push_back(kv.first);
    }
    std::sort(provided.begin(), provided.end());

    std::vector<std::string> diff;
    std::set_difference(provided.begin(), provided.end(), keys.begin(),
                        keys.end(), std::back_inserter(diff));
    if (diff.size() != 0) {
      std::stringstream ss;
      ss << "\nParameters: { ";
      for (size_t i = 0; i < diff.size() - 1; ++i) {
        ss << "\"" << diff[i] << "\", ";
      }
      ss << "\"" << diff.back() << "\"";
      ss << R"W( } are not used.
)W";
      LOG(WARNING) << ss.str();
    }
  }

  void ConfigureNumFeatures() {
    // Compute number of global features if parameter not already set
    if (mparam_.num_feature == 0) {
      // TODO(hcho3): Change num_feature to 64-bit integer
      unsigned num_feature = 0;
      auto local_cache = this->GetPredictionCache();
      for (auto& matrix : local_cache->Container()) {
        CHECK(matrix.first);
        CHECK(!matrix.second.ref.expired());
        const uint64_t num_col = matrix.first->Info().num_col_;
        CHECK_LE(num_col, static_cast<uint64_t>(std::numeric_limits<unsigned>::max()))
            << "Unfortunately, XGBoost does not support data matrices with "
            << std::numeric_limits<unsigned>::max() << " features or greater";
        num_feature = std::max(num_feature, static_cast<uint32_t>(num_col));
      }

      rabit::Allreduce<rabit::op::Max>(&num_feature, 1);
      if (num_feature > mparam_.num_feature) {
        mparam_.num_feature = num_feature;
      }
    }
    CHECK_NE(mparam_.num_feature, 0)
        << "0 feature is supplied.  Are you using raw Booster interface?";
    // Remove these once binary IO is gone.
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);
    cfg_["num_class"] = common::ToString(mparam_.num_class);
  }

  void ConfigureGBM(LearnerTrainParam const& old, Args const& args) {
    if (gbm_ == nullptr || old.booster != tparam_.booster) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &generic_parameters_,
                                         &learner_model_param_));
    }
    gbm_->Configure(args);
  }

  void ConfigureObjective(LearnerTrainParam const& old, Args* p_args) {
    // Once binary IO is gone, NONE of these config is useful.
    if (cfg_.find("num_class") != cfg_.cend() && cfg_.at("num_class") != "0" &&
        tparam_.objective != "multi:softprob") {
      cfg_["num_output_group"] = cfg_["num_class"];
      if (atoi(cfg_["num_class"].c_str()) > 1 && cfg_.count("objective") == 0) {
        tparam_.objective = "multi:softmax";
      }
    }

    if (cfg_.find("max_delta_step") == cfg_.cend() &&
        cfg_.find("objective") != cfg_.cend() &&
        tparam_.objective == "count:poisson") {
      // max_delta_step is a duplicated parameter in Poisson regression and tree param.
      // Rename one of them once binary IO is gone.
      cfg_["max_delta_step"] = kMaxDeltaStepDefaultValue;
    }
    if (obj_ == nullptr || tparam_.objective != old.objective) {
      obj_.reset(ObjFunction::Create(tparam_.objective, &generic_parameters_));
    }
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
        metrics_.emplace_back(std::unique_ptr<Metric>(Metric::Create(name, &generic_parameters_)));
        mparam_.contain_eval_metrics = 1;
      }
    }
    for (auto& p_metric : metrics_) {
      p_metric->Configure(args);
    }
  }

  /**
   * Get number of targets from objective function.
   */
  ObjInfo ConfigureTargets() {
    CHECK(this->obj_);
    auto const& cache = this->GetPredictionCache()->Container();
    size_t n_targets = 1;
    for (auto const& d : cache) {
      if (n_targets == 1) {
        n_targets = this->obj_->Targets(d.first->Info());
      } else {
        auto t = this->obj_->Targets(d.first->Info());
        CHECK(n_targets == t || 1 == t) << "Inconsistent labels.";
      }
    }
    if (mparam_.num_target != 1) {
      CHECK(n_targets == 1 || n_targets == mparam_.num_target)
          << "Inconsistent configuration of num_target.  Configuration result from input data:"
          << n_targets << ", configuration from parameter:" << mparam_.num_target;
    } else {
      mparam_.num_target = n_targets;
    }
    return this->obj_->Task();
  }
};

std::string const LearnerConfiguration::kEvalMetric {"eval_metric"};  // NOLINT

class LearnerIO : public LearnerConfiguration {
 private:
  std::set<std::string> saved_configs_ = {"num_round"};
  // Used to identify the offset of JSON string when
  // Will be removed once JSON takes over.  Right now we still loads some RDS files from R.
  std::string const serialisation_header_ { u8"CONFIG-offset:" };

 public:
  explicit LearnerIO(std::vector<std::shared_ptr<DMatrix>> cache) : LearnerConfiguration{cache} {}

  void LoadModel(Json const& in) override {
    CHECK(IsA<Object>(in));
    auto version = Version::Load(in);
    if (std::get<0>(version) == 1 && std::get<1>(version) < 6) {
      LOG(WARNING)
          << "Found JSON model saved before XGBoost 1.6, please save the model using current "
             "version again. The support for old JSON model will be discontinued in XGBoost 2.3.";
    }

    auto const& learner = get<Object>(in["learner"]);
    mparam_.FromJson(learner.at("learner_model_param"));

    auto const& objective_fn = learner.at("objective");

    std::string name = get<String>(objective_fn["name"]);
    tparam_.UpdateAllowUnknown(Args{{"objective", name}});
    obj_.reset(ObjFunction::Create(name, &generic_parameters_));
    obj_->LoadConfig(objective_fn);

    auto const& gradient_booster = learner.at("gradient_booster");
    name = get<String>(gradient_booster["name"]);
    tparam_.UpdateAllowUnknown(Args{{"booster", name}});
    gbm_.reset(
        GradientBooster::Create(tparam_.booster, &generic_parameters_, &learner_model_param_));
    gbm_->LoadModel(gradient_booster);

    auto const& j_attributes = get<Object const>(learner.at("attributes"));
    attributes_.clear();
    for (auto const& kv : j_attributes) {
      attributes_[kv.first] = get<String const>(kv.second);
    }

    // feature names and types are saved in xgboost 1.4
    auto it = learner.find("feature_names");
    if (it != learner.cend()) {
      auto const& feature_names = get<Array const>(it->second);
      feature_names_.resize(feature_names.size());
      std::transform(feature_names.cbegin(), feature_names.cend(), feature_names_.begin(),
                     [](Json const& fn) { return get<String const>(fn); });
    }
    it = learner.find("feature_types");
    if (it != learner.cend()) {
      auto const& feature_types = get<Array const>(it->second);
      feature_types_.resize(feature_types.size());
      std::transform(feature_types.cbegin(), feature_types.cend(), feature_types_.begin(),
                     [](Json const& fn) { return get<String const>(fn); });
    }

    this->need_configuration_ = true;
  }

  void SaveModel(Json* p_out) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";

    Version::Save(p_out);
    Json& out { *p_out };

    out["learner"] = Object();
    auto& learner = out["learner"];

    learner["learner_model_param"] = mparam_.ToJson();
    learner["gradient_booster"] = Object();
    auto& gradient_booster = learner["gradient_booster"];
    gbm_->SaveModel(&gradient_booster);

    learner["objective"] = Object();
    auto& objective_fn = learner["objective"];
    obj_->SaveConfig(&objective_fn);

    learner["attributes"] = Object();
    for (auto const& kv : attributes_) {
      learner["attributes"][kv.first] = String(kv.second);
    }

    learner["feature_names"] = Array();
    auto& feature_names = get<Array>(learner["feature_names"]);
    for (auto const& name : feature_names_) {
      feature_names.emplace_back(name);
    }
    learner["feature_types"] = Array();
    auto& feature_types = get<Array>(learner["feature_types"]);
    for (auto const& type : feature_types_) {
      feature_types.emplace_back(type);
    }
  }

  // About to be deprecated by JSON format
  void LoadModel(dmlc::Stream* fi) override {
    generic_parameters_.UpdateAllowUnknown(Args{});
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

    // FIXME(jiamingy): Move this out of learner after the old binary model is remove.
    auto first_non_space = [&](std::string::const_iterator beg, std::string::const_iterator end) {
      for (auto i = beg; i != end; ++i) {
        if (!std::isspace(*i)) {
          return i;
        }
      }
      return end;
    };

    if (header[0] == '{') {  // Dispatch to JSON
      auto buffer = common::ReadAll(fi, &fp);
      Json model;
      auto it = first_non_space(buffer.cbegin() + 1, buffer.cend());
      if (it != buffer.cend() && *it == '"') {
        model = Json::Load(StringView{buffer});
      } else if (it != buffer.cend() && std::isalpha(*it)) {
        model = Json::Load(StringView{buffer}, std::ios::binary);
      } else {
        LOG(FATAL) << "Invalid model format";
      }
      this->LoadModel(model);
      return;
    }

    // use the peekable reader.
    fi = &fp;
    // read parameter
    CHECK_EQ(fi->Read(&mparam_, sizeof(mparam_)), sizeof(mparam_))
        << "BoostLearner: wrong model format";
    if (!DMLC_IO_NO_ENDIAN_SWAP) {
      mparam_ = mparam_.ByteSwap();
    }
    if (mparam_.num_target == 0) {
      mparam_.num_target = 1;
    }
    CHECK(fi->Read(&tparam_.objective)) << "BoostLearner: wrong model format";
    CHECK(fi->Read(&tparam_.booster)) << "BoostLearner: wrong model format";

    obj_.reset(ObjFunction::Create(tparam_.objective, &generic_parameters_));
    gbm_.reset(GradientBooster::Create(tparam_.booster, &generic_parameters_,
                                       &learner_model_param_));
    gbm_->Load(fi);
    if (mparam_.contain_extra_attrs != 0) {
      std::vector<std::pair<std::string, std::string> > attr;
      fi->Read(&attr);
      for (auto& kv : attr) {
        const std::string prefix = "SAVED_PARAM_";
        if (kv.first.find(prefix) == 0) {
          const std::string saved_param = kv.first.substr(prefix.length());
          if (saved_configs_.find(saved_param) != saved_configs_.end()) {
            cfg_[saved_param] = kv.second;
          }
        }
      }
      attributes_ = std::map<std::string, std::string>(attr.begin(), attr.end());
    }
    bool warn_old_model { false };
    if (attributes_.find("count_poisson_max_delta_step") != attributes_.cend()) {
      // Loading model from < 1.0.0, objective is not saved.
      cfg_["max_delta_step"] = attributes_.at("count_poisson_max_delta_step");
      attributes_.erase("count_poisson_max_delta_step");
      warn_old_model = true;
    } else {
      warn_old_model = false;
    }

    if (mparam_.major_version < 1) {
      // Before 1.0.0, base_score is saved as a transformed value, and there's no version
      // attribute (saved a 0) in the saved model.
      std::string multi{"multi:"};
      if (!std::equal(multi.cbegin(), multi.cend(), tparam_.objective.cbegin())) {
        HostDeviceVector<float> t;
        t.HostVector().resize(1);
        t.HostVector().at(0) = mparam_.base_score;
        this->obj_->PredTransform(&t);
        auto base_score = t.HostVector().at(0);
        mparam_.base_score = base_score;
      }
      warn_old_model = true;
    }

    learner_model_param_ =
        LearnerModelParam(mparam_, obj_->ProbToMargin(mparam_.base_score), obj_->Task());
    if (attributes_.find("objective") != attributes_.cend()) {
      auto obj_str = attributes_.at("objective");
      auto j_obj = Json::Load({obj_str.c_str(), obj_str.size()});
      obj_->LoadConfig(j_obj);
      attributes_.erase("objective");
    } else {
      warn_old_model = true;
    }
    if (attributes_.find("metrics") != attributes_.cend()) {
      auto metrics_str = attributes_.at("metrics");
      std::vector<std::string> names { common::Split(metrics_str, ';') };
      attributes_.erase("metrics");
      for (auto const& n : names) {
        this->SetParam(kEvalMetric, n);
      }
    }

    if (warn_old_model) {
      LOG(WARNING) << "Loading model from XGBoost < 1.0.0, consider saving it "
                      "again for improved compatibility";
    }

    // Renew the version.
    mparam_.major_version = std::get<0>(Version::Self());
    mparam_.minor_version = std::get<1>(Version::Self());

    cfg_["num_class"] = common::ToString(mparam_.num_class);
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);

    auto n = tparam_.__DICT__();
    cfg_.insert(n.cbegin(), n.cend());

    // copy dsplit from config since it will not run again during restore
    if (tparam_.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      tparam_.dsplit = DataSplitMode::kRow;
    }

    this->need_configuration_ = true;
  }

  // Save model into binary format.  The code is about to be deprecated by more robust
  // JSON serialization format.
  void SaveModel(dmlc::Stream* fo) const override {
    LearnerModelParamLegacy mparam = mparam_;  // make a copy to potentially modify
    std::vector<std::pair<std::string, std::string> > extra_attr;
    mparam.contain_extra_attrs = 1;

    {
      std::vector<std::string> saved_params;
      for (const auto& key : saved_params) {
        auto it = cfg_.find(key);
        if (it != cfg_.end()) {
          mparam.contain_extra_attrs = 1;
          extra_attr.emplace_back("SAVED_PARAM_" + key, it->second);
        }
      }
    }
    {
      // Similar to JSON model IO, we save the objective.
      Json j_obj { Object() };
      obj_->SaveConfig(&j_obj);
      std::string obj_doc;
      Json::Dump(j_obj, &obj_doc);
      extra_attr.emplace_back("objective", obj_doc);
    }
    // As of 1.0.0, JVM Package and R Package uses Save/Load model for serialization.
    // Remove this part once they are ported to use actual serialization methods.
    if (mparam.contain_eval_metrics != 0) {
      std::stringstream os;
      for (auto& ev : metrics_) {
        os << ev->Name() << ";";
      }
      extra_attr.emplace_back("metrics", os.str());
    }
    std::string header {"binf"};
    fo->Write(header.data(), 4);
    if (DMLC_IO_NO_ENDIAN_SWAP) {
      fo->Write(&mparam, sizeof(LearnerModelParamLegacy));
    } else {
      LearnerModelParamLegacy x = mparam.ByteSwap();
      fo->Write(&x, sizeof(LearnerModelParamLegacy));
    }
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
  }

  void Save(dmlc::Stream* fo) const override {
    Json memory_snapshot{Object()};
    memory_snapshot["Model"] = Object();
    auto& model = memory_snapshot["Model"];
    this->SaveModel(&model);
    memory_snapshot["Config"] = Object();
    auto& config = memory_snapshot["Config"];
    this->SaveConfig(&config);

    std::vector<char> stream;
    Json::Dump(memory_snapshot, &stream, std::ios::binary);
    fo->Write(stream.data(), stream.size());
  }

  void Load(dmlc::Stream* fi) override {
    common::PeekableInStream fp(fi);
    char header[2];
    fp.PeekRead(header, 2);
    if (header[0] == '{') {
      auto buffer = common::ReadAll(fi, &fp);
      Json memory_snapshot;
      if (header[1] == '"') {
        memory_snapshot = Json::Load(StringView{buffer});
        LOG(WARNING) << ModelMsg();
      } else if (std::isalpha(header[1])) {
        memory_snapshot = Json::Load(StringView{buffer}, std::ios::binary);
      } else {
        LOG(FATAL) << "Invalid serialization file.";
      }
      if (IsA<Null>(memory_snapshot["Model"])) {
        // R has xgb.load that doesn't distinguish whether configuration is saved.
        // We should migrate to use `xgb.load.raw` instead.
        this->LoadModel(memory_snapshot);
      } else {
        this->LoadModel(memory_snapshot["Model"]);
        this->LoadConfig(memory_snapshot["Config"]);
      }
    } else {
      std::string header;
      header.resize(serialisation_header_.size());
      CHECK_EQ(fp.Read(&header[0], header.size()), serialisation_header_.size());
      // Avoid printing the content in loaded header, which might be random binary code.
      CHECK(header == serialisation_header_) << ModelMsg();
      int64_t sz {-1};
      CHECK_EQ(fp.Read(&sz, sizeof(sz)), sizeof(sz));
      if (!DMLC_IO_NO_ENDIAN_SWAP) {
        dmlc::ByteSwap(&sz, sizeof(sz), 1);
      }
      CHECK_GT(sz, 0);
      size_t json_offset = static_cast<size_t>(sz);
      std::string buffer;
      common::FixedSizeStream{&fp}.Take(&buffer);

      common::MemoryFixSizeBuffer binary_buf(&buffer[0], json_offset);
      this->LoadModel(&binary_buf);

      auto config = Json::Load({buffer.c_str() + json_offset, buffer.size() - json_offset});
      this->LoadConfig(config);
    }
  }
};

/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
class LearnerImpl : public LearnerIO {
 public:
  explicit LearnerImpl(std::vector<std::shared_ptr<DMatrix> > cache)
      : LearnerIO{cache} {}
  ~LearnerImpl() override {
    auto local_map = LearnerAPIThreadLocalStore::Get();
    if (local_map->find(this) != local_map->cend()) {
      local_map->erase(this);
    }
  }
  // Configuration before data is known.
  void CheckDataSplitMode() {
    if (rabit::IsDistributed()) {
      CHECK(tparam_.dsplit != DataSplitMode::kAuto)
        << "Precondition violated; dsplit cannot be 'auto' in distributed mode";
      if (tparam_.dsplit == DataSplitMode::kCol) {
        LOG(FATAL) << "Column-wise data split is currently not supported.";
      }
    }
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) override {
    this->Configure();
    return gbm_->DumpModel(fmap, with_stats, format);
  }

  Learner *Slice(int32_t begin_layer, int32_t end_layer, int32_t step,
                 bool *out_of_bound) override {
    this->Configure();
    CHECK_NE(this->learner_model_param_.num_feature, 0);
    CHECK_GE(begin_layer, 0);
    auto *out_impl = new LearnerImpl({});
    out_impl->learner_model_param_ = this->learner_model_param_;
    out_impl->generic_parameters_ = this->generic_parameters_;
    auto gbm = std::unique_ptr<GradientBooster>(GradientBooster::Create(
        this->tparam_.booster, &out_impl->generic_parameters_,
        &out_impl->learner_model_param_));
    this->gbm_->Slice(begin_layer, end_layer, step, gbm.get(), out_of_bound);
    out_impl->gbm_ = std::move(gbm);

    Json config { Object() };
    this->SaveConfig(&config);
    out_impl->mparam_ = this->mparam_;
    out_impl->attributes_ = this->attributes_;
    out_impl->SetFeatureNames(this->feature_names_);
    out_impl->SetFeatureTypes(this->feature_types_);
    out_impl->LoadConfig(config);
    out_impl->Configure();
    CHECK_EQ(out_impl->learner_model_param_.num_feature, this->learner_model_param_.num_feature);
    CHECK_NE(out_impl->learner_model_param_.num_feature, 0);

    auto erase_attr = [&](std::string attr) {
      // Erase invalid attributes.
      auto attr_it = out_impl->attributes_.find(attr);
      if (attr_it != out_impl->attributes_.cend()) {
        out_impl->attributes_.erase(attr_it);
      }
    };
    erase_attr("best_iteration");
    erase_attr("best_score");
    return out_impl;
  }

  void UpdateOneIter(int iter, std::shared_ptr<DMatrix> train) override {
    monitor_.Start("UpdateOneIter");
    TrainingObserver::Instance().Update(iter);
    this->Configure();
    if (generic_parameters_.seed_per_iteration) {
      common::GlobalRandom().seed(generic_parameters_.seed * kRandSeedMagic + iter);
    }

    this->CheckDataSplitMode();
    this->ValidateDMatrix(train.get(), true);

    auto local_cache = this->GetPredictionCache();
    auto& predt = local_cache->Cache(train, generic_parameters_.gpu_id);

    monitor_.Start("PredictRaw");
    this->PredictRaw(train.get(), &predt, true, 0, 0);
    TrainingObserver::Instance().Observe(predt.predictions, "Predictions");
    monitor_.Stop("PredictRaw");

    monitor_.Start("GetGradient");
    obj_->GetGradient(predt.predictions, train->Info(), iter, &gpair_);
    monitor_.Stop("GetGradient");
    TrainingObserver::Instance().Observe(gpair_, "Gradients");

    gbm_->DoBoost(train.get(), &gpair_, &predt, obj_.get());
    monitor_.Stop("UpdateOneIter");
  }

  void BoostOneIter(int iter, std::shared_ptr<DMatrix> train,
                    HostDeviceVector<GradientPair>* in_gpair) override {
    monitor_.Start("BoostOneIter");
    this->Configure();
    if (generic_parameters_.seed_per_iteration) {
      common::GlobalRandom().seed(generic_parameters_.seed * kRandSeedMagic + iter);
    }

    this->CheckDataSplitMode();
    this->ValidateDMatrix(train.get(), true);
    auto local_cache = this->GetPredictionCache();
    local_cache->Cache(train, generic_parameters_.gpu_id);

    gbm_->DoBoost(train.get(), in_gpair, &local_cache->Entry(train.get()), obj_.get());
    monitor_.Stop("BoostOneIter");
  }

  std::string EvalOneIter(int iter,
                          const std::vector<std::shared_ptr<DMatrix>>& data_sets,
                          const std::vector<std::string>& data_names) override {
    monitor_.Start("EvalOneIter");
    this->Configure();

    std::ostringstream os;
    os.precision(std::numeric_limits<double>::max_digits10);
    os << '[' << iter << ']' << std::setiosflags(std::ios::fixed);
    if (metrics_.size() == 0 && tparam_.disable_default_eval_metric <= 0) {
      metrics_.emplace_back(Metric::Create(obj_->DefaultEvalMetric(), &generic_parameters_));
      metrics_.back()->Configure({cfg_.begin(), cfg_.end()});
    }

    auto local_cache = this->GetPredictionCache();
    for (size_t i = 0; i < data_sets.size(); ++i) {
      std::shared_ptr<DMatrix> m = data_sets[i];
      auto &predt = local_cache->Cache(m, generic_parameters_.gpu_id);
      this->ValidateDMatrix(m.get(), false);
      this->PredictRaw(m.get(), &predt, false, 0, 0);

      auto &out = output_predictions_.Cache(m, generic_parameters_.gpu_id).predictions;
      out.Resize(predt.predictions.Size());
      out.Copy(predt.predictions);

      obj_->EvalTransform(&out);
      for (auto& ev : metrics_) {
        os << '\t' << data_names[i] << '-' << ev->Name() << ':' << ev->Eval(out, m->Info());
      }
    }

    monitor_.Stop("EvalOneIter");
    return os.str();
  }

  void Predict(std::shared_ptr<DMatrix> data, bool output_margin,
               HostDeviceVector<bst_float> *out_preds, unsigned layer_begin,
               unsigned layer_end, bool training,
               bool pred_leaf, bool pred_contribs, bool approx_contribs,
               bool pred_interactions) override {
    int multiple_predictions = static_cast<int>(pred_leaf) +
                               static_cast<int>(pred_interactions) +
                               static_cast<int>(pred_contribs);
    this->Configure();
    CHECK_LE(multiple_predictions, 1) << "Perform one kind of prediction at a time.";
    if (pred_contribs) {
      gbm_->PredictContribution(data.get(), out_preds, layer_begin, layer_end, approx_contribs);
    } else if (pred_interactions) {
      gbm_->PredictInteractionContributions(data.get(), out_preds, layer_begin, layer_end,
                                            approx_contribs);
    } else if (pred_leaf) {
      gbm_->PredictLeaf(data.get(), out_preds, layer_begin, layer_end);
    } else {
      auto local_cache = this->GetPredictionCache();
      auto& prediction = local_cache->Cache(data, generic_parameters_.gpu_id);
      this->PredictRaw(data.get(), &prediction, training, layer_begin, layer_end);
      // Copy the prediction cache to output prediction. out_preds comes from C API
      out_preds->SetDevice(generic_parameters_.gpu_id);
      out_preds->Resize(prediction.predictions.Size());
      out_preds->Copy(prediction.predictions);
      if (!output_margin) {
        obj_->PredTransform(out_preds);
      }
    }
  }

  int32_t BoostedRounds() const override {
    if (!this->gbm_) { return 0; }  // haven't call train or LoadModel.
    CHECK(!this->need_configuration_);
    return this->gbm_->BoostedRounds();
  }
  uint32_t Groups() const override {
    CHECK(!this->need_configuration_);
    return this->learner_model_param_.num_output_group;
  }

  XGBAPIThreadLocalEntry& GetThreadLocal() const override {
    return (*LearnerAPIThreadLocalStore::Get())[this];
  }

  void InplacePredict(std::shared_ptr<DMatrix> p_m, PredictionType type, float missing,
                      HostDeviceVector<bst_float>** out_preds, uint32_t iteration_begin,
                      uint32_t iteration_end) override {
    this->Configure();
    auto& out_predictions = this->GetThreadLocal().prediction_entry;
    this->gbm_->InplacePredict(p_m, missing, &out_predictions, iteration_begin, iteration_end);
    if (type == PredictionType::kValue) {
      obj_->PredTransform(&out_predictions.predictions);
    } else if (type == PredictionType::kMargin) {
      // do nothing
    } else {
      LOG(FATAL) << "Unsupported prediction type:" << static_cast<int>(type);
    }
    *out_preds = &out_predictions.predictions;
  }

  void CalcFeatureScore(std::string const& importance_type, common::Span<int32_t const> trees,
                        std::vector<bst_feature_t>* features, std::vector<float>* scores) override {
    this->Configure();
    gbm_->FeatureScore(importance_type, trees, features, scores);
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
   * \param training allow dropout when the DART booster is being used
   */
  void PredictRaw(DMatrix *data, PredictionCacheEntry *out_preds, bool training,
                  unsigned layer_begin, unsigned layer_end) const {
    CHECK(gbm_ != nullptr) << "Predict must happen after Load or configuration";
    this->ValidateDMatrix(data, false);
    gbm_->PredictBatch(data, out_preds, training, layer_begin, layer_end);
  }

  void ValidateDMatrix(DMatrix* p_fmat, bool is_training) const {
    MetaInfo const& info = p_fmat->Info();
    info.Validate(generic_parameters_.gpu_id);

    auto const row_based_split = [this]() {
      return tparam_.dsplit == DataSplitMode::kRow ||
             tparam_.dsplit == DataSplitMode::kAuto;
    };
    if (row_based_split()) {
      if (is_training) {
        CHECK_EQ(learner_model_param_.num_feature, p_fmat->Info().num_col_)
            << "Number of columns does not match number of features in "
               "booster.";
      } else {
        CHECK_GE(learner_model_param_.num_feature, p_fmat->Info().num_col_)
            << "Number of columns does not match number of features in "
               "booster.";
      }
    }

    if (p_fmat->Info().num_row_ == 0) {
      LOG(WARNING) << "Empty dataset at worker: " << rabit::GetRank();
    }
  }

 private:
  /*! \brief random number transformation seed. */
  static int32_t constexpr kRandSeedMagic = 127;
  // gradient pairs
  HostDeviceVector<GradientPair> gpair_;
  /*! \brief Temporary storage to prediction.  Useful for storing data transformed by
   *  objective function */
  PredictionContainer output_predictions_;
};

constexpr int32_t LearnerImpl::kRandSeedMagic;

Learner* Learner::Create(
    const std::vector<std::shared_ptr<DMatrix> >& cache_data) {
  return new LearnerImpl(cache_data);
}
}  // namespace xgboost
