/**
 * Copyright 2014-2025, XGBoost Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include "xgboost/learner.h"

#include <dmlc/io.h>                      // for Stream
#include <dmlc/parameter.h>               // for FieldEntry, DMLC_DECLARE_FIELD, Parameter, DMLC...
#include <dmlc/thread_local.h>            // for ThreadLocalStore

#include <algorithm>                      // for equal, max, transform, sort, find_if, all_of
#include <atomic>                         // for atomic
#include <cctype>                         // for isalpha, isspace
#include <cmath>                          // for isnan, isinf
#include <cstdint>                        // for int32_t, uint32_t, int64_t, uint64_t
#include <cstdlib>                        // for atoi
#include <cstring>                        // for memcpy, size_t, memset
#include <iomanip>                        // for operator<<, setiosflags
#include <iterator>                       // for back_insert_iterator, distance, back_inserter
#include <limits>                         // for numeric_limits
#include <memory>                         // for allocator, unique_ptr, shared_ptr, operator==
#include <mutex>                          // for mutex, lock_guard
#include <sstream>                        // for operator<<, basic_ostream, basic_ostream::opera...
#include <stack>                          // for stack
#include <string>                         // for basic_string, char_traits, operator<, string
#include <system_error>                   // for errc
#include <unordered_map>                  // for operator!=, unordered_map
#include <utility>                        // for pair, as_const, move, swap
#include <vector>                         // for vector

#include "collective/aggregator.h"        // for ApplyWithLabels
#include "collective/communicator-inl.h"  // for Allreduce, Broadcast, GetRank, IsDistributed
#include "common/api_entry.h"             // for XGBAPIThreadLocalEntry
#include "common/param_array.h"           // for ParamArray
#include "common/charconv.h"              // for to_chars, to_chars_result, NumericLimits, from_...
#include "common/error_msg.h"             // for MaxFeatureSize, WarnOldSerialization, ...
#include "common/io.h"                    // for PeekableInStream, ReadAll, FixedSizeStream, Mem...
#include "common/observer.h"              // for TrainingObserver
#include "common/random.h"                // for GlobalRandom
#include "common/timer.h"                 // for Monitor
#include "common/version.h"               // for Version
#include "xgboost/base.h"                 // for Args, GradientPair, bst_feature_t
#include "xgboost/context.h"              // for Context
#include "xgboost/data.h"                 // for DMatrix, MetaInfo
#include "xgboost/gbm.h"                  // for GradientBooster
#include "xgboost/global_config.h"        // for GlobalConfiguration, GlobalConfigThreadLocalStore
#include "xgboost/host_device_vector.h"   // for HostDeviceVector
#include "xgboost/json.h"                 // for Json, get, Object, String, IsA, Array, ToJson
#include "xgboost/linalg.h"               // for Vector, VectorView
#include "xgboost/logging.h"              // for CHECK, LOG, CHECK_EQ
#include "xgboost/metric.h"               // for Metric
#include "xgboost/objective.h"            // for ObjFunction
#include "xgboost/parameter.h"            // for DECLARE_FIELD_ENUM_CLASS, XGBoostParameter
#include "xgboost/predictor.h"            // for PredictionContainer, PredictionCacheEntry
#include "xgboost/string_view.h"          // for operator<<, StringView
#include "xgboost/task.h"                 // for ObjInfo

namespace {
const char* kMaxDeltaStepDefaultValue = "0.7";
}  // anonymous namespace

DECLARE_FIELD_ENUM_CLASS(xgboost::MultiStrategy);

namespace xgboost {
Learner::~Learner() = default;
namespace {
StringView ModelNotFitted() { return "Model is not yet initialized (not fitted)."; }

template <typename T>
T& UsePtr(T& ptr) {  // NOLINT
  CHECK(ptr);
  return ptr;
}
}  // anonymous namespace

/*! \brief training parameter for regression
 *
 * Should be deprecated, but still used for being compatible with binary IO.
 * Once it's gone, `LearnerModelParam` should handle transforming `base_score`
 * with objective by itself.
 */
struct LearnerModelParamLegacy : public dmlc::Parameter<LearnerModelParamLegacy> {
  /** @brief Global bias/intercept. */
  common::ParamArray<float> base_score{"base_score"};
  /** @brief number of features  */
  bst_feature_t num_feature{0};
  /** @brief number of classes, if it is multi-class classification, 0 otherwise.  */
  std::int32_t num_class{0};
  /**! @brief the version of XGBoost. */
  std::int32_t major_version{std::get<0>(Version::Self())};
  std::int32_t minor_version{std::get<1>(Version::Self())};
  /**
   * @brief Number of target variables.
   */
  bst_target_t num_target{1};
  /**
   * @brief Whether we should calculate the base score from training data.
   *
   *   This is a private parameter as we can't expose it as boolean due to binary model
   *   format. Exposing it as integer creates inconsistency with other parameters.
   *
   *   Automatically disabled when base_score is specifed by user. int32 is used instead
   *   of bool for the ease of serialization.
   */
  std::int32_t boost_from_average{true};

  LearnerModelParamLegacy() = default;

  [[nodiscard]] Json ToJson() const {
    Json obj{Object{}};
    std::stringstream ss;
    ss << base_score;
    obj["base_score"] = ss.str();

    char integers[NumericLimits<int64_t>::kToCharsSize];
    auto ret = to_chars(integers, integers + NumericLimits<int64_t>::kToCharsSize,
                        static_cast<int64_t>(num_feature));
    CHECK(ret.ec == std::errc());
    obj["num_feature"] =
        std::string{integers, static_cast<size_t>(std::distance(integers, ret.ptr))};
    ret = to_chars(integers, integers + NumericLimits<int64_t>::kToCharsSize,
                   static_cast<int64_t>(num_class));
    CHECK(ret.ec == std::errc());
    obj["num_class"] = std::string{integers, static_cast<size_t>(std::distance(integers, ret.ptr))};

    ret = to_chars(integers, integers + NumericLimits<int64_t>::kToCharsSize,
                   static_cast<int64_t>(num_target));
    obj["num_target"] =
        std::string{integers, static_cast<size_t>(std::distance(integers, ret.ptr))};

    ret = to_chars(integers, integers + NumericLimits<std::int64_t>::kToCharsSize,
                   static_cast<std::int64_t>(boost_from_average));
    obj["boost_from_average"] =
        std::string{integers, static_cast<std::size_t>(std::distance(integers, ret.ptr))};

    return obj;
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
    auto bse_it = j_param.find("boost_from_average");
    if (bse_it != j_param.cend()) {
      m["boost_from_average"] = get<String const>(bse_it->second);
    }
    std::string str = get<String const>(j_param.at("base_score"));
    m["base_score"] = str;
    this->Init(m);
    this->HandleOldFormat();
  }
  // Handle old model formats, before 3.1, the intercept was always a scalar.
  void HandleOldFormat() {
    if (this->base_score.size() == 1 && this->OutputLength() > 1) {
      this->base_score.Resize(this->OutputLength(), this->base_score[0]);
    }
  }

  template <typename Container>
  Args UpdateAllowUnknown(Container const& kwargs) {
    // Detect whether user has made their own base score.
    auto has_key = [&kwargs](char const* key) {
      return std::find_if(kwargs.cbegin(), kwargs.cend(),
                          [key](auto const& kv) { return kv.first == key; }) != kwargs.cend();
    };
    if (has_key("base_score")) {
      this->boost_from_average = false;
    }
    return dmlc::Parameter<LearnerModelParamLegacy>::UpdateAllowUnknown(kwargs);
  }
  // The number of outputs of the model.
  [[nodiscard]] bst_target_t OutputLength() const noexcept {
    return std::max({this->num_target, static_cast<bst_target_t>(this->num_class),
                     static_cast<bst_target_t>(1)});
  }

  // Sanity checks
  void Validate(Context const* ctx) const {
    this->ValidateLength();
    CHECK(std::none_of(base_score.cbegin(), base_score.cend(),
                       [](float v) { return std::isnan(v) || std::isinf(v); }));

    if (!collective::IsDistributed()) {
      return;
    }

    std::vector<char> data;
    Json::Dump(this->ToJson(), &data, std::ios::binary);
    std::vector<char> sync{data};

    auto rc = collective::Broadcast(ctx, linalg::MakeVec(sync.data(), sync.size()), 0);
    collective::SafeColl(rc);

    CHECK(std::equal(data.cbegin(), data.cend(), sync.cbegin()))
        << "Different model parameter across workers:\n\t"
        << Json::Load(StringView{data.data(), data.size()}, std::ios::binary) << "\nvs.\n\t"
        << Json::Load(StringView{sync.data(), sync.size()}, std::ios::binary);
  }

  void ValidateLength() const {
    CHECK_GE(this->base_score.size(), 1);
    std::size_t n_classes = static_cast<std::size_t>(num_class),
                n_targets = static_cast<std::size_t>(num_target);
    if (!(base_score.size() == n_classes || base_score.size() == n_targets)) {
      error::InvalidIntercept(n_classes, n_targets, base_score.size());
    }
  }

  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerModelParamLegacy) {
    DMLC_DECLARE_FIELD(base_score)
        .describe("Global bias of the model.")
        .set_default(common::ParamArray<float>{"base_score"});
    DMLC_DECLARE_FIELD(num_feature)
        .set_default(0)
        .describe(
            "Number of features in training data, this parameter will be automatically detected by "
            "learner.");
    DMLC_DECLARE_FIELD(num_class).set_default(0).set_lower_bound(0).describe(
        "Number of class option for multi-class classifier. "
        " By default equals 0 and corresponds to binary classifier.");
    DMLC_DECLARE_FIELD(num_target)
        .set_default(1)
        .set_lower_bound(1)
        .describe("Number of output targets. Can be set automatically if not specified.");
    DMLC_DECLARE_FIELD(boost_from_average)
        .set_default(true)
        .describe("Whether we should calculate the base score from training data.");
  }
};
}  // namespace xgboost

namespace xgboost {
LearnerModelParam::LearnerModelParam(LearnerModelParamLegacy const& user_param, ObjInfo t,
                                     MultiStrategy multi_strategy)
    : num_feature{user_param.num_feature},
      num_output_group{user_param.OutputLength()},
      task{t},
      multi_strategy{multi_strategy} {
  if (user_param.num_class > 1 && user_param.num_target > 1) {
    LOG(FATAL) << "multi-target-multi-class is not yet supported. Output classes:"
               << user_param.num_class << ", output targets:" << user_param.num_target;
  }
}

LearnerModelParam::LearnerModelParam(Context const* ctx, LearnerModelParamLegacy const& user_param,
                                     linalg::Vector<float> base_score, ObjInfo t,
                                     MultiStrategy multi_strategy)
    : LearnerModelParam{user_param, t, multi_strategy} {
  std::swap(base_score_, base_score);
  // Make sure read access everywhere for thread-safe prediction.
  std::as_const(base_score_).HostView();
  if (ctx->IsCUDA()) {
    std::as_const(base_score_).View(ctx->Device());
  }
  CHECK(std::as_const(base_score_).Data()->HostCanRead());
}

linalg::VectorView<float const> LearnerModelParam::BaseScore(DeviceOrd device) const {
  // multi-class is not yet supported.
  CHECK_GE(base_score_.Size(), 1) << ModelNotFitted();
  if (!device.IsCUDA()) {
    // Make sure that we won't run into race condition.
    CHECK(base_score_.Data()->HostCanRead());
    return base_score_.HostView();
  }
  // Make sure that we won't run into race condition.
  CHECK(base_score_.Data()->DeviceCanRead());
  auto v = base_score_.View(device);
  CHECK(base_score_.Data()->HostCanRead());  // make sure read access is not removed.
  return v;
}

linalg::VectorView<float const> LearnerModelParam::BaseScore(Context const* ctx) const {
  return this->BaseScore(ctx->Device());
}

void LearnerModelParam::Copy(LearnerModelParam const& that) {
  base_score_.Reshape(that.base_score_.Shape());
  base_score_.Data()->SetDevice(that.base_score_.Device());
  base_score_.Data()->Copy(*that.base_score_.Data());
  std::as_const(base_score_).HostView();
  if (!that.base_score_.Device().IsCPU()) {
    std::as_const(base_score_).View(that.base_score_.Device());
  }
  CHECK_EQ(base_score_.Data()->DeviceCanRead(), that.base_score_.Data()->DeviceCanRead());
  CHECK(base_score_.Data()->HostCanRead());

  num_feature = that.num_feature;
  num_output_group = that.num_output_group;
  task = that.task;
  multi_strategy = that.multi_strategy;
}

struct LearnerTrainParam : public XGBoostParameter<LearnerTrainParam> {
  // flag to disable default metric
  bool disable_default_eval_metric {false};
  // FIXME(trivialfis): The following parameters belong to model itself, but can be
  // specified by users.  Move them to model parameter once we can get rid of binary IO.
  std::string booster;
  std::string objective;
  // This is a training parameter and is not saved (nor loaded) in the model.
  MultiStrategy multi_strategy{MultiStrategy::kOneOutputPerTree};

  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerTrainParam) {
    DMLC_DECLARE_FIELD(disable_default_eval_metric)
        .set_default(false)
        .describe("Flag to disable default metric. Set to >0 to disable");
    DMLC_DECLARE_FIELD(booster).set_default("gbtree").describe(
        "Gradient booster used for training.");
    DMLC_DECLARE_FIELD(objective)
        .set_default("reg:squarederror")
        .describe("Objective function used for obtaining gradient.");
    DMLC_DECLARE_FIELD(multi_strategy)
        .add_enum("one_output_per_tree", MultiStrategy::kOneOutputPerTree)
        .add_enum("multi_output_tree", MultiStrategy::kMultiOutputTree)
        .set_default(MultiStrategy::kOneOutputPerTree)
        .describe(
            "Strategy used for training multi-target models. `multi_output_tree` means building "
            "one single tree for all targets.");
  }
};


DMLC_REGISTER_PARAMETER(LearnerModelParamLegacy);
DMLC_REGISTER_PARAMETER(LearnerTrainParam);

using LearnerAPIThreadLocalStore =
    dmlc::ThreadLocalStore<std::map<Learner const *, XGBAPIThreadLocalEntry>>;

namespace {
/**
 * @brief Handler for the `n_targets` property and the intercept.
 */
class Intercept : public Learner {
  using CacheT = common::GetValueT<decltype(std::declval<PredictionContainer>().Container())>;

 protected:
  /**
   * @brief User-provided model parameter.
   *
   * This parameter is the most difficult one in XGBoost. It stores basic properties of
   * the booster model and is saved as part of the booster. We need to configure it
   * automatically from input training data while taking user-provided parameters into
   * account.
   *
   * It's difficult because XGBoost has an interface that exposes many states. For
   * instance, we need to have a valid model after configuration, without seeing the
   * training data. This exposes a partially initialized model that's semi-valid.
   */
  LearnerModelParamLegacy mparam_;
  /**
   * @brief Internal model parameter.
   */
  LearnerModelParam learner_model_param_;

 private:
  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) {
    base_score->SetDevice(this->Ctx()->Device());
    base_score->Reshape(this->mparam_.OutputLength());
    collective::ApplyWithLabels(this->Ctx(), info, base_score->Data(),
                                [&] { UsePtr(obj_)->InitEstimation(info, base_score); });
  }

  [[nodiscard]] bool NeedFit() const {
    return this->mparam_.boost_from_average && !UsePtr(gbm_)->ModelFitted();
  }

  // Create the internal model parameter from user inputs, this requires the user input to
  // be initialized first.
  //
  // Don't apply the link function if the base_score is a dummy value.
  //
  // This function should be called for every `Configure` call ot make sure the base_score
  // is stored in the right place.
  void InitModelParam(LearnerTrainParam const& tparam, bool apply_link) {
    auto const& in = this->mparam_.base_score;
    auto task = UsePtr(this->obj_)->Task();
    linalg::Vector<float> base_score{in.cbegin(), in.cend(), {in.size()}, this->ctx_.Device()};
    if (apply_link) {
      UsePtr(this->obj_)->ProbToMargin(&base_score);
    }

    learner_model_param_ =
        LearnerModelParam{Ctx(), mparam_, std::move(base_score), task, tparam.multi_strategy};
  }

  /**
   * Get the number of targets from the cache using the objective function.
   */
  void GetNumTargets(CacheT const& cache) {
    CHECK(this->obj_);
    bst_target_t n_targets = 1;
    for (auto const& d : cache) {
      if (n_targets == 1) {
        n_targets = this->obj_->Targets(d.first.ptr->Info());
      } else {
        auto t = this->obj_->Targets(d.first.ptr->Info());
        CHECK(n_targets == t || 1 == t) << "Inconsistent labels.";
      }
    }

    if (mparam_.num_target > 1) {
      CHECK(n_targets == 1 || n_targets == mparam_.num_target)
          << "Inconsistent configuration of the `num_target`.  Configuration result from input "
          << "data:" << n_targets << ", configuration from parameters:" << mparam_.num_target;
    } else {
      mparam_.num_target = n_targets;
    }
  }

 protected:
  void CheckModelInitialized() const {
    CHECK(learner_model_param_.Initialized()) << ModelNotFitted();
    CHECK_NE(learner_model_param_.BaseScore(this->Ctx()).Size(), 0) << ModelNotFitted();
  }

  void InitModelUserParam(LearnerTrainParam const& tparam, CacheT const& cache) {
    this->GetNumTargets(cache);

    if (this->NeedFit()) {
      // Initialize with a sensible default value to get prediction/model io going.
      this->mparam_.base_score.Resize(this->mparam_.OutputLength(),
                                      ObjFunction::DefaultBaseScore());
      this->InitModelParam(tparam, false);
      // This should not be altered, we will estimate it later.
      CHECK(this->NeedFit());
    } else if (this->gbm_->ModelFitted()) {
      this->mparam_.ValidateLength();
      // Init with a valid (configured) mparam
      this->InitModelParam(tparam, true);
    } else {
      // user-provided
      this->mparam_.HandleOldFormat();
      this->InitModelParam(tparam, true);
    }
  }

  /**
   * @brief Calculate the `base_score` based on input data.
   *
   * @param p_fmat The training DMatrix used to estimate the base score.
   */
  void FitIntercept(LearnerTrainParam const& tparam, DMatrix const* p_fmat) {
    // Estimate the intercept if this is the first iteration.
    if (this->NeedFit()) {
      // The DMatrix can be null if a method other than training is called.
      if (p_fmat) {
        auto const& info = p_fmat->Info();
        info.Validate(Ctx()->Device());
        // We estimate it from the input data.
        linalg::Vector<float> base_score;
        this->InitEstimation(info, &base_score);

        mparam_.base_score = base_score.Data()->ConstHostVector();
      }
      this->InitModelParam(tparam, true);
      // Check whether the base score is valid.
      mparam_.Validate(&ctx_);
    }

    this->CheckModelInitialized();
  }
};
}  // namespace

class LearnerConfiguration : public Intercept {
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
  LearnerTrainParam tparam_;
  // Initial prediction.
  PredictionContainer prediction_container_;

  std::vector<std::string> metric_names_;

 public:
  explicit LearnerConfiguration(std::vector<std::shared_ptr<DMatrix>> cache)
      : need_configuration_{true} {
    monitor_.Init("Learner");
    for (std::shared_ptr<DMatrix> const& d : cache) {
      if (d) {
        prediction_container_.Cache(d, DeviceOrd::CPU());
      }
    }
  }

  // Configuration before data is known.
  void Configure() override {
    // Varient of double checked lock
    if (!this->need_configuration_) {
      return;
    }
    std::lock_guard<std::mutex> guard(config_lock_);
    if (!this->need_configuration_) {
      return;
    }

    monitor_.Start("Configure");
    auto old_tparam = tparam_;
    Args args = {cfg_.cbegin(), cfg_.cend()};

    tparam_.UpdateAllowUnknown(args);
    mparam_.UpdateAllowUnknown(args);

    auto initialized = ctx_.GetInitialised();
    auto old_seed = ctx_.seed;
    ctx_.UpdateAllowUnknown(args);

    ConsoleLogger::Configure(args);

    // set seed only before the model is initialized
    if (!initialized || ctx_.seed != old_seed) {
      common::GlobalRandom().seed(ctx_.seed);
    }

    // must precede configure gbm since num_features is required for gbm
    this->ConfigureNumFeatures();
    args = {cfg_.cbegin(), cfg_.cend()};  // renew
    this->ConfigureObjective(old_tparam, &args);

    learner_model_param_.task = obj_->Task();  // required by gbm configuration.
    this->ConfigureGBM(old_tparam, args);

    this->InitModelUserParam(this->tparam_, this->prediction_container_.Container());

    this->ConfigureMetrics(args);

    this->need_configuration_ = false;
    if (ctx_.validate_parameters) {
      this->ValidateParameters();
    }

    cfg_.clear();
    monitor_.Stop("Configure");
  }

  void LoadConfig(Json const& in) override {
    // If configuration is loaded, ensure that the model came from the same version
    CHECK(IsA<Object>(in));
    auto origin_version = Version::Load(in);
    if (std::get<0>(Version::kInvalid) == std::get<0>(origin_version)) {
      LOG(WARNING) << "Invalid version string in config";
    }

    if (!Version::Same(origin_version)) {
      error::WarnOldSerialization();
      return;  // skip configuration if version is not matched
    }

    auto const& learner_parameters = get<Object>(in["learner"]);
    FromJson(learner_parameters.at("learner_train_param"), &tparam_);

    auto const& gradient_booster = learner_parameters.at("gradient_booster");

    auto const& objective_fn = learner_parameters.at("objective");
    if (!obj_) {
      CHECK_EQ(get<String const>(objective_fn["name"]), tparam_.objective);
      obj_.reset(ObjFunction::Create(tparam_.objective, &ctx_));
    }
    obj_->LoadConfig(objective_fn);
    learner_model_param_.task = obj_->Task();

    tparam_.booster = get<String>(gradient_booster["name"]);
    if (!gbm_) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &ctx_, &learner_model_param_));
    }
    gbm_->LoadConfig(gradient_booster);

    auto const& j_metrics = learner_parameters.at("metrics");
    auto n_metrics = get<Array const>(j_metrics).size();
    metric_names_.resize(n_metrics);
    metrics_.resize(n_metrics);
    for (size_t i = 0; i < n_metrics; ++i) {
      auto old_serialization = IsA<String>(j_metrics[i]);
      if (old_serialization) {
        error::WarnOldSerialization();
        metric_names_[i] = get<String>(j_metrics[i]);
      } else {
        metric_names_[i] = get<String>(j_metrics[i]["name"]);
      }
      metrics_[i] = std::unique_ptr<Metric>(Metric::Create(metric_names_[i], &ctx_));
      if (!old_serialization) {
        metrics_[i]->LoadConfig(j_metrics[i]);
      }
    }

    FromJson(learner_parameters.at("generic_param"), &ctx_);

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

    std::vector<Json> metrics(metrics_.size());
    for (size_t i = 0; i < metrics_.size(); ++i) {
      metrics[i] = Object{};
      metrics_[i]->SaveConfig(&metrics[i]);
    }
    learner_parameters["metrics"] = Array(std::move(metrics));

    learner_parameters["generic_param"] = ToJson(ctx_);
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
  [[nodiscard]] CatContainer const* Cats() const override {
    this->CheckModelInitialized();
    return this->gbm_->Cats();
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

  Context const* Ctx() const override { return &ctx_; }

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

      for (auto const& kv : obj) {
        if (is_parameter(kv.first)) {
          auto parameter = get<Object const>(kv.second);
          std::transform(
              parameter.begin(), parameter.end(), std::back_inserter(keys),
              [](std::pair<std::string const&, Json const&> const& kv) { return kv.first; });
        } else if (IsA<Object>(kv.second)) {
          stack.push(kv.second);
        } else if (IsA<Array>(kv.second)) {
          auto const& array = get<Array const>(kv.second);
          for (auto const& v : array) {
            if (IsA<Object>(v) || IsA<Array>(v)) {
              stack.push(v);
            }
          }
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
      for (auto const& matrix : prediction_container_.Container()) {
        CHECK(matrix.first.ptr);
        CHECK(!matrix.second.ref.expired());
        const uint64_t num_col = matrix.first.ptr->Info().num_col_;
        error::MaxFeatureSize(num_col);
        num_feature = std::max(num_feature, static_cast<uint32_t>(num_col));
      }

      auto rc =
          collective::Allreduce(&ctx_, linalg::MakeVec(&num_feature, 1), collective::Op::kMax);
      collective::SafeColl(rc);
      if (num_feature > mparam_.num_feature) {
        mparam_.num_feature = num_feature;
      }
    }
    CHECK_NE(mparam_.num_feature, 0)
        << "0 feature is supplied.  Are you using raw Booster interface?";
  }

  void ConfigureGBM(LearnerTrainParam const& old, Args const& args) {
    if (gbm_ == nullptr || old.booster != tparam_.booster) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &ctx_,
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
      obj_.reset(ObjFunction::Create(tparam_.objective, &ctx_));
    }

    bool has_nc {cfg_.find("num_class") != cfg_.cend()};
    // Inject num_class into configuration.
    // FIXME(jiamingy): Remove the duplicated parameter in softmax
    cfg_["num_class"] = std::to_string(mparam_.num_class);
    auto& args = *p_args;
    args = {cfg_.cbegin(), cfg_.cend()};  // renew
    obj_->Configure(args);
    if (!has_nc) {
      cfg_.erase("num_class");
    }
  }

  void ConfigureMetrics(Args const& args) {
    for (auto const& name : metric_names_) {
      auto DupCheck = [&name](std::unique_ptr<Metric> const& m) { return m->Name() != name; };
      if (std::all_of(metrics_.begin(), metrics_.end(), DupCheck)) {
        metrics_.emplace_back(std::unique_ptr<Metric>(Metric::Create(name, &ctx_)));
      }
    }

    for (auto& p_metric : metrics_) {
      p_metric->Configure(args);
    }
  }

  /**
   * Get number of targets from objective function.
   */
  void ConfigureTargets() {
    CHECK(this->obj_);
    auto const& cache = this->prediction_container_.Container();
    bst_target_t n_targets = 1;
    for (auto const& d : cache) {
      if (n_targets == 1) {
        n_targets = this->obj_->Targets(d.first.ptr->Info());
      } else {
        auto t = this->obj_->Targets(d.first.ptr->Info());
        CHECK(n_targets == t || 1 == t) << "Inconsistent labels.";
      }
    }

    if (mparam_.num_target > 1) {
      CHECK(n_targets == 1 || n_targets == mparam_.num_target)
          << "Inconsistent configuration of the `num_target`.  Configuration result from input "
          << "data:" << n_targets << ", configuration from parameters:" << mparam_.num_target;
    } else {
      mparam_.num_target = n_targets;
    }
  }

  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) {
    base_score->SetDevice(this->Ctx()->Device());
    base_score->Reshape(this->mparam_.OutputLength());
    collective::ApplyWithLabels(this->Ctx(), info, base_score->Data(),
                                [&] { UsePtr(obj_)->InitEstimation(info, base_score); });
  }
};

std::string const LearnerConfiguration::kEvalMetric {"eval_metric"};  // NOLINT

class LearnerIO : public LearnerConfiguration {
 protected:
  void ClearCaches() { this->prediction_container_ = PredictionContainer{}; }

 public:
  explicit LearnerIO(std::vector<std::shared_ptr<DMatrix>> cache) : LearnerConfiguration{cache} {}

  void LoadModel(Json const& in) override {
    CHECK(IsA<Object>(in));
    auto version = Version::Load(in);
    if (std::get<0>(version) == 1 && std::get<1>(version) < 6) {
      LOG(WARNING)
          << "Found JSON model saved before XGBoost 1.6, please save the model using current "
             "version again. The support for old JSON model will be discontinued in XGBoost 3.2";
    }

    auto const& learner = get<Object>(in["learner"]);
    mparam_.FromJson(learner.at("learner_model_param"));

    auto const& objective_fn = learner.at("objective");

    std::string name = get<String>(objective_fn["name"]);
    tparam_.UpdateAllowUnknown(Args{{"objective", name}});
    obj_.reset(ObjFunction::Create(name, &ctx_));
    obj_->LoadConfig(objective_fn);

    auto const& gradient_booster = learner.at("gradient_booster");
    name = get<String>(gradient_booster["name"]);
    tparam_.UpdateAllowUnknown(Args{{"booster", name}});
    gbm_.reset(
        GradientBooster::Create(tparam_.booster, &ctx_, &learner_model_param_));
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
    this->ClearCaches();
  }

  void SaveModel(Json* p_out) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";
    this->CheckModelInitialized();

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

  void Save(dmlc::Stream* fo) const override {
    this->CheckModelInitialized();

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
    StringView msg = "Invalid serialization file.";
    CHECK_EQ(header[0], '{') << msg;

    auto buffer = common::ReadAll(fi, &fp);
    Json memory_snapshot;
    CHECK(std::isalpha(header[1])) << msg;
    if (header[1] == '"') {
      memory_snapshot = Json::Load(StringView{buffer});
      error::WarnOldSerialization();
    } else if (std::isalpha(header[1])) {
      memory_snapshot = Json::Load(StringView{buffer}, std::ios::binary);
    } else {
      LOG(FATAL) << "Invalid serialization file.";
    }

    this->LoadModel(memory_snapshot["Model"]);
    this->LoadConfig(memory_snapshot["Config"]);
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

  std::vector<std::string> DumpModel(const FeatureMap& fmap, bool with_stats,
                                     std::string format) override {
    this->Configure();
    this->CheckModelInitialized();

    return gbm_->DumpModel(fmap, with_stats, format);
  }

  Learner* Slice(bst_layer_t begin, bst_layer_t end, bst_layer_t step,
                 bool* out_of_bound) override {
    this->Configure();
    this->CheckModelInitialized();

    CHECK_NE(this->learner_model_param_.num_feature, 0);
    CHECK_GE(begin, 0);
    auto* out_impl = new LearnerImpl({});
    out_impl->learner_model_param_.Copy(this->learner_model_param_);
    out_impl->ctx_ = this->ctx_;
    auto gbm = std::unique_ptr<GradientBooster>(GradientBooster::Create(
        this->tparam_.booster, &out_impl->ctx_, &out_impl->learner_model_param_));
    this->gbm_->Slice(begin, end, step, gbm.get(), out_of_bound);
    out_impl->gbm_ = std::move(gbm);

    Json config{Object()};
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

  void Reset() override {
    this->Configure();
    this->CheckModelInitialized();
    // Global data
    auto local_map = LearnerAPIThreadLocalStore::Get();
    if (local_map->find(this) != local_map->cend()) {
      local_map->erase(this);
    }

    // Model
    std::string buf;
    common::MemoryBufferStream fo(&buf);
    this->Save(&fo);

    common::MemoryFixSizeBuffer fs(buf.data(), buf.size());
    this->Load(&fs);

    // Learner self cache. Prediction is cleared in the load method
    CHECK(this->prediction_container_.Container().empty());
    this->gpair_ = decltype(this->gpair_){};
  }

  void UpdateOneIter(int iter, std::shared_ptr<DMatrix> train) override {
    monitor_.Start("UpdateOneIter");
    TrainingObserver::Instance().Update(iter);
    this->Configure();
    this->FitIntercept(this->tparam_, train.get());

    if (ctx_.seed_per_iteration) {
      common::GlobalRandom().seed(ctx_.seed * kRandSeedMagic + iter);
    }

    this->ValidateDMatrix(train.get(), true);

    auto predt = prediction_container_.Cache(train, ctx_.Device());

    monitor_.Start("PredictRaw");
    this->PredictRaw(train.get(), predt.get(), true, 0, 0);
    TrainingObserver::Instance().Observe(predt->predictions, "Predictions");
    monitor_.Stop("PredictRaw");

    monitor_.Start("GetGradient");
    GetGradient(predt->predictions, train->Info(), iter, &gpair_);
    monitor_.Stop("GetGradient");
    TrainingObserver::Instance().Observe(*gpair_.Data(), "Gradients");

    gbm_->DoBoost(train.get(), &gpair_, predt.get(), obj_.get());
    monitor_.Stop("UpdateOneIter");
  }

  void BoostOneIter(int iter, std::shared_ptr<DMatrix> train,
                    linalg::Matrix<GradientPair>* in_gpair) override {
    monitor_.Start("BoostOneIter");
    this->Configure();

    if (ctx_.seed_per_iteration) {
      common::GlobalRandom().seed(ctx_.seed * kRandSeedMagic + iter);
    }

    this->ValidateDMatrix(train.get(), true);

    CHECK_EQ(this->learner_model_param_.OutputLength(), in_gpair->Shape(1))
        << "The number of columns in gradient should be equal to the number of targets/classes in "
           "the model.";
    auto predt = prediction_container_.Cache(train, ctx_.Device());
    gbm_->DoBoost(train.get(), in_gpair, predt.get(), obj_.get());
    monitor_.Stop("BoostOneIter");
  }

  std::string EvalOneIter(int iter,
                          const std::vector<std::shared_ptr<DMatrix>>& data_sets,
                          const std::vector<std::string>& data_names) override {
    monitor_.Start("EvalOneIter");
    this->Configure();
    this->CheckModelInitialized();

    std::ostringstream os;
    os.precision(std::numeric_limits<double>::max_digits10);
    os << '[' << iter << ']' << std::setiosflags(std::ios::fixed);
    if (metrics_.empty() && !tparam_.disable_default_eval_metric) {
      metrics_.emplace_back(Metric::Create(obj_->DefaultEvalMetric(), &ctx_));
      auto config = obj_->DefaultMetricConfig();
      if (!IsA<Null>(config)) {
        metrics_.back()->LoadConfig(config);
      }
      metrics_.back()->Configure({cfg_.begin(), cfg_.end()});
    }

    for (size_t i = 0; i < data_sets.size(); ++i) {
      std::shared_ptr<DMatrix> m = data_sets[i];
      auto predt = prediction_container_.Cache(m, ctx_.Device());
      this->ValidateDMatrix(m.get(), false);
      this->PredictRaw(m.get(), predt.get(), false, 0, 0);

      auto &out = output_predictions_.Cache(m, ctx_.Device())->predictions;
      out.Resize(predt->predictions.Size());
      out.Copy(predt->predictions);

      obj_->EvalTransform(&out);
      for (auto& ev : metrics_) {
        os << '\t' << data_names[i] << '-' << ev->Name() << ':' << ev->Evaluate(out, m);
      }
    }

    monitor_.Stop("EvalOneIter");
    return os.str();
  }

  void Predict(std::shared_ptr<DMatrix> data, bool output_margin,
               HostDeviceVector<float>* out_preds, bst_layer_t layer_begin, bst_layer_t layer_end,
               bool training, bool pred_leaf, bool pred_contribs, bool approx_contribs,
               bool pred_interactions) override {
    int multiple_predictions = static_cast<int>(pred_leaf) +
                               static_cast<int>(pred_interactions) +
                               static_cast<int>(pred_contribs);
    this->Configure();
    if (training) {
      this->FitIntercept(this->tparam_, nullptr);
    }
    this->CheckModelInitialized();

    CHECK_LE(multiple_predictions, 1) << "Perform one kind of prediction at a time.";
    if (pred_contribs) {
      gbm_->PredictContribution(data.get(), out_preds, layer_begin, layer_end, approx_contribs);
    } else if (pred_interactions) {
      gbm_->PredictInteractionContributions(data.get(), out_preds, layer_begin, layer_end,
                                            approx_contribs);
    } else if (pred_leaf) {
      gbm_->PredictLeaf(data.get(), out_preds, layer_begin, layer_end);
    } else {
      auto predt = prediction_container_.Cache(data, ctx_.Device());
      this->PredictRaw(data.get(), predt.get(), training, layer_begin, layer_end);
      // Copy the prediction cache to output prediction. out_preds comes from C API
      out_preds->SetDevice(ctx_.Device());
      out_preds->Resize(predt->predictions.Size());
      out_preds->Copy(predt->predictions);
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
    this->CheckModelInitialized();
    return this->learner_model_param_.num_output_group;
  }

  XGBAPIThreadLocalEntry& GetThreadLocal() const override {
    return (*LearnerAPIThreadLocalStore::Get())[this];
  }

  void InplacePredict(std::shared_ptr<DMatrix> p_m, PredictionType type, float missing,
                      HostDeviceVector<float>** out_preds, bst_layer_t iteration_begin,
                      bst_layer_t iteration_end) override {
    this->Configure();
    this->CheckModelInitialized();

    auto& out_predictions = this->GetThreadLocal().prediction_entry;
    out_predictions.Reset();

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
    this->CheckModelInitialized();

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
    this->CheckModelInitialized();
    this->ValidateDMatrix(data, false);
    gbm_->PredictBatch(data, out_preds, training, layer_begin, layer_end);
  }

  void ValidateDMatrix(DMatrix* p_fmat, bool is_training) const {
    MetaInfo const& info = p_fmat->Info();
    info.Validate(ctx_.Device());

    if (is_training) {
      CHECK_EQ(learner_model_param_.num_feature, p_fmat->Info().num_col_)
          << "Number of columns does not match number of features in "
             "booster.";
    } else {
      CHECK_GE(learner_model_param_.num_feature, p_fmat->Info().num_col_)
          << "Number of columns does not match number of features in "
             "booster.";
    }

    if (p_fmat->Info().num_row_ == 0) {
      error::WarnEmptyDataset();
    }
  }

 private:
  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) {
    out_gpair->Reshape(info.num_row_, this->learner_model_param_.OutputLength());
    collective::ApplyWithLabels(&ctx_, info, out_gpair->Data(),
                                [&] { obj_->GetGradient(preds, info, iter, out_gpair); });
  }

  /*! \brief random number transformation seed. */
  static int32_t constexpr kRandSeedMagic = 127;
  // gradient pairs
  linalg::Matrix<GradientPair> gpair_;
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
