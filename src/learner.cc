/**
 * Copyright 2014-2026, XGBoost Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include "xgboost/learner.h"

#include <dmlc/io.h>            // for Stream
#include <dmlc/parameter.h>     // for FieldEntry, DMLC_DECLARE_FIELD, Parameter, DMLC...
#include <dmlc/thread_local.h>  // for ThreadLocalStore

#include <algorithm>      // for equal, max, transform, sort, find_if, all_of
#include <atomic>         // for atomic
#include <cctype>         // for isalpha, isspace
#include <cmath>          // for isnan, isinf
#include <cstdint>        // for int32_t, uint32_t, int64_t, uint64_t
#include <cstdlib>        // for atoi
#include <cstring>        // for memcpy, size_t, memset
#include <iomanip>        // for operator<<, setiosflags
#include <iterator>       // for back_insert_iterator, distance, back_inserter
#include <limits>         // for numeric_limits
#include <memory>         // for allocator, unique_ptr, shared_ptr, operator==
#include <mutex>          // for mutex, lock_guard
#include <sstream>        // for operator<<, basic_ostream, basic_ostream::opera...
#include <string>         // for basic_string, char_traits, operator<, string
#include <system_error>   // for errc
#include <unordered_map>  // for operator!=, unordered_map
#include <utility>        // for pair, as_const, move, swap
#include <vector>         // for vector

#include "collective/allreduce.h"         // for Allreduce, SafeColl
#include "collective/broadcast.h"         // for Broadcast
#include "collective/communicator-inl.h"  // for GetRank, IsDistributed
#include "common/api_entry.h"             // for XGBAPIThreadLocalEntry
#include "common/charconv.h"              // for to_chars, to_chars_result, NumericLimits, from_...
#include "common/error_msg.h"             // for MaxFeatureSize, WarnOldSerialization, ...
#include "common/io.h"                    // for PeekableInStream, ReadAll, FixedSizeStream, Mem...
#include "common/observer.h"              // for TrainingObserver
#include "common/param_array.h"           // for ParamArray
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
#include "xgboost/string_view.h"          // for operator<<, StringView
#include "xgboost/task.h"                 // for ObjInfo

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

LearnerModelState::LearnerModelState(Context const* ctx, bst_feature_t n_features,
                                     std::int32_t n_classes, bst_target_t n_targets,
                                     bool boost_from_average, std::vector<float> base_score_value,
                                     linalg::Vector<float> base_score, ObjInfo task,
                                     MultiStrategy multi_strategy)
    : num_feature{n_features},
      num_class{n_classes},
      num_target{n_targets},
      boost_from_average{boost_from_average},
      num_output_group{std::max(
          {n_targets, static_cast<bst_target_t>(n_classes), static_cast<bst_target_t>(1)})},
      task{task},
      multi_strategy{multi_strategy} {
  if (num_class > 1 && num_target > 1) {
    LOG(FATAL) << "multi-target-multi-class is not yet supported. Output classes:" << num_class
               << ", output targets:" << num_target;
  }
  this->SetBaseScore(ctx, std::move(base_score_value), std::move(base_score));
}

void LearnerModelState::SetBaseScore(Context const* ctx, std::vector<float> value,
                                     linalg::Vector<float> base_score) {
  base_score_value_ = std::move(value);
  std::swap(base_score_, base_score);
  this->ConfigureDevice(ctx);
}

void LearnerModelState::ConfigureDevice(Context const* ctx) {
  if (base_score_.Device() != ctx->Device()) {
    auto const& h_base_score = std::as_const(base_score_).Data()->ConstHostVector();
    linalg::Vector<float> base_score{
        h_base_score.cbegin(), h_base_score.cend(), {h_base_score.size()}, ctx->Device()};
    std::swap(base_score_, base_score);
  }
  // Make sure read access everywhere for thread-safe prediction.
  std::as_const(base_score_).HostView();
  if (!ctx->IsCPU()) {
    std::as_const(base_score_).View(ctx->Device());
  }
  CHECK(std::as_const(base_score_).Data()->HostCanRead());
}

linalg::VectorView<float const> LearnerModelState::BaseScore(DeviceOrd device) const {
  // multi-class is not yet supported.
  CHECK_GE(base_score_.Size(), 1) << ModelNotFitted();
  if (device.IsCPU()) {
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

linalg::VectorView<float const> LearnerModelState::BaseScore(Context const* ctx) const {
  return this->BaseScore(ctx->Device());
}

void LearnerModelState::Copy(LearnerModelState const& that) {
  base_score_.Reshape(that.base_score_.Shape());
  base_score_.Data()->SetDevice(that.base_score_.Device());
  base_score_.Data()->Copy(*that.base_score_.Data());
  std::as_const(base_score_).HostView();
  if (!that.base_score_.Device().IsCPU()) {
    std::as_const(base_score_).View(that.base_score_.Device());
  }
  CHECK_EQ(base_score_.Data()->DeviceCanRead(), that.base_score_.Data()->DeviceCanRead());
  CHECK(base_score_.Data()->HostCanRead());

  base_score_value_ = that.base_score_value_;
  num_feature = that.num_feature;
  num_class = that.num_class;
  num_target = that.num_target;
  boost_from_average = that.boost_from_average;
  num_output_group = that.num_output_group;
  task = that.task;
  multi_strategy = that.multi_strategy;
}

struct LearnerTrainParam : public XGBoostParameter<LearnerTrainParam> {
  // Parameters consumed when the model state is initialized.
  common::ParamArray<float> base_score{"base_score"};
  std::int32_t num_class{0};
  bst_target_t num_target{1};
  std::int32_t boost_from_average{true};

  // flag to disable default metric
  bool disable_default_eval_metric{false};
  std::string booster;
  std::string objective;
  // This is a training parameter and is not saved (nor loaded) in the model.
  MultiStrategy multi_strategy{MultiStrategy::kOneOutputPerTree};

  template <typename Container>
  Args UpdateAllowUnknown(Container const& kwargs) {
    auto has_key = [&kwargs](char const* key) {
      return std::find_if(kwargs.cbegin(), kwargs.cend(),
                          [key](auto const& kv) { return kv.first == key; }) != kwargs.cend();
    };
    auto unknown = XGBoostParameter<LearnerTrainParam>::UpdateAllowUnknown(kwargs);
    // An explicit boost_from_average takes precedence when both parameters are supplied.
    if (has_key("base_score") && !has_key("boost_from_average")) {
      this->boost_from_average = false;
    }
    return unknown;
  }

  [[nodiscard]] bst_target_t OutputLength() const noexcept {
    return std::max({this->num_target, static_cast<bst_target_t>(this->num_class),
                     static_cast<bst_target_t>(1)});
  }

  DMLC_DECLARE_PARAMETER(LearnerTrainParam) {
    DMLC_DECLARE_FIELD(base_score)
        .describe("Global bias of the model.")
        .set_default(common::ParamArray<float>{"base_score"});
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

DMLC_REGISTER_PARAMETER(LearnerTrainParam);

using LearnerAPIThreadLocalStore =
    dmlc::ThreadLocalStore<std::map<Learner const*, XGBAPIThreadLocalEntry>>;

namespace {
std::string CanonicalizeBoosterName(std::string booster) {
  if (booster == "dart") {
    static std::once_flag flag;
    std::call_once(flag, [] {
      LOG(WARNING) << "`booster=dart` is deprecated. Use the tree booster directly with "
                      "dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.";
    });
    return "gbtree";
  }
  return booster;
}

/**
 * @brief Handler for learner model inputs and state initialization.
 */
class LearnerModelStateContainer : public Learner {
  using CacheT = std::vector<std::weak_ptr<DMatrix>>;

 protected:
  enum class InterceptInitialization { kEstimateIntercept, kUseDefaultIntercept };
  /** @brief User-configurable inputs for constructing model state. */
  LearnerTrainParam tparam_;
  /** @brief Authoritative model state shared with the gradient booster. */
  LearnerModelState model_state_;

 private:
  void InitEstimation(MetaInfo const& info, bst_target_t output_length,
                      linalg::Vector<float>* base_score) {
    base_score->SetDevice(this->Ctx()->Device());
    base_score->Reshape(output_length);
    UsePtr(obj_)->InitEstimation(info, base_score);
  }

  static void HandleOldFormat(std::vector<float>* base_score, bst_target_t output_length) {
    if (base_score->size() == 1 && output_length > 1) {
      base_score->resize(output_length, base_score->front());
    }
  }

  [[nodiscard]] Json ModelStateToJson() const {
    auto n_features = model_state_.Initialized() ? model_state_.num_feature : 0;
    auto n_classes = model_state_.Initialized() ? model_state_.num_class : tparam_.num_class;
    auto n_targets = model_state_.Initialized() ? model_state_.num_target : tparam_.num_target;
    auto boost_from_average =
        model_state_.Initialized() ? model_state_.boost_from_average : tparam_.boost_from_average;
    std::vector<float> base_score;
    if (model_state_.Initialized()) {
      base_score = model_state_.BaseScoreValue();
    } else {
      base_score.assign(tparam_.base_score.cbegin(), tparam_.base_score.cend());
    }

    common::ParamArray<float> value{"base_score"};
    value = base_score;
    std::stringstream ss;
    ss << value;

    Json out{Object{}};
    out["base_score"] = ss.str();
    out["num_feature"] = std::to_string(n_features);
    out["num_class"] = std::to_string(n_classes);
    out["num_target"] = std::to_string(n_targets);
    out["boost_from_average"] = std::to_string(static_cast<std::int32_t>(boost_from_average));
    return out;
  }

  void ValidateModelState() const {
    CHECK(model_state_.Initialized()) << ModelNotFitted();
    auto const& base_score = model_state_.BaseScoreValue();
    CHECK_GE(base_score.size(), 1);
    auto n_classes = static_cast<std::size_t>(model_state_.num_class);
    auto n_targets = static_cast<std::size_t>(model_state_.num_target);
    if (!(base_score.size() == n_classes || base_score.size() == n_targets)) {
      error::InvalidIntercept(n_classes, n_targets, base_score.size());
    }
    CHECK(std::none_of(base_score.cbegin(), base_score.cend(),
                       [](float v) { return std::isnan(v) || std::isinf(v); }));

    if (!collective::IsDistributed()) {
      return;
    }
    std::vector<char> data;
    Json::Dump(this->ModelStateToJson(), &data, std::ios::binary);
    std::vector<char> sync{data};
    auto rc = collective::Broadcast(&ctx_, linalg::MakeVec(sync.data(), sync.size()), 0);
    collective::SafeColl(rc);
    CHECK(std::equal(data.cbegin(), data.cend(), sync.cbegin()))
        << "Different model state across workers:\n\t"
        << Json::Load(StringView{data.data(), data.size()}, std::ios::binary) << "\nvs.\n\t"
        << Json::Load(StringView{sync.data(), sync.size()}, std::ios::binary);
  }

  void InitModelState(bst_feature_t n_features, std::int32_t n_classes, bst_target_t n_targets,
                      bool boost_from_average, std::vector<float> base_score_value) {
    auto output_length =
        std::max({n_targets, static_cast<bst_target_t>(n_classes), static_cast<bst_target_t>(1)});
    HandleOldFormat(&base_score_value, output_length);
    linalg::Vector<float> base_score{base_score_value.cbegin(),
                                     base_score_value.cend(),
                                     {base_score_value.size()},
                                     this->ctx_.Device()};
    UsePtr(this->obj_)->ProbToMargin(&base_score);
    model_state_ = LearnerModelState{Ctx(),
                                     n_features,
                                     n_classes,
                                     n_targets,
                                     boost_from_average,
                                     std::move(base_score_value),
                                     std::move(base_score),
                                     UsePtr(this->obj_)->Task(),
                                     tparam_.multi_strategy};
    this->ValidateModelState();
  }

  [[nodiscard]] bst_feature_t InitNumFeatures(DMatrix const& train) const {
    auto n_features = train.Info().num_col_;
    error::MaxFeatureSize(n_features);
    CHECK_NE(n_features, 0) << "0 feature is supplied. Are you using the raw Booster interface?";
    return static_cast<bst_feature_t>(n_features);
  }

  [[nodiscard]] bst_target_t InitNumTargets(DMatrix const& train, CacheT const& cache) const {
    CHECK(this->obj_);
    auto n_targets = this->obj_->Targets(train.Info());
    for (auto const& weak : cache) {
      auto d = weak.lock();
      if (!d) {
        continue;
      }
      auto t = this->obj_->Targets(d->Info());
      CHECK(n_targets == t || 1 == t) << "Inconsistent labels.";
    }

    if (model_state_.Initialized()) {
      CHECK(n_targets == 1 || n_targets == model_state_.num_target)
          << "Inconsistent number of targets between data and model.";
      return model_state_.num_target;
    }
    if (tparam_.num_target > 1) {
      CHECK(n_targets == 1 || n_targets == tparam_.num_target)
          << "Inconsistent configuration of the `num_target`.  Configuration result from input "
          << "data:" << n_targets << ", configuration from parameters:" << tparam_.num_target;
      return tparam_.num_target;
    }
    return n_targets;
  }

 protected:
  [[nodiscard]] Json SaveModelState() const { return this->ModelStateToJson(); }

  void CheckModelInitialized() const { CHECK(model_state_.Initialized()) << ModelNotFitted(); }

  void ConfigureModelState(LearnerTrainParam const& old_tparam, Args const& args) {
    if (model_state_.NeedsInitialization()) {
      return;
    }
    model_state_.ConfigureDevice(Ctx());
    auto has = [&args](char const* key) {
      return std::any_of(args.cbegin(), args.cend(),
                         [key](auto const& kv) { return kv.first == key; });
    };
    bool model_input_changed =
        has("base_score") || has("num_class") || has("num_target") || has("boost_from_average");
    bool structure_changed = old_tparam.objective != tparam_.objective ||
                             old_tparam.multi_strategy != tparam_.multi_strategy;
    if (!model_input_changed && !structure_changed) {
      return;
    }

    auto base_score = model_state_.BaseScoreValue();
    auto n_classes = model_state_.num_class;
    auto n_targets = model_state_.num_target;
    auto boost_from_average = model_state_.boost_from_average;
    if (has("base_score")) {
      base_score.assign(tparam_.base_score.cbegin(), tparam_.base_score.cend());
    }
    if (has("num_class")) {
      n_classes = tparam_.num_class;
    }
    if (has("num_target")) {
      n_targets = tparam_.num_target;
    }
    if (has("boost_from_average") || has("base_score")) {
      boost_from_average = tparam_.boost_from_average;
    }
    this->InitModelState(model_state_.num_feature, n_classes, n_targets, boost_from_average,
                         std::move(base_score));
  }

  void InitializeModel(DMatrix const& train, CacheT const& cache, InterceptInitialization mode) {
    auto n_features = this->InitNumFeatures(train);
    auto n_targets = this->InitNumTargets(train, cache);
    if (model_state_.Initialized()) {
      return;
    }

    auto output_length = std::max(
        {n_targets, static_cast<bst_target_t>(tparam_.num_class), static_cast<bst_target_t>(1)});
    std::vector<float> base_score;
    if (!tparam_.boost_from_average) {
      base_score.assign(tparam_.base_score.cbegin(), tparam_.base_score.cend());
    } else if (mode == InterceptInitialization::kEstimateIntercept) {
      auto const& info = train.Info();
      info.Validate(Ctx()->Device());
      linalg::Vector<float> estimated;
      this->InitEstimation(info, output_length, &estimated);
      base_score = estimated.Data()->ConstHostVector();
    } else {
      base_score.resize(output_length, ObjFunction::DefaultBaseScore());
    }
    this->InitModelState(n_features, tparam_.num_class, n_targets, tparam_.boost_from_average,
                         std::move(base_score));
  }

  void LoadModelState(Json const& in) {
    auto const& values = get<Object const>(in);
    common::ParamArray<float> base_score_param{"base_score"};
    std::istringstream is{get<String const>(values.at("base_score"))};
    is >> base_score_param;
    CHECK(!is.fail()) << "Invalid base_score in model.";
    std::vector<float> base_score{base_score_param.cbegin(), base_score_param.cend()};

    auto n_features_value = std::stoull(get<String const>(values.at("num_feature")));
    error::MaxFeatureSize(n_features_value);
    auto n_features = static_cast<bst_feature_t>(n_features_value);
    auto n_classes = std::stoi(get<String const>(values.at("num_class")));
    CHECK_GE(n_classes, 0);
    bst_target_t n_targets{1};
    if (auto it = values.find("num_target"); it != values.cend()) {
      auto value = std::stoull(get<String const>(it->second));
      CHECK_LE(value, std::numeric_limits<bst_target_t>::max());
      n_targets = static_cast<bst_target_t>(value);
      CHECK_GE(n_targets, 1);
    }
    bool boost_from_average{true};
    if (auto it = values.find("boost_from_average"); it != values.cend()) {
      boost_from_average = std::stoi(get<String const>(it->second));
    }

    tparam_.base_score = base_score;
    tparam_.num_class = n_classes;
    tparam_.num_target = n_targets;
    tparam_.boost_from_average = boost_from_average;
    if (n_features == 0) {
      model_state_ = LearnerModelState{};
      return;
    }
    this->InitModelState(n_features, n_classes, n_targets, boost_from_average,
                         std::move(base_score));
  }
};
}  // namespace

class LearnerConfiguration : public LearnerModelStateContainer {
 private:
  std::mutex config_lock_;

 protected:
  static std::string const kEvalMetric;  // NOLINT

 protected:
  std::atomic<bool> need_configuration_;
  // Stores information like best-iteration for early stopping.
  std::map<std::string, std::string> attributes_;
  // Name of each feature, usually set from DMatrix.
  std::vector<std::string> feature_names_;
  // Type of each feature, usually set from DMatrix.
  std::vector<std::string> feature_types_;

  common::Monitor monitor_;
  // Initial prediction.
  std::vector<std::weak_ptr<DMatrix>> cache_data_;

  std::vector<std::string> metric_names_;

 public:
  explicit LearnerConfiguration(std::vector<std::shared_ptr<DMatrix>> cache)
      : need_configuration_{true} {
    monitor_.Init("Learner");
    for (std::shared_ptr<DMatrix> const& d : cache) {
      if (d) {
        cache_data_.emplace_back(d);
      }
    }
  }

  void Configure(Args const& args = {}) override {
    bool has_args = !args.empty();
    if (has_args) {
      this->need_configuration_ = true;
    }

    if (!has_args && !this->need_configuration_) {
      return;
    }
    std::lock_guard<std::mutex> guard(config_lock_);
    if (!has_args && !this->need_configuration_) {
      return;
    }

    monitor_.Start("Configure");
    auto old_tparam = tparam_;
    std::map<std::string, std::string> config;
    std::set<std::string> used;

    for (auto const& [key, value] : args) {
      if (key == kEvalMetric) {
        used.insert(kEvalMetric);
        if (std::find(metric_names_.cbegin(), metric_names_.cend(), value) ==
            metric_names_.cend()) {
          metric_names_.emplace_back(value);
        }
      } else {
        config[key] = value;
      }
    }

    Args config_args{config.cbegin(), config.cend()};

    used.merge(UpdateAndGetUsedParameters(&tparam_, config_args));

    auto initialized = ctx_.GetInitialised();
    auto old_seed = ctx_.seed;
    used.merge(UpdateAndGetUsedParameters(&ctx_, config_args));

    used.merge(ConsoleLogger::Configure(config_args));

    // set seed only before the model is initialized
    if (!initialized || ctx_.seed != old_seed) {
      ctx_.Rng().seed(ctx_.seed);
    }

    used.merge(this->ConfigureObjective(old_tparam, &config, &config_args));

    model_state_.task = obj_->Task();  // required by gbm configuration.
    used.merge(this->ConfigureGBM(old_tparam, config_args));

    this->ConfigureModelState(old_tparam, args);
    used.merge(this->ConfigureMetrics(config_args));

    this->need_configuration_ = false;
    if (ctx_.validate_parameters) {
      this->ValidateParameters(args, used);
    }

    monitor_.Stop("Configure");
  }

 protected:
  void LoadConfigImpl(Json const& in) {
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
    model_state_.task = obj_->Task();

    tparam_.booster = CanonicalizeBoosterName(get<String>(gradient_booster["name"]));
    if (!gbm_) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &ctx_, &model_state_));
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

    ctx_.FromJson(learner_parameters.at("generic_param"));

    this->need_configuration_ = true;
  }

 public:
  void LoadConfig(Json const& in) override {
    this->LoadConfigImpl(in);
    this->Configure();
  }

  void SaveConfig(Json* p_out) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";
    Version::Save(p_out);
    Json& out{*p_out};
    // parameters
    out["learner"] = Object();
    auto& learner_parameters = out["learner"];

    learner_parameters["learner_train_param"] = ToJson(tparam_);
    learner_parameters["learner_model_param"] = this->SaveModelState();
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

    learner_parameters["generic_param"] = ctx_.ToJson();
  }

  uint32_t GetNumFeature() const override { return model_state_.num_feature; }

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
    if (it == attributes_.end()) {
      return false;
    }
    attributes_.erase(it);
    return true;
  }

  void SetFeatureNames(std::vector<std::string> const& fn) override { feature_names_ = fn; }

  void GetFeatureNames(std::vector<std::string>* fn) const override { *fn = feature_names_; }

  void SetFeatureTypes(std::vector<std::string> const& ft) override { this->feature_types_ = ft; }

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

  Context const* Ctx() const override { return &ctx_; }

 private:
  void ValidateParameters(Args const& args, std::set<std::string> const& used) {
    std::set<std::string> provided;
    for (auto const& kv : args) {
      if (std::any_of(kv.first.cbegin(), kv.first.cend(),
                      [](char ch) { return std::isspace(ch); })) {
        LOG(FATAL) << "Invalid parameter \"" << kv.first << "\" contains whitespace.";
      }
      provided.insert(kv.first);
    }

    std::vector<std::string> diff;
    std::set_difference(provided.begin(), provided.end(), used.begin(), used.end(),
                        std::back_inserter(diff));
    if (!diff.empty()) {
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

  std::set<std::string> ConfigureGBM(LearnerTrainParam const& old, Args const& args) {
    tparam_.booster = CanonicalizeBoosterName(tparam_.booster);
    if (tparam_.booster == "gblinear") {
      LOG(WARNING) << "`booster=gblinear` is deprecated and support will be removed in a future "
                      "release.";
    }
    auto old_booster = CanonicalizeBoosterName(old.booster);
    if (gbm_ == nullptr || old_booster != tparam_.booster) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &ctx_, &model_state_));
    }
    return gbm_->Configure(args);
  }

  std::set<std::string> ConfigureObjective(LearnerTrainParam const& old,
                                           std::map<std::string, std::string>* p_config,
                                           Args* p_args) {
    auto& config = *p_config;
    // Once binary IO is gone, NONE of these config is useful.
    if (config.find("num_class") != config.cend() && config.at("num_class") != "0" &&
        tparam_.objective != "multi:softprob") {
      config["num_output_group"] = config["num_class"];
      if (atoi(config["num_class"].c_str()) > 1 && config.count("objective") == 0) {
        tparam_.objective = "multi:softmax";
      }
    }

    if (obj_ == nullptr || tparam_.objective != old.objective) {
      obj_.reset(ObjFunction::Create(tparam_.objective, &ctx_));
    }

    bool has_nc{config.find("num_class") != config.cend()};
    // Inject num_class into configuration.
    // FIXME(jiamingy): Remove the duplicated parameter in softmax
    config["num_class"] = std::to_string(tparam_.num_class);
    auto& args = *p_args;
    args = {config.cbegin(), config.cend()};
    auto used = obj_->Configure(args);
    if (!has_nc) {
      config.erase("num_class");
    }
    return used;
  }

  std::set<std::string> ConfigureMetrics(Args const& args) {
    std::set<std::string> used;
    for (auto const& name : metric_names_) {
      auto DupCheck = [&name](std::unique_ptr<Metric> const& m) {
        return m->Name() != name;
      };
      if (std::all_of(metrics_.begin(), metrics_.end(), DupCheck)) {
        metrics_.emplace_back(std::unique_ptr<Metric>(Metric::Create(name, &ctx_)));
      }
    }

    for (auto& p_metric : metrics_) {
      used.merge(p_metric->Configure(args));
    }
    return used;
  }
};

std::string const LearnerConfiguration::kEvalMetric{"eval_metric"};  // NOLINT

class LearnerIO : public LearnerConfiguration {
 protected:
  void ClearCaches() { this->cache_data_.clear(); }

 public:
  explicit LearnerIO(std::vector<std::shared_ptr<DMatrix>> cache) : LearnerConfiguration{cache} {}

 protected:
  void LoadModelImpl(Json const& in) {
    CHECK(IsA<Object>(in));
    model_state_ = LearnerModelState{};
    auto version = Version::Load(in);
    if (std::get<0>(version) == 1 && std::get<1>(version) < 6) {
      LOG(WARNING)
          << "Found JSON model saved before XGBoost 1.6, please save the model using current "
             "version again. The support for old JSON model will be discontinued in XGBoost 3.2";
    }

    auto const& learner = get<Object>(in["learner"]);

    auto const& objective_fn = learner.at("objective");

    std::string name = get<String>(objective_fn["name"]);
    tparam_.UpdateAllowUnknown(Args{{"objective", name}});
    obj_.reset(ObjFunction::Create(name, &ctx_));
    obj_->LoadConfig(objective_fn);

    this->LoadModelState(learner.at("learner_model_param"));
    auto const& gradient_booster = learner.at("gradient_booster");
    name = get<String>(gradient_booster["name"]);
    tparam_.UpdateAllowUnknown(Args{{"booster", name}});
    tparam_.booster = CanonicalizeBoosterName(tparam_.booster);
    gbm_.reset(GradientBooster::Create(tparam_.booster, &ctx_, &model_state_));
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

 public:
  void LoadModel(Json const& in) override {
    this->LoadModelImpl(in);
    this->Configure();
  }

 private:
  void SaveModelUnchecked(Json* p_out) const {
    Version::Save(p_out);
    Json& out{*p_out};

    out["learner"] = Object();
    auto& learner = out["learner"];

    learner["learner_model_param"] = this->SaveModelState();
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

 public:
  void SaveModel(Json* p_out) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";
    this->CheckModelInitialized();
    this->SaveModelUnchecked(p_out);
  }

  void Save(dmlc::Stream* fo) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";
    Json memory_snapshot{Object()};
    memory_snapshot["Model"] = Object();
    auto& model = memory_snapshot["Model"];
    // A memory snapshot preserves pending parameters and metadata for an uninitialized
    // learner. Unlike a model file, it does not require a fitted model.
    this->SaveModelUnchecked(&model);
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

    this->LoadModelImpl(memory_snapshot["Model"]);
    this->LoadConfigImpl(memory_snapshot["Config"]);
    this->Configure();
  }
};

/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
class LearnerImpl : public LearnerIO {
 public:
  explicit LearnerImpl(std::vector<std::shared_ptr<DMatrix>> cache) : LearnerIO{cache} {}
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

    CHECK_NE(this->model_state_.num_feature, 0);
    CHECK_GE(begin, 0);
    auto* out_impl = new LearnerImpl({});
    out_impl->model_state_.Copy(this->model_state_);
    out_impl->tparam_ = this->tparam_;
    out_impl->ctx_ = this->ctx_;
    auto gbm = std::unique_ptr<GradientBooster>(
        GradientBooster::Create(this->tparam_.booster, &out_impl->ctx_, &out_impl->model_state_));
    this->gbm_->Slice(begin, end, step, gbm.get(), out_of_bound);
    out_impl->gbm_ = std::move(gbm);

    Json config{Object()};
    this->SaveConfig(&config);
    out_impl->attributes_ = this->attributes_;
    out_impl->SetFeatureNames(this->feature_names_);
    out_impl->SetFeatureTypes(this->feature_types_);
    out_impl->LoadConfig(config);
    out_impl->Configure();
    CHECK_EQ(out_impl->model_state_.num_feature, this->model_state_.num_feature);
    CHECK_NE(out_impl->model_state_.num_feature, 0);

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
    if (model_state_.NeedsInitialization()) {
      for (auto const& weak : cache_data_) {
        if (auto data = weak.lock()) {
          this->InitializeModel(*data, this->cache_data_,
                                InterceptInitialization::kEstimateIntercept);
          break;
        }
      }
    }
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
    CHECK(this->cache_data_.empty());
    this->gpair_ = decltype(this->gpair_){};
  }

  void UpdateOneIter(int iter, std::shared_ptr<DMatrix> train) override {
    monitor_.Start("UpdateOneIter");
    TrainingObserver::Instance().Update(iter);
    this->Configure();
    this->InitializeModel(*train, this->cache_data_, InterceptInitialization::kEstimateIntercept);

    if (ctx_.seed_per_iteration) {
      ctx_.Rng().seed(ctx_.seed * kRandSeedMagic + this->BoostedRounds());
    }

    this->ValidateDMatrix(train.get(), true);

    HostDeviceVector<float> predt;

    monitor_.Start("PredictRaw");
    this->PredictRaw(train, &predt, true, 0, 0);
    TrainingObserver::Instance().Observe(predt, "Predictions");
    monitor_.Stop("PredictRaw");

    monitor_.Start("GetGradient");
    GetGradient(predt, train->Info(), iter, &gpair_.gpair);
    monitor_.Stop("GetGradient");
    TrainingObserver::Instance().Observe(gpair_.Grad()->Data(), "Gradients");

    gbm_->DoBoost(train, &gpair_, obj_.get());
    monitor_.Stop("UpdateOneIter");
  }

  void BoostOneIter(std::int32_t, std::shared_ptr<DMatrix> train,
                    GradientContainer* in_gpair) override {
    this->monitor_.Start(__func__);
    this->Configure();
    this->InitializeModel(*train, this->cache_data_, InterceptInitialization::kUseDefaultIntercept);

    if (ctx_.seed_per_iteration) {
      ctx_.Rng().seed(ctx_.seed * kRandSeedMagic + this->BoostedRounds());
    }

    this->ValidateDMatrix(train.get(), true);
    if (in_gpair->HasValueGrad()) {
      CHECK_EQ(this->model_state_.OutputLength(), in_gpair->NumTargets())
          << "Value gradient should have the same number of targets as the overall model.";
    } else {
      CHECK_EQ(this->model_state_.OutputLength(), in_gpair->NumSplitTargets())
          << "The number of columns in gradient should be equal to the number of "
             "targets/classes in the model.";
    }
    this->gbm_->DoBoost(train, in_gpair, obj_.get());
    this->monitor_.Stop(__func__);
  }

  std::string EvalOneIter(int iter, const std::vector<std::shared_ptr<DMatrix>>& data_sets,
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
      metrics_.back()->Configure({});
    }

    for (size_t i = 0; i < data_sets.size(); ++i) {
      std::shared_ptr<DMatrix> m = data_sets[i];
      this->ValidateDMatrix(m.get(), false);
      HostDeviceVector<float> out;
      this->PredictRaw(m, &out, false, 0, 0);

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
               bool pred_interactions, bool strict_shape) override {
    int multiple_predictions = static_cast<int>(pred_leaf) + static_cast<int>(pred_interactions) +
                               static_cast<int>(pred_contribs);
    this->Configure();
    if (training) {
      this->InitializeModel(*data, this->cache_data_,
                            InterceptInitialization::kUseDefaultIntercept);
    }
    this->CheckModelInitialized();

    CHECK_LE(multiple_predictions, 1) << "Perform one kind of prediction at a time.";
    if (pred_contribs) {
      this->ValidateDMatrix(data.get(), false);
      gbm_->PredictContribution(data.get(), out_preds, layer_begin, layer_end, approx_contribs);
    } else if (pred_interactions) {
      this->ValidateDMatrix(data.get(), false);
      gbm_->PredictInteractionContributions(data.get(), out_preds, layer_begin, layer_end,
                                            approx_contribs);
    } else if (pred_leaf) {
      this->ValidateDMatrix(data.get(), false);
      gbm_->PredictLeaf(data.get(), out_preds, layer_begin, layer_end, strict_shape);
    } else {
      this->PredictRaw(data, out_preds, training, layer_begin, layer_end);
      if (!output_margin) {
        obj_->PredTransform(out_preds);
      }
    }
  }

  int32_t BoostedRounds() const override {
    if (!this->gbm_) {
      return 0;
    }  // haven't called train or LoadModel.
    CHECK(!this->need_configuration_);
    return this->gbm_->BoostedRounds();
  }

  uint32_t Groups() const override {
    CHECK(!this->need_configuration_);
    this->CheckModelInitialized();
    return this->model_state_.num_output_group;
  }

  XGBAPIThreadLocalEntry& GetThreadLocal() const override {
    return (*LearnerAPIThreadLocalStore::Get())[this];
  }

  void InplacePredict(std::shared_ptr<DMatrix> p_m, PredictionType type, float missing,
                      HostDeviceVector<float>** out_preds, bst_layer_t iteration_begin,
                      bst_layer_t iteration_end) override {
    this->Configure();
    this->CheckModelInitialized();

    auto& out_predictions = this->GetThreadLocal().predictions;
    out_predictions.Resize(0);

    this->gbm_->InplacePredict(p_m, missing, &out_predictions, iteration_begin, iteration_end);

    if (type == PredictionType::kValue) {
      obj_->PredTransform(&out_predictions);
    } else if (type == PredictionType::kMargin) {
      // do nothing
    } else {
      LOG(FATAL) << "Unsupported prediction type:" << static_cast<int>(type);
    }
    *out_preds = &out_predictions;
  }

  void CalcFeatureScore(std::string const& importance_type, common::Span<int32_t const> trees,
                        std::vector<bst_feature_t>* features, std::vector<float>* scores) override {
    this->Configure();
    this->CheckModelInitialized();

    gbm_->FeatureScore(importance_type, trees, features, scores);
  }

 protected:
  /*!
   * \brief get un-transformed prediction
   * \param data training data matrix
   * \param out_preds output vector that stores the prediction
   * \param layer_begin Beginning of the boosting iteration range.
   * \param layer_end End of the boosting iteration range. Zero uses all iterations.
   * \param training allow dropout when the DART booster is being used
   */
  void PredictRaw(std::shared_ptr<DMatrix> data, HostDeviceVector<float>* out_preds, bool training,
                  unsigned layer_begin, unsigned layer_end) const {
    CHECK(gbm_ != nullptr) << "Predict must happen after Load or configuration";
    this->CheckModelInitialized();
    this->ValidateDMatrix(data.get(), false);
    gbm_->PredictBatch(data, out_preds, training, layer_begin, layer_end);
  }

  void ValidateDMatrix(DMatrix* p_fmat, bool is_training) const {
    MetaInfo const& info = p_fmat->Info();
    info.Validate(ctx_.Device());

    if (is_training) {
      CHECK_EQ(model_state_.num_feature, p_fmat->Info().num_col_)
          << "Number of columns does not match number of features in "
             "booster.";
    } else {
      CHECK_GE(model_state_.num_feature, p_fmat->Info().num_col_)
          << "Number of columns does not match number of features in "
             "booster.";
    }

    if (p_fmat->Info().num_row_ == 0) {
      error::WarnEmptyDataset();
    }
    if (!p_fmat->Info().base_margin_.Empty()) {
      CHECK_EQ(p_fmat->Info().base_margin_.Shape(1), this->model_state_.OutputLength());
    }
  }

 private:
  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) {
    out_gpair->Reshape(info.num_row_, this->model_state_.OutputLength());
    obj_->GetGradient(preds, info, iter, out_gpair);
  }

  /*! \brief random number transformation seed. */
  static int32_t constexpr kRandSeedMagic = 127;
  // gradient pairs
  GradientContainer gpair_;
};

constexpr int32_t LearnerImpl::kRandSeedMagic;

Learner* Learner::Create(const std::vector<std::shared_ptr<DMatrix>>& cache_data) {
  return new LearnerImpl(cache_data);
}
}  // namespace xgboost
