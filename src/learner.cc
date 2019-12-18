/*!
 * Copyright 2014-2019 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include <dmlc/io.h>
#include <dmlc/parameter.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <ios>
#include <utility>
#include <vector>

#include "xgboost/base.h"
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
#include "common/version.h"

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
// implementation of base learner.
bool Learner::AllowLazyCheckPoint() const {
  return gbm_->AllowLazyCheckPoint();
}

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
  LearnerModelParamLegacy() {
    std::memset(this, 0, sizeof(LearnerModelParamLegacy));
    base_score = 0.5f;
  }
  // Skip other legacy fields.
  Json ToJson() const {
    Object obj;
    obj["base_score"] = std::to_string(base_score);
    obj["num_feature"] = std::to_string(num_feature);
    obj["num_class"] = std::to_string(num_class);
    return Json(std::move(obj));
  }
  void FromJson(Json const& obj) {
    auto const& j_param = get<Object const>(obj);
    std::map<std::string, std::string> m;
    m["base_score"] = get<String const>(j_param.at("base_score"));
    m["num_feature"] = get<String const>(j_param.at("num_feature"));
    m["num_class"] = get<String const>(j_param.at("num_class"));
    this->Init(m);
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
  }
};

LearnerModelParam::LearnerModelParam(
    LearnerModelParamLegacy const &user_param, float base_margin)
    : base_score{base_margin}, num_feature{user_param.num_feature},
      num_output_group{user_param.num_class == 0
                           ? 1
                           : static_cast<uint32_t>(user_param.num_class)} {}

struct LearnerTrainParam : public XGBoostParameter<LearnerTrainParam> {
  // data split mode, can be row, col, or none.
  DataSplitMode dsplit;
  // flag to disable default metric
  int disable_default_eval_metric;
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


DMLC_REGISTER_PARAMETER(LearnerModelParamLegacy);
DMLC_REGISTER_PARAMETER(LearnerTrainParam);
DMLC_REGISTER_PARAMETER(GenericParameter);

int constexpr GenericParameter::kCpuId;

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
    this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(kCpuId)}});
  } else if (gpu_id != kCpuId && gpu_id >= n_gpus) {
    this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(gpu_id % n_gpus)}});
  }
#else
  // Just set it to CPU, don't think about it.
  this->UpdateAllowUnknown(Args{{"gpu_id", std::to_string(kCpuId)}});
#endif  // defined(XGBOOST_USE_CUDA)
}

/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
class LearnerImpl : public Learner {
 public:
  explicit LearnerImpl(std::vector<std::shared_ptr<DMatrix> >  cache)
      : need_configuration_{true}, cache_(std::move(cache)) {
    monitor_.Init("Learner");
  }
  // Configuration before data is known.
  void Configure() override {
    if (!this->need_configuration_) { return; }

    monitor_.Start("Configure");
    auto old_tparam = tparam_;
    Args args = {cfg_.cbegin(), cfg_.cend()};

    tparam_.UpdateAllowUnknown(args);
    mparam_.UpdateAllowUnknown(args);
    generic_parameters_.UpdateAllowUnknown(args);
    generic_parameters_.CheckDeprecated();

    ConsoleLogger::Configure(args);
    if (generic_parameters_.nthread != 0) {
      omp_set_num_threads(generic_parameters_.nthread);
    }

    // add additional parameters
    // These are cosntraints that need to be satisfied.
    if (tparam_.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      tparam_.dsplit = DataSplitMode::kRow;
    }


    // set seed only before the model is initialized
    common::GlobalRandom().seed(generic_parameters_.seed);
    // must precede configure gbm since num_features is required for gbm
    this->ConfigureNumFeatures();
    args = {cfg_.cbegin(), cfg_.cend()};  // renew
    this->ConfigureObjective(old_tparam, &args);
    this->ConfigureGBM(old_tparam, args);
    this->ConfigureMetrics(args);

    generic_parameters_.ConfigureGpuId(this->gbm_->UseGPU());

    learner_model_param_ = LearnerModelParam(mparam_,
                                             obj_->ProbToMargin(mparam_.base_score));

    this->need_configuration_ = false;
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

  void LoadModel(Json const& in) override {
    CHECK(IsA<Object>(in));
    Version::Load(in, false);
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
    gbm_.reset(GradientBooster::Create(tparam_.booster,
                                       &generic_parameters_, &learner_model_param_,
                                       cache_));
    gbm_->LoadModel(gradient_booster);

    learner_model_param_ = LearnerModelParam(mparam_,
                                             obj_->ProbToMargin(mparam_.base_score));

    auto const& j_attributes = get<Object const>(learner.at("attributes"));
    attributes_.clear();
    for (auto const& kv : j_attributes) {
      attributes_[kv.first] = get<String const>(kv.second);
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
  }

  void LoadConfig(Json const& in) override {
    CHECK(IsA<Object>(in));
    Version::Load(in, true);

    auto const& learner_parameters = get<Object>(in["learner"]);
    fromJson(learner_parameters.at("learner_train_param"), &tparam_);

    auto const& gradient_booster = learner_parameters.at("gradient_booster");

    auto const& objective_fn = learner_parameters.at("objective");
    if (!obj_) {
      obj_.reset(ObjFunction::Create(tparam_.objective, &generic_parameters_));
    }
    obj_->LoadConfig(objective_fn);

    tparam_.booster = get<String>(gradient_booster["name"]);
    if (!gbm_) {
      gbm_.reset(GradientBooster::Create(tparam_.booster,
                                         &generic_parameters_, &learner_model_param_,
                                         cache_));
    }
    gbm_->LoadConfig(gradient_booster);

    auto const& j_metrics = learner_parameters.at("metrics");
    auto n_metrics = get<Array const>(j_metrics).size();
    metric_names_.resize(n_metrics);
    metrics_.resize(n_metrics);
    for (size_t i = 0; i < n_metrics; ++i) {
      metric_names_[i]= get<String>(j_metrics[i]);
      metrics_[i] = std::unique_ptr<Metric>(
          Metric::Create(metric_names_[i], &generic_parameters_));
    }

    fromJson(learner_parameters.at("generic_param"), &generic_parameters_);

    this->need_configuration_ = true;
  }

  void SaveConfig(Json* p_out) const override {
    CHECK(!this->need_configuration_) << "Call Configure before saving model.";
    Version::Save(p_out);
    Json& out { *p_out };
    // parameters
    out["learner"] = Object();
    auto& learner_parameters = out["learner"];

    learner_parameters["learner_train_param"] = toJson(tparam_);
    learner_parameters["gradient_booster"] = Object();
    auto& gradient_booster = learner_parameters["gradient_booster"];
    gbm_->SaveConfig(&gradient_booster);

    learner_parameters["objective"] = Object();
    auto& objective_fn = learner_parameters["objective"];
    obj_->SaveConfig(&objective_fn);

    std::vector<Json> metrics(metrics_.size());
    for (size_t i = 0; i < metrics_.size(); ++i) {
      metrics[i] = String(metrics_[i]->Name());
    }
    learner_parameters["metrics"] = Array(std::move(metrics));

    learner_parameters["generic_param"] = toJson(generic_parameters_);
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

    if (header[0] == '{') {
      auto json_stream = common::FixedSizeStream(&fp);
      std::string buffer;
      json_stream.Take(&buffer);
      auto model = Json::Load({buffer.c_str(), buffer.size()});
      this->LoadModel(model);
      return;
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
    obj_.reset(ObjFunction::Create(tparam_.objective, &generic_parameters_));
    gbm_.reset(GradientBooster::Create(tparam_.booster, &generic_parameters_,
                                       &learner_model_param_, cache_));
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
    if (tparam_.objective == "count:poisson") {
      std::string max_delta_step;
      fi->Read(&max_delta_step);
      cfg_["max_delta_step"] = max_delta_step;
    }
    if (mparam_.contain_eval_metrics != 0) {
      std::vector<std::string> metr;
      fi->Read(&metr);
      for (auto name : metr) {
        metrics_.emplace_back(Metric::Create(name, &generic_parameters_));
      }
    }

    cfg_["num_class"] = common::ToString(mparam_.num_class);
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);

    auto n = tparam_.__DICT__();
    cfg_.insert(n.cbegin(), n.cend());

    Args args = {cfg_.cbegin(), cfg_.cend()};
    generic_parameters_.UpdateAllowUnknown(args);
    gbm_->Configure(args);
    obj_->Configure({cfg_.begin(), cfg_.end()});

    for (auto& p_metric : metrics_) {
      p_metric->Configure({cfg_.begin(), cfg_.end()});
    }

    // copy dsplit from config since it will not run again during restore
    if (tparam_.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      tparam_.dsplit = DataSplitMode::kRow;
    }

    this->Configure();
  }

  // Save model into binary format.  The code is about to be deprecated by more robust
  // JSON serialization format.  This function is uneffected by
  // `enable_experimental_json_serialization` as user might enable this flag for pickle
  // while still want a binary output.  As we are progressing at replacing the binary
  // format, there's no need to put too much effort on it.
  void SaveModel(dmlc::Stream* fo) const override {
    LearnerModelParamLegacy mparam = mparam_;  // make a copy to potentially modify
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
      std::vector<std::string> saved_params;
      // check if rabit_bootstrap_cache were set to non zero before adding to checkpoint
      if (cfg_.find("rabit_bootstrap_cache") != cfg_.end() &&
        (cfg_.find("rabit_bootstrap_cache"))->second != "0") {
        std::copy(saved_configs_.begin(), saved_configs_.end(),
                  std::back_inserter(saved_params));
      }
      for (const auto& key : saved_params) {
        auto it = cfg_.find(key);
        if (it != cfg_.end()) {
          mparam.contain_extra_attrs = 1;
          extra_attr.emplace_back("SAVED_PARAM_" + key, it->second);
        }
      }
    }
    fo->Write(&mparam, sizeof(LearnerModelParamLegacy));
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

  void Save(dmlc::Stream* fo) const override {
    if (generic_parameters_.enable_experimental_json_serialization) {
      Json memory_snapshot{Object()};
      memory_snapshot["Model"] = Object();
      auto &model = memory_snapshot["Model"];
      this->SaveModel(&model);
      memory_snapshot["Config"] = Object();
      auto &config = memory_snapshot["Config"];
      this->SaveConfig(&config);
      std::string out_str;
      Json::Dump(memory_snapshot, &out_str);
      fo->Write(out_str.c_str(), out_str.size());
    } else {
      std::string binary_buf;
      common::MemoryBufferStream s(&binary_buf);
      this->SaveModel(&s);
      Json config{ Object() };
      // Do not use std::size_t as it's not portable.
      int64_t const json_offset = binary_buf.size();
      this->SaveConfig(&config);
      std::string config_str;
      Json::Dump(config, &config_str);
      // concatonate the model and config at final output, it's a temporary solution for
      // continuing support for binary model format
      fo->Write(&serialisation_header_[0], serialisation_header_.size());
      fo->Write(&json_offset, sizeof(json_offset));
      fo->Write(&binary_buf[0], binary_buf.size());
      fo->Write(&config_str[0], config_str.size());
    }
  }

  void Load(dmlc::Stream* fi) override {
    common::PeekableInStream fp(fi);
    char c {0};
    fp.PeekRead(&c, 1);
    if (c == '{') {
      std::string buffer;
      common::FixedSizeStream{&fp}.Take(&buffer);
      auto memory_snapshot = Json::Load({buffer.c_str(), buffer.size()});
      this->LoadModel(memory_snapshot["Model"]);
      this->LoadConfig(memory_snapshot["Config"]);
    } else {
      std::string header;
      header.resize(serialisation_header_.size());
      CHECK_EQ(fp.Read(&header[0], header.size()), serialisation_header_.size());
      CHECK_EQ(header, serialisation_header_);

      int64_t json_offset {-1};
      CHECK_EQ(fp.Read(&json_offset, sizeof(json_offset)), sizeof(json_offset));
      CHECK_GT(json_offset, 0);
      std::string buffer;
      common::FixedSizeStream{&fp}.Take(&buffer);

      common::MemoryFixSizeBuffer binary_buf(&buffer[0], json_offset);
      this->LoadModel(&binary_buf);

      common::MemoryFixSizeBuffer json_buf {&buffer[0] + json_offset,
                                            buffer.size() - json_offset};
      auto config = Json::Load({buffer.c_str() + json_offset, buffer.size() - json_offset});
      this->LoadConfig(config);
    }
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    CHECK(!this->need_configuration_)
        << "The model hasn't been built yet.  Are you using raw Booster interface?";
    return gbm_->DumpModel(fmap, with_stats, format);
  }

  void UpdateOneIter(int iter, DMatrix* train) override {
    monitor_.Start("UpdateOneIter");
    TrainingObserver::Instance().Update(iter);
    this->Configure();
    if (generic_parameters_.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(generic_parameters_.seed * kRandSeedMagic + iter);
    }
    this->CheckDataSplitMode();
    this->ValidateDMatrix(train);

    monitor_.Start("PredictRaw");
    this->PredictRaw(train, &preds_[train]);
    monitor_.Stop("PredictRaw");
    TrainingObserver::Instance().Observe(preds_[train], "Predictions");

    monitor_.Start("GetGradient");
    obj_->GetGradient(preds_[train], train->Info(), iter, &gpair_);
    monitor_.Stop("GetGradient");
    TrainingObserver::Instance().Observe(gpair_, "Gradients");

    gbm_->DoBoost(train, &gpair_, obj_.get());
    monitor_.Stop("UpdateOneIter");
  }

  void BoostOneIter(int iter, DMatrix* train,
                    HostDeviceVector<GradientPair>* in_gpair) override {
    monitor_.Start("BoostOneIter");
    this->Configure();
    if (generic_parameters_.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(generic_parameters_.seed * kRandSeedMagic + iter);
    }
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
      metrics_.emplace_back(Metric::Create(obj_->DefaultEvalMetric(), &generic_parameters_));
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
    return generic_parameters_;
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
        << "Predict must happen after Load or configuration";
    this->ValidateDMatrix(data);
    gbm_->PredictBatch(data, out_preds, ntree_limit);
  }

  void ConfigureObjective(LearnerTrainParam const& old, Args* p_args) {
    // Once binary IO is gone, NONE of these config is useful.
    if (cfg_.find("num_class") != cfg_.cend() && cfg_.at("num_class") != "0") {
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

  void ConfigureGBM(LearnerTrainParam const& old, Args const& args) {
    if (gbm_ == nullptr || old.booster != tparam_.booster) {
      gbm_.reset(GradientBooster::Create(tparam_.booster, &generic_parameters_,
                                         &learner_model_param_, cache_));
    }
    gbm_->Configure(args);
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
      num_feature = std::max(num_feature, static_cast<uint32_t>(num_col));
    }
    // run allreduce on num_feature to find the maximum value
    rabit::Allreduce<rabit::op::Max>(&num_feature, 1, nullptr, nullptr, "num_feature");
    if (num_feature > mparam_.num_feature) {
      mparam_.num_feature = num_feature;
    }
    CHECK_NE(mparam_.num_feature, 0)
        << "0 feature is supplied.  Are you using raw Booster interface?";
    learner_model_param_.num_feature = mparam_.num_feature;
    // Remove these once binary IO is gone.
    cfg_["num_feature"] = common::ToString(mparam_.num_feature);
    cfg_["num_class"] = common::ToString(mparam_.num_class);
  }

  void ValidateDMatrix(DMatrix* p_fmat) const {
    MetaInfo const& info = p_fmat->Info();
    auto const& weights = info.weights_;
    if (info.group_ptr_.size() != 0 && weights.Size() != 0) {
      CHECK(weights.Size() == info.group_ptr_.size() - 1)
          << "\n"
          << "weights size: " << weights.Size()            << ", "
          << "groups size: "  << info.group_ptr_.size() -1 << ", "
          << "num rows: "     << p_fmat->Info().num_row_   << "\n"
          << "Number of weights should be equal to number of groups in ranking task.";
    }
  }

  // model parameter
  LearnerModelParamLegacy mparam_;
  LearnerModelParam learner_model_param_;
  LearnerTrainParam tparam_;
  // Used to identify the offset of JSON string when
  // `enable_experimental_json_serialization' is set to false.  Will be removed once JSON
  // takes over.
  std::string const serialisation_header_ { u8"CONFIG-offset:" };
  // configurations
  std::map<std::string, std::string> cfg_;
  std::map<std::string, std::string> attributes_;
  std::vector<std::string> metric_names_;
  static std::string const kEvalMetric;  // NOLINT
  // temporal storages for prediction
  std::map<DMatrix*, HostDeviceVector<bst_float>> preds_;
  // gradient pairs
  HostDeviceVector<GradientPair> gpair_;
  bool need_configuration_;

 private:
  /*! \brief random number transformation seed. */
  static int32_t constexpr kRandSeedMagic = 127;
  // internal cached dmatrix
  std::vector<std::shared_ptr<DMatrix> > cache_;

  common::Monitor monitor_;

  /*! \brief (Deprecated) saved config keys used to restore failed worker */
  std::set<std::string> saved_configs_ = {"num_round"};
};

std::string const LearnerImpl::kEvalMetric {"eval_metric"};  // NOLINT

constexpr int32_t LearnerImpl::kRandSeedMagic;

Learner* Learner::Create(
    const std::vector<std::shared_ptr<DMatrix> >& cache_data) {
  return new LearnerImpl(cache_data);
}
}  // namespace xgboost
