/*!
 * Copyright 2014-2020 by Contributors
 * \file learner.cc
 * \brief Implementation of learning algorithm.
 * \author Tianqi Chen
 */
#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <stack>
#include <utility>
#include <vector>

#include "xgboost/base.h"
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
  /*! \brief reserved field */
  int reserved[27];
  /*! \brief constructor */
  LearnerModelParamLegacy() {
    std::memset(this, 0, sizeof(LearnerModelParamLegacy));
    base_score = 0.5f;
    major_version = std::get<0>(Version::Self());
    minor_version = std::get<1>(Version::Self());
    static_assert(sizeof(LearnerModelParamLegacy) == 136,
                  "Do not change the size of this struct, as it will break binary IO.");
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
                       : static_cast<uint32_t>(user_param.num_class)}
{}

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


using XGBAPIThreadLocalStore =
    dmlc::ThreadLocalStore<std::map<Learner const *, XGBAPIThreadLocalEntry>>;

struct LearnerAttributes {
  static std::string const kEvalMetric;  // NOLINT

  PredictionContainer cache;
  bool need_configuration { true };
  std::map<std::string, std::string> cfg;
  // Stores information like best-iteration for early stopping.
  std::map<std::string, std::string> attributes;
  common::Monitor monitor;
  LearnerModelParamLegacy mparam;
  LearnerModelParam learner_model_param;
  LearnerTrainParam tparam;
  std::vector<std::string> metric_names;
  GenericParameter generic_parameters;

  /*! \brief objective function */
  std::unique_ptr<ObjFunction> obj;
  /*! \brief The gradient booster used by the model*/
  std::unique_ptr<GradientBooster> gbm;
  /*! \brief The evaluation metrics used to evaluate the model. */
  std::vector<std::unique_ptr<Metric> > metrics;

  void SetParam(const std::string& key, const std::string& value) {
    this->need_configuration = true;
    if (key == kEvalMetric) {
      if (std::find(metric_names.cbegin(), metric_names.cend(),
                    value) == metric_names.cend()) {
        metric_names.emplace_back(value);
      }
    } else {
      cfg[key] = value;
    }
  }
  // Short hand for setting multiple parameters
  void SetParams(std::vector<std::pair<std::string, std::string>> const& args) {
    for (auto const& kv : args) {
      this->SetParam(kv.first, kv.second);
    }
  }

  void SetAttr(const std::string& key, const std::string& value) {
    attributes[key] = value;
    mparam.contain_extra_attrs = 1;
  }

  bool GetAttr(const std::string& key, std::string* out) const {
    auto it = attributes.find(key);
    if (it == attributes.end()) return false;
    *out = it->second;
    return true;
  }

  bool DelAttr(const std::string& key) {
    auto it = attributes.find(key);
    if (it == attributes.end()) { return false; }
    attributes.erase(it);
    return true;
  }

  std::vector<std::string> GetAttrNames() const {
    std::vector<std::string> out;
    for (auto const& kv : attributes) {
      out.emplace_back(kv.first);
    }
    return out;
  }

  const std::map<std::string, std::string>& GetConfigurationArguments() const {
    return cfg;
  }
};

std::string const LearnerAttributes::kEvalMetric {"eval_metric"};  // NOLINT


class LearnerConfiguration {
 public:
  void Configure(LearnerAttributes* attr) {
    if (!attr->need_configuration) { return; }

    attr->monitor.Start("Configure");
    auto old_tparam = attr->tparam;
    Args args = {attr->cfg.cbegin(), attr->cfg.cend()};

    attr->tparam.UpdateAllowUnknown(args);
    auto mparam_backup = attr->mparam;
    attr->mparam.UpdateAllowUnknown(args);
    attr->generic_parameters.UpdateAllowUnknown(args);
    attr->generic_parameters.CheckDeprecated();

    ConsoleLogger::Configure(args);
    if (attr->generic_parameters.nthread != 0) {
      omp_set_num_threads(attr->generic_parameters.nthread);
    }

    // add additional parameters
    // These are cosntraints that need to be satisfied.
    if (attr->tparam.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      attr->tparam.dsplit = DataSplitMode::kRow;
    }

    // set seed only before the model is initialized
    common::GlobalRandom().seed(attr->generic_parameters.seed);

    // must precede configure gbm since num_features is required for gbm
    this->ConfigureNumFeatures(attr);
    args = {attr->cfg.cbegin(), attr->cfg.cend()};  // renew
    this->ConfigureObjective(old_tparam, &args, attr);

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
    if (!attr->learner_model_param.Initialized() ||
        attr->mparam.base_score != mparam_backup.base_score) {
      attr->learner_model_param = LearnerModelParam(
          attr->mparam,
          attr->obj->ProbToMargin(attr->mparam.base_score));
    }

    this->ConfigureGBM(old_tparam, args, attr);
    attr->generic_parameters.ConfigureGpuId(attr->gbm->UseGPU());

    this->ConfigureMetrics(args, attr);

    attr->need_configuration = false;
    if (attr->generic_parameters.validate_parameters) {
      this->ValidateParameters(attr);
    }

    // FIXME(trivialfis): Clear the cache once binary IO is gone.
    attr->monitor.Stop("Configure");
  }

  void LoadConfig(Json const& in, LearnerAttributes* attr) {
    CHECK(IsA<Object>(in));
    Version::Load(in, true);

    auto const& learner_parameters = get<Object>(in["learner"]);
    fromJson(learner_parameters.at("learner_train_param"), &attr->tparam);

    auto const& gradient_booster = learner_parameters.at("gradient_booster");

    auto const& objective_fn = learner_parameters.at("objective");
    if (!attr->obj) {
      attr->obj.reset(ObjFunction::Create(attr->tparam.objective, &attr->generic_parameters));
    }
    attr->obj->LoadConfig(objective_fn);

    attr->tparam.booster = get<String>(gradient_booster["name"]);
    if (!attr->gbm) {
      attr->gbm.reset(GradientBooster::Create(attr->tparam.booster,
                                              &attr->generic_parameters,
                                              &attr->learner_model_param));
    }
    attr->gbm->LoadConfig(gradient_booster);

    auto const& j_metrics = learner_parameters.at("metrics");
    auto n_metrics = get<Array const>(j_metrics).size();
    attr->metric_names.resize(n_metrics);
    attr->metrics.resize(n_metrics);
    for (size_t i = 0; i < n_metrics; ++i) {
      attr->metric_names[i]= get<String>(j_metrics[i]);
      attr->metrics[i] = std::unique_ptr<Metric>(
          Metric::Create(attr->metric_names[i], &attr->generic_parameters));
    }

    fromJson(learner_parameters.at("generic_param"), &attr->generic_parameters);
    // make sure the GPU ID is valid in new environment before start running configure.
    attr->generic_parameters.ConfigureGpuId(false);

    attr->need_configuration = true;
  }

  void SaveConfig(Json* p_out, LearnerAttributes const* attr) const {
    CHECK(!attr->need_configuration) << "Call Configure before saving model.";
    Version::Save(p_out);
    Json& out { *p_out };
    // parameters
    out["learner"] = Object();
    auto& learner_parameters = out["learner"];

    learner_parameters["learner_train_param"] = toJson(attr->tparam);
    learner_parameters["learner_model_param"] = attr->mparam.ToJson();
    learner_parameters["gradient_booster"] = Object();
    auto& gradient_booster = learner_parameters["gradient_booster"];
    attr->gbm->SaveConfig(&gradient_booster);

    learner_parameters["objective"] = Object();
    auto& objective_fn = learner_parameters["objective"];
    attr->obj->SaveConfig(&objective_fn);

    std::vector<Json> metrics(attr->metrics.size());
    for (size_t i = 0; i < attr->metrics.size(); ++i) {
      metrics[i] = String(attr->metrics[i]->Name());
    }
    learner_parameters["metrics"] = Array(std::move(metrics));

    learner_parameters["generic_param"] = toJson(attr->generic_parameters);
  }

 private:
  void ValidateParameters(LearnerAttributes const* attr) {
    Json config { Object() };
    this->SaveConfig(&config, attr);
    std::stack<Json> stack;
    stack.push(config);
    std::string const postfix{"_param"};

    auto is_parameter = [&postfix](std::string const &key) {
      return key.size() > postfix.size() &&
             std::equal(postfix.rbegin(), postfix.rend(), key.rbegin());
    };

    // Extract all parameters
    std::vector<std::string> keys;
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

    keys.emplace_back(LearnerAttributes::kEvalMetric);
    keys.emplace_back("verbosity");
    keys.emplace_back("num_output_group");

    std::sort(keys.begin(), keys.end());

    std::vector<std::string> provided;
    for (auto const &kv : attr->cfg) {
      // FIXME(trivialfis): Make eval_metric a training parameter.
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
        ss << diff[i] << ", ";
      }
      ss << diff.back();
      ss << R"W( } might not be used.

  This may not be accurate due to some parameters are only used in language bindings but
  passed down to XGBoost core.  Or some parameters are not used but slip through this
  verification. Please open an issue if you find above cases.

)W";
      LOG(WARNING) << ss.str();
    }
  }

  void ConfigureNumFeatures(LearnerAttributes* attr) {
    // estimate feature bound
    // TODO(hcho3): Change num_feature to 64-bit integer
    unsigned num_feature = 0;
    for (auto & matrix : attr->cache.Container()) {
      CHECK(matrix.first);
      CHECK(!matrix.second.ref.expired());
      const uint64_t num_col = matrix.first->Info().num_col_;
      CHECK_LE(num_col, static_cast<uint64_t>(std::numeric_limits<unsigned>::max()))
          << "Unfortunately, XGBoost does not support data matrices with "
          << std::numeric_limits<unsigned>::max() << " features or greater";
      num_feature = std::max(num_feature, static_cast<uint32_t>(num_col));
    }
    // run allreduce on num_feature to find the maximum value
    rabit::Allreduce<rabit::op::Max>(&num_feature, 1, nullptr, nullptr, "num_feature");
    if (num_feature > attr->mparam.num_feature) {
      attr->mparam.num_feature = num_feature;
    }
    CHECK_NE(attr->mparam.num_feature, 0)
        << "0 feature is supplied.  Are you using raw Booster interface?";
    // Remove these once binary IO is gone.
    attr->cfg["num_feature"] = common::ToString(attr->mparam.num_feature);
    attr->cfg["num_class"] = common::ToString(attr->mparam.num_class);
  }

  void ConfigureGBM(LearnerTrainParam const &old, Args const &args,
                    LearnerAttributes *attr) const {
    if (attr->gbm == nullptr || old.booster != attr->tparam.booster) {
      attr->gbm.reset(GradientBooster::Create(attr->tparam.booster,
                                               &attr->generic_parameters,
                                               &attr->learner_model_param));
    }
    attr->gbm->Configure(args);
  }

  void ConfigureObjective(LearnerTrainParam const& old, Args* p_args,
                          LearnerAttributes *attr) {
    // Once binary IO is gone, NONE of these config is useful.
    if (attr->cfg.find("num_class") != attr->cfg.cend() && attr->cfg.at("num_class") != "0" &&
        attr->tparam.objective != "multi:softprob") {
      attr->cfg["num_output_group"] = attr->cfg["num_class"];
      if (atoi(attr->cfg["num_class"].c_str()) > 1 &&
          attr->cfg.count("objective") == 0) {
        attr->tparam.objective = "multi:softmax";
      }
    }

    if (attr->cfg.find("max_delta_step") == attr->cfg.cend() &&
        attr->cfg.find("objective") != attr->cfg.cend() &&
        attr->tparam.objective == "count:poisson") {
      // max_delta_step is a duplicated parameter in Poisson regression and tree param.
      // Rename one of them once binary IO is gone.
      attr->cfg["max_delta_step"] = kMaxDeltaStepDefaultValue;
    }
    if (attr->obj == nullptr || attr->tparam.objective != old.objective) {
      attr->obj.reset(ObjFunction::Create(attr->tparam.objective, &attr->generic_parameters));
    }
    auto& args = *p_args;
    args = {attr->cfg.cbegin(), attr->cfg.cend()};  // renew
    attr->obj->Configure(args);
  }

  void ConfigureMetrics(Args const& args, LearnerAttributes *attr) {
    for (auto const& name : attr->metric_names) {
      auto DupCheck = [&name](std::unique_ptr<Metric> const& m) {
                        return m->Name() != name;
                      };
      if (std::all_of(attr->metrics.begin(), attr->metrics.end(), DupCheck)) {
        attr->metrics.emplace_back(std::unique_ptr<Metric>(
            Metric::Create(name, &attr->generic_parameters)));
        attr->mparam.contain_eval_metrics = 1;
      }
    }
    for (auto& p_metric : attr->metrics) {
      p_metric->Configure(args);
    }
  }
};

class LearnerIO {
 private:
  std::set<std::string> saved_configs_ = {"num_round"};

 public:
  void LoadModel(Json const& in, LearnerAttributes* attr) {
    CHECK(IsA<Object>(in));
    Version::Load(in, false);
    auto const& learner = get<Object>(in["learner"]);
    attr->mparam.FromJson(learner.at("learner_model_param"));

    auto const& objective_fn = learner.at("objective");

    std::string name = get<String>(objective_fn["name"]);
    attr->tparam.UpdateAllowUnknown(Args{{"objective", name}});
    attr->obj.reset(ObjFunction::Create(name, &attr->generic_parameters));
    attr->obj->LoadConfig(objective_fn);

    auto const& gradient_booster = learner.at("gradient_booster");
    name = get<String>(gradient_booster["name"]);
    attr->tparam.UpdateAllowUnknown(Args{{"booster", name}});
    attr->gbm.reset(GradientBooster::Create(attr->tparam.booster,
                                             &attr->generic_parameters,
                                             &attr->learner_model_param));
    attr->gbm->LoadModel(gradient_booster);

    auto const& j_attributes = get<Object const>(learner.at("attributes"));
    attr->attributes.clear();
    for (auto const& kv : j_attributes) {
      attr->attributes[kv.first] = get<String const>(kv.second);
    }

    attr->need_configuration = true;
  }

  void SaveModel(Json* p_out, LearnerAttributes const* attr) const {
    CHECK(!attr->need_configuration) << "Call Configure before saving model.";

    Version::Save(p_out);
    Json& out { *p_out };

    out["learner"] = Object();
    auto& learner = out["learner"];

    learner["learner_model_param"] = attr->mparam.ToJson();
    learner["gradient_booster"] = Object();
    auto& gradient_booster = learner["gradient_booster"];
    attr->gbm->SaveModel(&gradient_booster);

    learner["objective"] = Object();
    auto& objective_fn = learner["objective"];
    attr->obj->SaveConfig(&objective_fn);

    learner["attributes"] = Object();
    for (auto const& kv : attr->attributes) {
      learner["attributes"][kv.first] = String(kv.second);
    }
  }
  // About to be deprecated by JSON format
  void LoadModel(dmlc::Stream* fi, LearnerAttributes* attr) {
    attr->generic_parameters.UpdateAllowUnknown(Args{});
    attr->tparam.Init(std::vector<std::pair<std::string, std::string>>{});
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
      // Dispatch to JSON
      auto json_stream = common::FixedSizeStream(&fp);
      std::string buffer;
      json_stream.Take(&buffer);
      auto model = Json::Load({buffer.c_str(), buffer.size()});
      this->LoadModel(model, attr);
      return;
    }
    // use the peekable reader.
    fi = &fp;
    // read parameter
    CHECK_EQ(fi->Read(&attr->mparam, sizeof(attr->mparam)), sizeof(attr->mparam))
        << "BoostLearner: wrong model format";

    CHECK(fi->Read(&attr->tparam.objective)) << "BoostLearner: wrong model format";
    CHECK(fi->Read(&attr->tparam.booster)) << "BoostLearner: wrong model format";

    attr->obj.reset(ObjFunction::Create(attr->tparam.objective, &attr->generic_parameters));
    attr->gbm.reset(GradientBooster::Create(attr->tparam.booster, &attr->generic_parameters,
                                             &attr->learner_model_param));
    attr->gbm->Load(fi);
    if (attr->mparam.contain_extra_attrs != 0) {
      std::vector<std::pair<std::string, std::string> > _attr;
      fi->Read(&_attr);
      for (auto& kv : _attr) {
        const std::string prefix = "SAVED_PARAM_";
        if (kv.first.find(prefix) == 0) {
          const std::string saved_param = kv.first.substr(prefix.length());
          if (saved_configs_.find(saved_param) != saved_configs_.end()) {
            attr->cfg[saved_param] = kv.second;
          }
        }
      }
      attr->attributes = std::map<std::string, std::string>(_attr.begin(), _attr.end());
    }
    bool warn_old_model { false };
    if (attr->attributes.find("count_poisson_max_delta_step") != attr->attributes.cend()) {
      // Loading model from < 1.0.0, objective is not saved.
      attr->cfg["max_delta_step"] = attr->attributes.at("count_poisson_max_delta_step");
      attr->attributes.erase("count_poisson_max_delta_step");
      warn_old_model = true;
    } else {
      warn_old_model = false;
    }

    if (attr->mparam.major_version >= 1) {
      attr->learner_model_param = LearnerModelParam(
          attr->mparam, attr->obj->ProbToMargin(attr->mparam.base_score));
    } else {
      // Before 1.0.0, base_score is saved as a transformed value, and there's no version
      // attribute in the saved model.
      attr->learner_model_param = LearnerModelParam(attr->mparam, attr->mparam.base_score);
      warn_old_model = true;
    }
    if (attr->attributes.find("objective") != attr->attributes.cend()) {
      auto obj_str = attr->attributes.at("objective");
      auto j_obj = Json::Load({obj_str.c_str(), obj_str.size()});
      attr->obj->LoadConfig(j_obj);
      attr->attributes.erase("objective");
    } else {
      warn_old_model = true;
    }
    if (attr->attributes.find("metrics") != attr->attributes.cend()) {
      auto metrics_str = attr->attributes.at("metrics");
      std::vector<std::string> names { common::Split(metrics_str, ';') };
      attr->attributes.erase("metrics");
      for (auto const& n : names) {
        attr->SetParam(attr->kEvalMetric, n);
      }
    }

    if (warn_old_model) {
      LOG(WARNING) << "Loading model from XGBoost < 1.0.0, consider saving it "
                      "again for improved compatibility";
    }

    // Renew the version.
    attr->mparam.major_version = std::get<0>(Version::Self());
    attr->mparam.minor_version = std::get<1>(Version::Self());

    attr->cfg["num_class"] = common::ToString(attr->mparam.num_class);
    attr->cfg["num_feature"] = common::ToString(attr->mparam.num_feature);

    auto n = attr->tparam.__DICT__();
    attr->cfg.insert(n.cbegin(), n.cend());

    // copy dsplit from config since it will not run again during restore
    if (attr->tparam.dsplit == DataSplitMode::kAuto && rabit::IsDistributed()) {
      attr->tparam.dsplit = DataSplitMode::kRow;
    }
    attr->need_configuration = true;
  }

  // Save model into binary format.  The code is about to be deprecated by more robust
  // JSON serialization format.  This function is uneffected by
  // `enable_experimental_json_serialization` as user might enable this flag for pickle
  // while still want a binary output.  As we are progressing at replacing the binary
  // format, there's no need to put too much effort on it.
  void SaveModel(dmlc::Stream* fo, LearnerAttributes const* attr) const {
    LearnerModelParamLegacy mparam = attr->mparam;  // make a copy to potentially modify
    std::vector<std::pair<std::string, std::string> > extra_attr;
    mparam.contain_extra_attrs = 1;

    {
      std::vector<std::string> saved_params;
      // check if rabit_bootstrap_cache were set to non zero before adding to checkpoint
      if (attr->cfg.find("rabit_bootstrap_cache") != attr->cfg.end() &&
        (attr->cfg.find("rabit_bootstrap_cache"))->second != "0") {
        std::copy(saved_configs_.begin(), saved_configs_.end(),
                  std::back_inserter(saved_params));
      }
      for (const auto& key : saved_params) {
        auto it = attr->cfg.find(key);
        if (it != attr->cfg.end()) {
          mparam.contain_extra_attrs = 1;
          extra_attr.emplace_back("SAVED_PARAM_" + key, it->second);
        }
      }
    }
    {
      // Similar to JSON model IO, we save the objective.
      Json j_obj { Object() };
      attr->obj->SaveConfig(&j_obj);
      std::string obj_doc;
      Json::Dump(j_obj, &obj_doc);
      extra_attr.emplace_back("objective", obj_doc);
    }
    // As of 1.0.0, JVM Package and R Package uses Save/Load model for serialization.
    // Remove this part once they are ported to use actual serialization methods.
    if (mparam.contain_eval_metrics != 0) {
      std::stringstream os;
      for (auto& ev : attr->metrics) {
        os << ev->Name() << ";";
      }
      extra_attr.emplace_back("metrics", os.str());
    }

    fo->Write(&mparam, sizeof(LearnerModelParamLegacy));
    fo->Write(attr->tparam.objective);
    fo->Write(attr->tparam.booster);
    attr->gbm->Save(fo);
    if (mparam.contain_extra_attrs != 0) {
      std::map<std::string, std::string> _attr(attr->attributes);
      for (const auto& kv : extra_attr) {
        _attr[kv.first] = kv.second;
      }
      fo->Write(std::vector<std::pair<std::string, std::string>>(
          _attr.begin(), _attr.end()));
    }
  }
};

/*!
 * \brief learner that performs gradient boosting for a specific objective
 * function. It does training and prediction.
 */
class LearnerImpl : public Learner {
 public:
  explicit LearnerImpl(std::vector<std::shared_ptr<DMatrix> > cache) {
    attr_.monitor.Init("Learner");
    for (std::shared_ptr<DMatrix> const& d : cache) {
      attr_.cache.Cache(d, GenericParameter::kCpuId);
    }
  }
  ~LearnerImpl() override {
    auto local_map = XGBAPIThreadLocalStore::Get();
    if (local_map->find(this) != local_map->end()) {
      local_map->erase(this);
    }
  }

  void Configure() override {
    this->cfg_.Configure(&attr_);
  }
  void LoadModel(dmlc::Stream* fi) override {
    this->io_.LoadModel(fi, &attr_);
  }
  void LoadModel(Json const& in) override {
    this->io_.LoadModel(in, &attr_);
  }
  void SaveModel(dmlc::Stream* fo) const override {
    this->io_.SaveModel(fo, &attr_);
  }
  void SaveModel(Json* p_out) const override {
    this->io_.SaveModel(p_out, &attr_);
  }
  void SaveConfig(Json* p_out) const override {
    this->cfg_.SaveConfig(p_out, &attr_);
  }
  void LoadConfig(Json const& in) override {
    this->cfg_.LoadConfig(in, &attr_);
  }

  void Save(dmlc::Stream* fo) const override {
    if (attr_.generic_parameters.enable_experimental_json_serialization) {
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
      // Avoid printing the content in loaded header, which might be random binary code.
      CHECK(header == serialisation_header_)  // NOLINT
          << R"doc(

  If you are loading a serialized model (like pickle in Python) generated by older
  XGBoost, please export the model by calling `Booster.save_model` from that version
  first, then load it back in current version.  There's a simple script for helping
  the process. See:

    https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

  for reference to the script, and more details about differences between saving model and
  serializing.

)doc";
      int64_t sz {-1};
      CHECK_EQ(fp.Read(&sz, sizeof(sz)), sizeof(sz));
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

  // Configuration before data is known.
  void CheckDataSplitMode() {
    if (rabit::IsDistributed()) {
      CHECK(attr_.tparam.dsplit != DataSplitMode::kAuto)
        << "Precondition violated; dsplit cannot be 'auto' in distributed mode";
      if (attr_.tparam.dsplit == DataSplitMode::kCol) {
        // 'distcol' updater hidden until it becomes functional again
        // See discussion at https://github.com/dmlc/xgboost/issues/1832
        LOG(FATAL) << "Column-wise data split is currently not supported.";
      }
    }
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    CHECK(!attr_.need_configuration)
        << "The model hasn't been built yet.  Are you using raw Booster interface?";
    return attr_.gbm->DumpModel(fmap, with_stats, format);
  }

  void UpdateOneIter(int iter, std::shared_ptr<DMatrix> train) override {
    attr_.monitor.Start("UpdateOneIter");
    TrainingObserver::Instance().Update(iter);
    this->Configure();
    if (attr_.generic_parameters.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(attr_.generic_parameters.seed * kRandSeedMagic + iter);
    }
    this->CheckDataSplitMode();
    this->ValidateDMatrix(train.get());

    auto& predt = attr_.cache.Cache(train, attr_.generic_parameters.gpu_id);

    attr_.monitor.Start("PredictRaw");
    this->PredictRaw(train.get(), &predt, true);
    attr_.monitor.Stop("PredictRaw");

    attr_.monitor.Start("GetGradient");
    attr_.obj->GetGradient(predt.predictions, train->Info(), iter, &gpair_);
    attr_.monitor.Stop("GetGradient");
    TrainingObserver::Instance().Observe(gpair_, "Gradients");

    attr_.gbm->DoBoost(train.get(), &gpair_, &predt);
    attr_.monitor.Stop("UpdateOneIter");
  }

  void BoostOneIter(int iter, std::shared_ptr<DMatrix> train,
                    HostDeviceVector<GradientPair>* in_gpair) override {
    attr_.monitor.Start("BoostOneIter");
    this->Configure();
    if (attr_.generic_parameters.seed_per_iteration || rabit::IsDistributed()) {
      common::GlobalRandom().seed(attr_.generic_parameters.seed * kRandSeedMagic + iter);
    }
    this->CheckDataSplitMode();
    this->ValidateDMatrix(train.get());
    attr_.cache.Cache(train, attr_.generic_parameters.gpu_id);

    attr_.gbm->DoBoost(train.get(), in_gpair, &attr_.cache.Entry(train.get()));
    attr_.monitor.Stop("BoostOneIter");
  }

  std::string EvalOneIter(int iter,
                          const std::vector<std::shared_ptr<DMatrix>>& data_sets,
                          const std::vector<std::string>& data_names) override {
    attr_.monitor.Start("EvalOneIter");
    this->Configure();

    std::ostringstream os;
    os << '[' << iter << ']' << std::setiosflags(std::ios::fixed);
    if (attr_.metrics.size() == 0 && attr_.tparam.disable_default_eval_metric <= 0) {
      attr_.metrics.emplace_back(
          Metric::Create(attr_.obj->DefaultEvalMetric(), &attr_.generic_parameters));
      attr_.metrics.back()->Configure({attr_.cfg.begin(), attr_.cfg.end()});
    }
    for (size_t i = 0; i < data_sets.size(); ++i) {
      std::shared_ptr<DMatrix> m = data_sets[i];
      auto &predt = attr_.cache.Cache(m, attr_.generic_parameters.gpu_id);
      this->ValidateDMatrix(m.get());
      this->PredictRaw(m.get(), &predt, false);

      auto &out = output_predictions_.Cache(m, attr_.generic_parameters.gpu_id).predictions;
      out.Resize(predt.predictions.Size());
      out.Copy(predt.predictions);

      attr_.obj->EvalTransform(&out);
      for (auto& ev : attr_.metrics) {
        os << '\t' << data_names[i] << '-' << ev->Name() << ':'
           << ev->Eval(out, m->Info(), attr_.tparam.dsplit == DataSplitMode::kRow);
      }
    }

    attr_.monitor.Stop("EvalOneIter");
    return os.str();
  }

  void Predict(std::shared_ptr<DMatrix> data, bool output_margin,
               HostDeviceVector<bst_float>* out_preds, unsigned ntree_limit,
               bool training,
               bool pred_leaf, bool pred_contribs, bool approx_contribs,
               bool pred_interactions) override {
    int multiple_predictions = static_cast<int>(pred_leaf) +
                               static_cast<int>(pred_interactions) +
                               static_cast<int>(pred_contribs);
    this->Configure();
    CHECK_LE(multiple_predictions, 1) << "Perform one kind of prediction at a time.";
    if (pred_contribs) {
      attr_.gbm->PredictContribution(data.get(), &out_preds->HostVector(),
                                     ntree_limit, approx_contribs);
    } else if (pred_interactions) {
      attr_.gbm->PredictInteractionContributions(
          data.get(), &out_preds->HostVector(), ntree_limit, approx_contribs);
    } else if (pred_leaf) {
      attr_.gbm->PredictLeaf(data.get(), &out_preds->HostVector(), ntree_limit);
    } else {
      auto& prediction = attr_.cache.Cache(data, attr_.generic_parameters.gpu_id);
      this->PredictRaw(data.get(), &prediction, training, ntree_limit);
      // Copy the prediction cache to output prediction. out_preds comes from C API
      out_preds->SetDevice(attr_.generic_parameters.gpu_id);
      out_preds->Resize(prediction.predictions.Size());
      out_preds->Copy(prediction.predictions);
      if (!output_margin) {
        attr_.obj->PredTransform(out_preds);
      }
    }
  }

  XGBAPIThreadLocalEntry& GetThreadLocal() const override {
    return (*XGBAPIThreadLocalStore::Get())[this];
  }

  const std::map<std::string, std::string>& GetConfigurationArguments() const override {
    return attr_.GetConfigurationArguments();
  }

  GenericParameter const& GetGenericParameter() const override {
    return attr_.generic_parameters;
  }
  bool AllowLazyCheckPoint() const override {
    return attr_.gbm->AllowLazyCheckPoint();
  }

  void SetParams(Args const& args) override { return attr_.SetParams(args); }
  void SetParam(const std::string &key, const std::string &value) override {
    attr_.SetParam(key, value);
  }
  void SetAttr(const std::string &key, const std::string &value) override {
    return attr_.SetAttr(key, value);
  }
  bool GetAttr(const std::string &key, std::string *out) const override {
    return attr_.GetAttr(key, out);
  }
  bool DelAttr(const std::string &key) override { return attr_.DelAttr(key); }
  std::vector<std::string> GetAttrNames() const override {
    return attr_.GetAttrNames();
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
  void PredictRaw(DMatrix* data, PredictionCacheEntry* out_preds,
                  bool training,
                  unsigned ntree_limit = 0) const {
    CHECK(attr_.gbm != nullptr) << "Predict must happen after Load or configuration";
    this->ValidateDMatrix(data);
    attr_.gbm->PredictBatch(data, out_preds, training, ntree_limit);
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

    auto const row_based_split = [this]() {
      return attr_.tparam.dsplit == DataSplitMode::kRow ||
             attr_.tparam.dsplit == DataSplitMode::kAuto;
    };
    bool const valid_features =
        !row_based_split() ||
        (attr_.learner_model_param.num_feature == p_fmat->Info().num_col_);
    std::string const msg {
      "Number of columns does not match number of features in booster."
    };
    if (attr_.generic_parameters.validate_features) {
      CHECK_EQ(attr_.learner_model_param.num_feature, p_fmat->Info().num_col_) << msg;
    } else if (!valid_features) {
      // Remove this and make the equality check fatal once spark can fix all failing tests.
      LOG(WARNING) << msg << " "
                   << "Columns: " << p_fmat->Info().num_col_ << " "
                   << "Features: " << attr_.learner_model_param.num_feature;
    }
  }

 private:
  LearnerAttributes attr_;
  LearnerConfiguration cfg_;
  LearnerIO io_;
  /*! \brief random number transformation seed. */
  static int32_t constexpr kRandSeedMagic = 127;
  // gradient pairs
  HostDeviceVector<GradientPair> gpair_;
  // Used to identify the offset of JSON string when
  // `enable_experimental_json_serialization' is set to false.  Will be removed once JSON
  // takes over.
  std::string const serialisation_header_ { u8"CONFIG-offset:" };
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
