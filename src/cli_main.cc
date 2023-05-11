/*!
 * Copyright 2014-2020 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of xgboost.
 *  This file is not included in dynamic library.
 */
#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)

#include <dmlc/timer.h>

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/parameter.h>

#include <iomanip>
#include <ctime>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>
#include "collective/communicator-inl.h"
#include "common/common.h"
#include "common/config.h"
#include "common/io.h"
#include "common/version.h"
#include "c_api/c_api_utils.h"

namespace xgboost {
enum CLITask {
  kTrain = 0,
  kDumpModel = 1,
  kPredict = 2
};

struct CLIParam : public XGBoostParameter<CLIParam> {
  /*! \brief the task name */
  int task;
  /*! \brief whether evaluate training statistics */
  bool eval_train;
  /*! \brief number of boosting iterations */
  int num_round;
  /*! \brief the period to save the model, 0 means only save the final round model */
  int save_period;
  /*! \brief the path of training set */
  std::string train_path;
  /*! \brief path of test dataset */
  std::string test_path;
  /*! \brief the path of test model file, or file to restart training */
  std::string model_in;
  /*! \brief the path of final model file, to be saved */
  std::string model_out;
  /*! \brief the path of directory containing the saved models */
  std::string model_dir;
  /*! \brief name of predict file */
  std::string name_pred;
  /*! \brief data split mode */
  int dsplit;
  /*!\brief limit number of trees in prediction */
  int ntree_limit;
  int iteration_begin;
  int iteration_end;
  /*!\brief whether to directly output margin value */
  bool pred_margin;
  /*! \brief whether dump statistics along with model */
  int dump_stats;
  /*! \brief what format to dump the model in */
  std::string dump_format;
  /*! \brief name of feature map */
  std::string name_fmap;
  /*! \brief name of dump file */
  std::string name_dump;
  /*! \brief the paths of validation data sets */
  std::vector<std::string> eval_data_paths;
  /*! \brief the names of the evaluation data used in output log */
  std::vector<std::string> eval_data_names;
  /*! \brief all the configurations */
  std::vector<std::pair<std::string, std::string> > cfg;

  static constexpr char const* const kNull = "NULL";

  // declare parameters
  DMLC_DECLARE_PARAMETER(CLIParam) {
    // NOTE: declare everything except eval_data_paths.
    DMLC_DECLARE_FIELD(task).set_default(kTrain)
        .add_enum("train", kTrain)
        .add_enum("dump", kDumpModel)
        .add_enum("pred", kPredict)
        .describe("Task to be performed by the CLI program.");
    DMLC_DECLARE_FIELD(eval_train).set_default(false)
        .describe("Whether evaluate on training data during training.");
    DMLC_DECLARE_FIELD(num_round).set_default(10).set_lower_bound(1)
        .describe("Number of boosting iterations");
    DMLC_DECLARE_FIELD(save_period).set_default(0).set_lower_bound(0)
        .describe("The period to save the model, 0 means only save final model.");
    DMLC_DECLARE_FIELD(train_path).set_default("NULL")
        .describe("Training data path.");
    DMLC_DECLARE_FIELD(test_path).set_default("NULL")
        .describe("Test data path.");
    DMLC_DECLARE_FIELD(model_in).set_default("NULL")
        .describe("Input model path, if any.");
    DMLC_DECLARE_FIELD(model_out).set_default("NULL")
        .describe("Output model path, if any.");
    DMLC_DECLARE_FIELD(model_dir).set_default("./")
        .describe("Output directory of period checkpoint.");
    DMLC_DECLARE_FIELD(name_pred).set_default("pred.txt")
        .describe("Name of the prediction file.");
    DMLC_DECLARE_FIELD(dsplit).set_default(0)
        .add_enum("row", 0)
        .add_enum("col", 1)
        .describe("Data split mode.");
    DMLC_DECLARE_FIELD(ntree_limit).set_default(0).set_lower_bound(0)
        .describe("(Deprecated) Use iteration_begin/iteration_end instead.");
    DMLC_DECLARE_FIELD(iteration_begin).set_default(0).set_lower_bound(0)
        .describe("Begining of boosted tree iteration used for prediction.");
    DMLC_DECLARE_FIELD(iteration_end).set_default(0).set_lower_bound(0)
        .describe("End of boosted tree iteration used for prediction.  0 means all the trees.");
    DMLC_DECLARE_FIELD(pred_margin).set_default(false)
        .describe("Whether to predict margin value instead of probability.");
    DMLC_DECLARE_FIELD(dump_stats).set_default(false)
        .describe("Whether dump the model statistics.");
    DMLC_DECLARE_FIELD(dump_format).set_default("text")
        .describe("What format to dump the model in.");
    DMLC_DECLARE_FIELD(name_fmap).set_default("NULL")
        .describe("Name of the feature map file.");
    DMLC_DECLARE_FIELD(name_dump).set_default("dump.txt")
        .describe("Name of the output dump text file.");
    // alias
    DMLC_DECLARE_ALIAS(train_path, data);
    DMLC_DECLARE_ALIAS(test_path, test:data);
    DMLC_DECLARE_ALIAS(name_fmap, fmap);
  }
  // customized configure function of CLIParam
  inline void Configure(const std::vector<std::pair<std::string, std::string> >& _cfg) {
    // Don't copy the configuration to enable parameter validation.
    auto unknown_cfg = this->UpdateAllowUnknown(_cfg);
    this->cfg.emplace_back("validate_parameters", "True");
    for (const auto& kv : unknown_cfg) {
      if (!strncmp("eval[", kv.first.c_str(), 5)) {
        char evname[256];
        CHECK_EQ(sscanf(kv.first.c_str(), "eval[%[^]]", evname), 1)
            << "must specify evaluation name for display";
        eval_data_names.emplace_back(evname);
        eval_data_paths.push_back(kv.second);
      } else {
        this->cfg.emplace_back(kv);
      }
    }
    // constraint.
    if (name_pred == "stdout") {
      save_period = 0;
    }
  }
};

constexpr char const* const CLIParam::kNull;

DMLC_REGISTER_PARAMETER(CLIParam);

std::string CliHelp() {
  return "Use xgboost -h for showing help information.\n";
}

void CLIError(dmlc::Error const& e) {
  std::cerr << "Error running xgboost:\n\n"
            << e.what() << "\n"
            << CliHelp()
            << std::endl;
}

class CLI {
  CLIParam param_;
  std::unique_ptr<Learner> learner_;
  enum Print {
    kNone,
    kVersion,
    kHelp
  } print_info_ {kNone};

  void ResetLearner(std::vector<std::shared_ptr<DMatrix>> const &matrices) {
    learner_.reset(Learner::Create(matrices));
    if (param_.model_in != CLIParam::kNull) {
      this->LoadModel(param_.model_in, learner_.get());
      learner_->SetParams(param_.cfg);
    } else {
      learner_->SetParams(param_.cfg);
    }
    learner_->Configure();
  }

  void CLITrain() {
    const double tstart_data_load = dmlc::GetTime();
    if (collective::IsDistributed()) {
      std::string pname = collective::GetProcessorName();
      LOG(CONSOLE) << "start " << pname << ":" << collective::GetRank();
    }
    // load in data.
    std::shared_ptr<DMatrix> dtrain(DMatrix::Load(
        param_.train_path, ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(),
        static_cast<DataSplitMode>(param_.dsplit)));
    std::vector<std::shared_ptr<DMatrix>> deval;
    std::vector<std::shared_ptr<DMatrix>> cache_mats;
    std::vector<std::shared_ptr<DMatrix>> eval_datasets;
    cache_mats.push_back(dtrain);
    for (size_t i = 0; i < param_.eval_data_names.size(); ++i) {
      deval.emplace_back(std::shared_ptr<DMatrix>(
          DMatrix::Load(param_.eval_data_paths[i],
                        ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(),
                        static_cast<DataSplitMode>(param_.dsplit))));
      eval_datasets.push_back(deval.back());
      cache_mats.push_back(deval.back());
    }
    std::vector<std::string> eval_data_names = param_.eval_data_names;
    if (param_.eval_train) {
      eval_datasets.push_back(dtrain);
      eval_data_names.emplace_back("train");
    }
    // initialize the learner.
    this->ResetLearner(cache_mats);
    LOG(INFO) << "Loading data: " << dmlc::GetTime() - tstart_data_load
              << " sec";

    // start training.
    const double start = dmlc::GetTime();
    int32_t version = 0;
    for (int i = version / 2; i < param_.num_round; ++i) {
      double elapsed = dmlc::GetTime() - start;
      if (version % 2 == 0) {
        LOG(INFO) << "boosting round " << i << ", " << elapsed
                  << " sec elapsed";
        learner_->UpdateOneIter(i, dtrain);
        version += 1;
      }
      std::string res = learner_->EvalOneIter(i, eval_datasets, eval_data_names);
      if (collective::IsDistributed()) {
        if (collective::GetRank() == 0) {
          LOG(TRACKER) << res;
        }
      } else {
        LOG(CONSOLE) << res;
      }
      if (param_.save_period != 0 && (i + 1) % param_.save_period == 0 &&
          collective::GetRank() == 0) {
        std::ostringstream os;
        os << param_.model_dir << '/' << std::setfill('0') << std::setw(4)
           << i + 1 << ".model";
        this->SaveModel(os.str(), learner_.get());
      }

      version += 1;
    }
    LOG(INFO) << "Complete Training loop time: " << dmlc::GetTime() - start
              << " sec";
    // always save final round
    if ((param_.save_period == 0 ||
         param_.num_round % param_.save_period != 0) &&
         collective::GetRank() == 0) {
      std::ostringstream os;
      if (param_.model_out == CLIParam::kNull) {
        os << param_.model_dir << '/' << std::setfill('0') << std::setw(4)
           << param_.num_round << ".model";
      } else {
        os << param_.model_out;
      }
      this->SaveModel(os.str(), learner_.get());
    }

    double elapsed = dmlc::GetTime() - start;
    LOG(INFO) << "update end, " << elapsed << " sec in all";
  }

  void CLIDumpModel() {
    FeatureMap fmap;
    if (param_.name_fmap != CLIParam::kNull) {
      std::unique_ptr<dmlc::Stream> fs(
          dmlc::Stream::Create(param_.name_fmap.c_str(), "r"));
      dmlc::istream is(fs.get());
      fmap.LoadText(is);
    }
    // load model
    CHECK_NE(param_.model_in, CLIParam::kNull) << "Must specify model_in for dump";
    this->ResetLearner({});

    // dump data
    std::vector<std::string> dump =
        learner_->DumpModel(fmap, param_.dump_stats, param_.dump_format);
    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(param_.name_dump.c_str(), "w"));
    dmlc::ostream os(fo.get());
    if (param_.dump_format == "json") {
      os << "[" << std::endl;
      for (size_t i = 0; i < dump.size(); ++i) {
        if (i != 0) {
          os << "," << std::endl;
        }
        os << dump[i];  // Dump the previously generated JSON here
      }
      os << std::endl << "]" << std::endl;
    } else {
      for (size_t i = 0; i < dump.size(); ++i) {
        os << "booster[" << i << "]:\n";
        os << dump[i];
      }
    }
    // force flush before fo destruct.
    os.set_stream(nullptr);
  }

  void CLIPredict() {
    CHECK_NE(param_.test_path, CLIParam::kNull)
        << "Test dataset parameter test:data must be specified.";
    // load data
    std::shared_ptr<DMatrix> dtest(DMatrix::Load(
        param_.test_path,
        ConsoleLogger::GlobalVerbosity() > ConsoleLogger::DefaultVerbosity(),
        static_cast<DataSplitMode>(param_.dsplit)));
    // load model
    CHECK_NE(param_.model_in, CLIParam::kNull) << "Must specify model_in for predict";
    this->ResetLearner({});

    LOG(INFO) << "Start prediction...";
    HostDeviceVector<bst_float> preds;
    if (param_.ntree_limit != 0) {
      param_.iteration_end = GetIterationFromTreeLimit(param_.ntree_limit, learner_.get());
      LOG(WARNING) << "`ntree_limit` is deprecated, use `iteration_begin` and "
                      "`iteration_end` instead.";
    }
    learner_->Predict(dtest, param_.pred_margin, &preds, param_.iteration_begin,
                      param_.iteration_end);
    LOG(CONSOLE) << "Writing prediction to " << param_.name_pred;

    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(param_.name_pred.c_str(), "w"));
    dmlc::ostream os(fo.get());
    for (bst_float p : preds.ConstHostVector()) {
      os << std::setprecision(std::numeric_limits<bst_float>::max_digits10) << p
         << '\n';
    }
    // force flush before fo destruct.
    os.set_stream(nullptr);
  }

  void LoadModel(std::string const& path, Learner* learner) const {
    if (common::FileExtension(path) == "json") {
      auto str = common::LoadSequentialFile(path);
      CHECK_GT(str.size(), 2);
      CHECK_EQ(str[0], '{');
      Json in{Json::Load({str.c_str(), str.size()})};
      learner->LoadModel(in);
    } else {
      std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(path.c_str(), "r"));
      learner->LoadModel(fi.get());
    }
  }

  void SaveModel(std::string const& path, Learner* learner) const {
    learner->Configure();
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(path.c_str(), "w"));
    if (common::FileExtension(path) == "json") {
      Json out{Object()};
      learner->SaveModel(&out);
      std::string str;
      Json::Dump(out, &str);
      fo->Write(str.c_str(), str.size());
    } else {
      learner->SaveModel(fo.get());
    }
  }

  void PrintHelp() const {
    std::cout << "Usage: xgboost [ -h ] [ -V ] [ config file ] [ arguments ]" << std::endl;
    std::stringstream ss;
    ss << R"(
  Options and arguments:

    -h, --help
       Print this message.

    -V, --version
       Print XGBoost version.

    arguments
       Extra parameters that are not specified in config file, see below.

  Config file specifies the configuration for both training and testing.  Each line
  containing the [attribute] = [value] configuration.

  General XGBoost parameters:

    https://xgboost.readthedocs.io/en/latest/parameter.html

  Command line interface specfic parameters:

)";

    std::string help = param_.__DOC__();
    auto splited = common::Split(help, '\n');
    for (auto str : splited) {
      ss << "    " << str << '\n';
    }
    ss << R"(    eval[NAME]: string, optional, default='NULL'
        Path to evaluation data, with NAME as data name.
)";

    ss << R"(
  Example:  train.conf

    # General parameters
    booster = gbtree
    objective = reg:squarederror
    eta = 1.0
    gamma = 1.0
    seed = 0
    min_child_weight = 0
    max_depth = 3

    # Training arguments for CLI.
    num_round = 2
    save_period = 0
    data = "demo/data/agaricus.txt.train?format=libsvm"
    eval[test] = "demo/data/agaricus.txt.test?format=libsvm"

  See demo/ directory in XGBoost for more examples.
)";
    std::cout << ss.str() << std::endl;
  }

  void PrintVersion() const {
    auto ver = Version::String(Version::Self());
    std::cout << "XGBoost: " << ver << std::endl;
  }

 public:
  CLI(int argc, char* argv[]) {
    if (argc < 2) {
      this->PrintHelp();
      exit(1);
    }
    for (int i = 0; i < argc; ++i) {
      std::string str {argv[i]};
      if (str == "-h" || str == "--help") {
        print_info_ = kHelp;
        break;
      } else if (str == "-V" || str == "--version") {
        print_info_ = kVersion;
        break;
      }
    }
    if (print_info_ != kNone) {
      return;
    }

    std::string config_path = argv[1];

    common::ConfigParser cp(config_path);
    auto cfg = cp.Parse();

    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        cfg.emplace_back(std::string(name), std::string(val));
      }
    }

    // Initialize the collective communicator.
    Json json{JsonObject()};
    for (auto& kv : cfg) {
      json[kv.first] = String(kv.second);
    }
    collective::Init(json);

    param_.Configure(cfg);
  }

  int Run() {
    switch (this->print_info_) {
    case kNone:
      break;
    case kVersion: {
      this->PrintVersion();
      return 0;
    }
    case kHelp: {
      this->PrintHelp();
      return 0;
    }
    }

    try {
      switch (param_.task) {
      case kTrain:
        CLITrain();
        break;
      case kDumpModel:
        CLIDumpModel();
        break;
      case kPredict:
        CLIPredict();
        break;
      }
    } catch (dmlc::Error const& e) {
      xgboost::CLIError(e);
      return 1;
    }
    return 0;
  }

  ~CLI() {
    collective::Finalize();
  }
};
}  // namespace xgboost

int main(int argc, char *argv[]) {
  try {
    xgboost::CLI cli(argc, argv);
    return cli.Run();
  } catch (dmlc::Error const& e) {
    // This captures only the initialization error.
    xgboost::CLIError(e);
    return 1;
  }
  return 0;
}
