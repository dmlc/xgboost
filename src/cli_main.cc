/*!
 * Copyright 2014 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of xgboost.
 *  This file is not included in dynamic library.
 */
// Copyright 2014 by Contributors
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <dmlc/timer.h>
#include <iomanip>
#include <ctime>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>
#include "./common/sync.h"
#include "./common/config.h"


namespace xgboost {

enum CLITask {
  kTrain = 0,
  kDumpModel = 1,
  kPredict = 2
};

struct CLIParam : public dmlc::Parameter<CLIParam> {
  /*! \brief the task name */
  int task;
  /*! \brief whether silent */
  int silent;
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

  // declare parameters
  DMLC_DECLARE_PARAMETER(CLIParam) {
    // NOTE: declare everything except eval_data_paths.
    DMLC_DECLARE_FIELD(task).set_default(kTrain)
        .add_enum("train", kTrain)
        .add_enum("dump", kDumpModel)
        .add_enum("pred", kPredict)
        .describe("Task to be performed by the CLI program.");
    DMLC_DECLARE_FIELD(silent).set_default(0).set_range(0, 2)
        .describe("Silent level during the task.");
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
        .add_enum("auto", 0)
        .add_enum("col", 1)
        .add_enum("row", 2)
        .describe("Data split mode.");
    DMLC_DECLARE_FIELD(ntree_limit).set_default(0).set_lower_bound(0)
        .describe("Number of trees used for prediction, 0 means use all trees.");
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
  inline void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) {
    this->cfg = cfg;
    this->InitAllowUnknown(cfg);
    for (const auto& kv : cfg) {
      if (!strncmp("eval[", kv.first.c_str(), 5)) {
        char evname[256];
        CHECK_EQ(sscanf(kv.first.c_str(), "eval[%[^]]", evname), 1)
            << "must specify evaluation name for display";
        eval_data_names.emplace_back(evname);
        eval_data_paths.push_back(kv.second);
      }
    }
    // constraint.
    if (name_pred == "stdout") {
      save_period = 0;
      silent = 1;
    }
    if (dsplit == 0 && rabit::IsDistributed()) {
      dsplit = 2;
    }
    if (rabit::GetRank() != 0) {
      silent = 2;
    }
  }
};

DMLC_REGISTER_PARAMETER(CLIParam);

void CLITrain(const CLIParam& param) {
  const double tstart_data_load = dmlc::GetTime();
  if (rabit::IsDistributed()) {
    std::string pname = rabit::GetProcessorName();
    LOG(CONSOLE) << "start " << pname << ":" << rabit::GetRank();
  }
  // load in data.
  std::shared_ptr<DMatrix> dtrain(
      DMatrix::Load(param.train_path, param.silent != 0, param.dsplit == 2));
  std::vector<std::shared_ptr<DMatrix> > deval;
  std::vector<std::shared_ptr<DMatrix> > cache_mats;
  std::vector<DMatrix*> eval_datasets;
  cache_mats.push_back(dtrain);
  for (size_t i = 0; i < param.eval_data_names.size(); ++i) {
    deval.emplace_back(
        std::shared_ptr<DMatrix>(DMatrix::Load(param.eval_data_paths[i],
                                               param.silent != 0, param.dsplit == 2)));
    eval_datasets.push_back(deval.back().get());
    cache_mats.push_back(deval.back());
  }
  std::vector<std::string> eval_data_names = param.eval_data_names;
  if (param.eval_train) {
    eval_datasets.push_back(dtrain.get());
    eval_data_names.emplace_back("train");
  }
  // initialize the learner.
  std::unique_ptr<Learner> learner(Learner::Create(cache_mats));
  int version = rabit::LoadCheckPoint(learner.get());
  if (version == 0) {
    // initialize the model if needed.
    if (param.model_in != "NULL") {
      std::unique_ptr<dmlc::Stream> fi(
          dmlc::Stream::Create(param.model_in.c_str(), "r"));
      learner->Load(fi.get());
      learner->Configure(param.cfg);
    } else {
      learner->Configure(param.cfg);
      learner->InitModel();
    }
  }
  if (param.silent == 0) {
    LOG(INFO) << "Loading data: " << dmlc::GetTime() - tstart_data_load << " sec";
  }
  // start training.
  const double start = dmlc::GetTime();
  for (int i = version / 2; i < param.num_round; ++i) {
    double elapsed = dmlc::GetTime() - start;
    if (version % 2 == 0) {
      if (param.silent == 0) {
        LOG(CONSOLE) << "boosting round " << i << ", " << elapsed << " sec elapsed";
      }
      learner->UpdateOneIter(i, dtrain.get());
      if (learner->AllowLazyCheckPoint()) {
        rabit::LazyCheckPoint(learner.get());
      } else {
        rabit::CheckPoint(learner.get());
      }
      version += 1;
    }
    CHECK_EQ(version, rabit::VersionNumber());
    std::string res = learner->EvalOneIter(i, eval_datasets, eval_data_names);
    if (rabit::IsDistributed()) {
      if (rabit::GetRank() == 0) {
        LOG(TRACKER) << res;
      }
    } else {
      if (param.silent < 2) {
        LOG(CONSOLE) << res;
      }
    }
    if (param.save_period != 0 &&
        (i + 1) % param.save_period == 0 &&
        rabit::GetRank() == 0) {
      std::ostringstream os;
      os << param.model_dir << '/'
         << std::setfill('0') << std::setw(4)
         << i + 1 << ".model";
      std::unique_ptr<dmlc::Stream> fo(
          dmlc::Stream::Create(os.str().c_str(), "w"));
      learner->Save(fo.get());
    }

    if (learner->AllowLazyCheckPoint()) {
      rabit::LazyCheckPoint(learner.get());
    } else {
      rabit::CheckPoint(learner.get());
    }
    version += 1;
    CHECK_EQ(version, rabit::VersionNumber());
  }
  // always save final round
  if ((param.save_period == 0 || param.num_round % param.save_period != 0) &&
      param.model_out != "NONE" &&
      rabit::GetRank() == 0) {
    std::ostringstream os;
    if (param.model_out == "NULL") {
      os << param.model_dir << '/'
         << std::setfill('0') << std::setw(4)
         << param.num_round << ".model";
    } else {
      os << param.model_out;
    }
    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(os.str().c_str(), "w"));
    learner->Save(fo.get());
  }

  if (param.silent == 0) {
    double elapsed = dmlc::GetTime() - start;
    LOG(CONSOLE) << "update end, " << elapsed << " sec in all";
  }
}

void CLIDumpModel(const CLIParam& param) {
  FeatureMap fmap;
  if (param.name_fmap != "NULL") {
    std::unique_ptr<dmlc::Stream> fs(
        dmlc::Stream::Create(param.name_fmap.c_str(), "r"));
    dmlc::istream is(fs.get());
    fmap.LoadText(is);
  }
  // load model
  CHECK_NE(param.model_in, "NULL")
      << "Must specify model_in for dump";
  std::unique_ptr<Learner> learner(Learner::Create({}));
  std::unique_ptr<dmlc::Stream> fi(
      dmlc::Stream::Create(param.model_in.c_str(), "r"));
  learner->Configure(param.cfg);
  learner->Load(fi.get());
  // dump data
  std::vector<std::string> dump = learner->DumpModel(
      fmap, param.dump_stats, param.dump_format);
  std::unique_ptr<dmlc::Stream> fo(
      dmlc::Stream::Create(param.name_dump.c_str(), "w"));
  dmlc::ostream os(fo.get());
  if (param.dump_format == "json") {
    os << "[" << std::endl;
    for (size_t i = 0; i < dump.size(); ++i) {
      if (i != 0) os << "," << std::endl;
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

void CLIPredict(const CLIParam& param) {
  CHECK_NE(param.test_path, "NULL")
      << "Test dataset parameter test:data must be specified.";
  // load data
  std::unique_ptr<DMatrix> dtest(
      DMatrix::Load(param.test_path, param.silent != 0, param.dsplit == 2));
  // load model
  CHECK_NE(param.model_in, "NULL")
      << "Must specify model_in for predict";
  std::unique_ptr<Learner> learner(Learner::Create({}));
  std::unique_ptr<dmlc::Stream> fi(
      dmlc::Stream::Create(param.model_in.c_str(), "r"));
  learner->Load(fi.get());
  learner->Configure(param.cfg);

  if (param.silent == 0) {
    LOG(CONSOLE) << "start prediction...";
  }
  HostDeviceVector<bst_float> preds;
  learner->Predict(dtest.get(), param.pred_margin, &preds, param.ntree_limit);
  if (param.silent == 0) {
    LOG(CONSOLE) << "writing prediction to " << param.name_pred;
  }
  std::unique_ptr<dmlc::Stream> fo(
      dmlc::Stream::Create(param.name_pred.c_str(), "w"));
  dmlc::ostream os(fo.get());
  for (bst_float p : preds.HostVector()) {
    os << std::setprecision(std::numeric_limits<bst_float>::max_digits10 + 2)
       << p << '\n';
  }
  // force flush before fo destruct.
  os.set_stream(nullptr);
}

int CLIRunTask(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: <config>\n");
    return 0;
  }
  rabit::Init(argc, argv);

  std::vector<std::pair<std::string, std::string> > cfg;
  cfg.emplace_back("seed", "0");

  common::ConfigIterator itr(argv[1]);
  while (itr.Next()) {
    cfg.emplace_back(std::string(itr.Name()), std::string(itr.Val()));
  }

  for (int i = 2; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      cfg.emplace_back(std::string(name), std::string(val));
    }
  }
  CLIParam param;
  param.Configure(cfg);

  switch (param.task) {
    case kTrain: CLITrain(param); break;
    case kDumpModel: CLIDumpModel(param); break;
    case kPredict: CLIPredict(param); break;
  }
  rabit::Finalize();
  return 0;
}
}  // namespace xgboost

int main(int argc, char *argv[]) {
  return xgboost::CLIRunTask(argc, argv);
}
