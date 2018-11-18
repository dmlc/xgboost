// Copyright by Contributors
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <xgboost/learner.h>

#include <vector>
#include <fstream>

#include "helpers.h"
#include "../../src/common/json.h"

namespace xgboost {

TEST(learner, Test) {
  typedef std::pair<std::string, std::string> arg;
  auto args = {arg("tree_method", "exact")};
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));
  learner->Configure(args);

  delete mat_ptr;
}

TEST(learner, SelectTreeMethod) {
  using arg = std::pair<std::string, std::string>;
  auto mat_ptr = CreateDMatrix(10, 10, 0);
  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*mat_ptr};
  auto learner = std::unique_ptr<Learner>(Learner::Create(mat));

  // Test if `tree_method` can be set
  learner->Configure({arg("tree_method", "approx")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_histmaker,prune");
  learner->Configure({arg("tree_method", "exact")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_colmaker,prune");
  learner->Configure({arg("tree_method", "hist")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_quantile_histmaker");
#ifdef XGBOOST_USE_CUDA
  learner->Configure({arg("tree_method", "gpu_exact")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu,prune");
  learner->Configure({arg("tree_method", "gpu_hist")});
  ASSERT_EQ(learner->GetConfigurationArguments().at("updater"),
            "grow_gpu_hist");
#endif

  delete mat_ptr;
}

void TestModelIO(
    std::vector<std::pair<std::string, std::string>> const& args,
    std::string model_name) {
  auto pp_dmat = CreateDMatrix(10, 10, 0);

  HostDeviceVector<bst_float> labels {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  (*pp_dmat)->Info().labels_ = labels;

  std::vector<std::shared_ptr<xgboost::DMatrix>> mat = {*pp_dmat};
  dmlc::TemporaryDirectory tempdir;
  std::string tempdir_path = tempdir.path;

  std::string dump_path =
      tempdir_path + "/" + model_name + "_dump.json";
  std::string redump_path =
      tempdir_path + "/" + model_name+ "_redump.json";

  json::Json writer;
  {
    auto learner = std::unique_ptr<Learner>(Learner::Create(mat));

    learner->Configure(args);
    learner->InitModel();
    learner->UpdateOneIter(0, pp_dmat->get());

    learner->Save(&writer);

    std::ofstream fout(dump_path, std::ios_base::out);
    if (!fout) { LOG(FATAL) << "Failed to open file"; }
    json::Json::Dump(writer, &fout);
    fout.close();
  }

  json::Json load_back;
  {
    auto learner = std::unique_ptr<Learner>(Learner::Create(mat));

    std::ifstream fin(dump_path, std::ios_base::in);
    if (!fin) { LOG(FATAL) << "Failed to open file"; }
    // Load previously dumped model in json.
    json::Json reader = json::Json::Load(&fin);
    ASSERT_EQ(reader, writer);

    learner->Load(&reader);  // restore learner model.
    learner->InitModel();
    fin.close();

    std::ofstream fout(redump_path, std::ios_base::out);
    if (!fout) { LOG(FATAL) << "Failed to open file"; }
    json::Json reload_json;

    // Dump it out again
    learner->Save(&reload_json);
    json::Json::Dump(reload_json, &fout);
    fout.close();

    // Load the model dumped at second time
    fin.open(redump_path, std::ios_base::in);
    if (!fin) { LOG(FATAL) << "Failed to open file"; }
    load_back = json::Json::Load(&fin);
    fin.close();
  }

  std::stringstream ss;
  json::Json::Dump(writer, &ss);

  std::stringstream ss2;
  json::Json::Dump(load_back, &ss2);

  // Compare the difference between two dumps.
  ASSERT_EQ(ss.str(), ss2.str());
  ASSERT_EQ(writer, load_back);

  delete pp_dmat;
}

TEST(learner, TreeModelIO) {
  std::vector<std::pair<std::string, std::string>> gbtree_args =
      {{"tree_method", "hist"}};
  TestModelIO(gbtree_args, "tree");
}

TEST(learner, LinearModelIO) {
  std::vector<std::pair<std::string, std::string>> gblinear_args =
      {{"booster", "gblinear"}};
  TestModelIO(gblinear_args, "linear");
}

TEST(learner, DartModelIO) {
  std::vector<std::pair<std::string, std::string>> gbdark_args =
      {{"booster", "dart"}};
  TestModelIO(gbdark_args, "dart");
}
}  // namespace xgboost
