#include "./helpers.h"

std::string TempFileName() {
  return std::tmpnam(nullptr);
}

bool FileExists(const std::string name) {
  struct stat st;
  return stat(name.c_str(), &st) == 0; 
}

long GetFileSize(const std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  return st.st_size;
}

std::string CreateSimpleTestData() {
  std::string tmp_file = TempFileName();
  std::ofstream fo;
  fo.open(tmp_file);
  fo << "0 0:0 1:10 2:20\n";
  fo << "1 0:0 3:30 4:40\n";
  fo.close();
  return tmp_file;
}

void CheckObjFunction(xgboost::ObjFunction * obj,
                      std::vector<xgboost::bst_float> preds,
                      std::vector<xgboost::bst_float> labels,
                      std::vector<xgboost::bst_float> weights,
                      std::vector<xgboost::bst_float> out_grad,
                      std::vector<xgboost::bst_float> out_hess) {
  xgboost::MetaInfo info;
  info.num_row = labels.size();
  info.labels = labels;
  info.weights = weights;

  std::vector<xgboost::bst_gpair> gpair;
  obj->GetGradient(preds, info, 1, &gpair);

  ASSERT_EQ(gpair.size(), preds.size());
  for (int i = 0; i < static_cast<int>(gpair.size()); ++i) {
    EXPECT_NEAR(gpair[i].grad, out_grad[i], 0.01)
      << "Unexpected grad for pred=" << preds[i] << " label=" << labels[i]
      << " weight=" << weights[i];
    EXPECT_NEAR(gpair[i].hess, out_hess[i], 0.01)
      << "Unexpected hess for pred=" << preds[i] << " label=" << labels[i]
      << " weight=" << weights[i];
  }
}

xgboost::bst_float GetMetricEval(xgboost::Metric * metric,
                                 std::vector<xgboost::bst_float> preds,
                                 std::vector<xgboost::bst_float> labels,
                                 std::vector<xgboost::bst_float> weights) {
  xgboost::MetaInfo info;
  info.num_row = labels.size();
  info.labels = labels;
  info.weights = weights;
  return metric->Eval(preds, info, false);
}
