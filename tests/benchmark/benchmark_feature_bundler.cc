#include <xgboost/c_api.h>
#include <xgboost/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/filesystem.h>
#include "../../src/common/common.h"
#include "../../src/common/feature_bundling.h"
#include "../../src/common/hist_util.h"
#include "../../src/common/timer.h"
#include "../cpp/helpers.h"

namespace xgboost {

std::shared_ptr<xgboost::DMatrix>* createDMatrix(int rows, int columns,
                                                 float sparsity, int seed) {
  const float missing_value = -1;
  std::vector<float> test_data(rows * columns);

  std::random_device dev;
  std::uniform_real_distribution<> dist;

#pragma omp parallel for
  for (size_t i = 0; i < test_data.size(); ++i) {
    auto& e = test_data[i];
    if (dist(dev) < sparsity) {
      e = missing_value;
    } else {
      e = dist(dev);
    }
  }

  DMatrixHandle handle;
  auto nt = omp_get_max_threads();
  LOG(INFO) << "nt: " << nt;
  XGDMatrixCreateFromMat_omp(test_data.data(), rows, columns, missing_value,
                             &handle, 8);
  return static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
}

struct BenchmarkParameter : public dmlc::Parameter<BenchmarkParameter> {
  int32_t n_rows;
  int32_t n_cols;
  float sparsity;
  int32_t bins;
  DMLC_DECLARE_PARAMETER(BenchmarkParameter) {
    DMLC_DECLARE_FIELD(n_rows)
        .set_default(100);
    DMLC_DECLARE_FIELD(n_cols)
        .set_default(100);
    DMLC_DECLARE_FIELD(sparsity)
        .set_default(0.7);
    DMLC_DECLARE_FIELD(bins)
        .set_default(100);
  }
};

DMLC_REGISTER_PARAMETER(BenchmarkParameter);

class Benchmark {
  common::ColumnMatrix column_matrix_;
  common::GHistIndexMatrix index_matrix_;
  static float constexpr kSparsityThreshold = 0.2;

  tree::TrainParam tparam_;
  common::Monitor monitor_;

 public:
  Benchmark() {
    monitor_.Init("Benchmark");
  }

  void Run(BenchmarkParameter const& param) {
    monitor_.Start("Create dmatrix");
    dmlc::TemporaryDirectory tempdir;
    const std::string tmp_file = tempdir.path + "/simple.libsvm";
    LOG(INFO) << "file name: " << tmp_file;
    // CreateBigTestData(tmp_file, param.n_rows * param.n_cols, param.n_cols);
    // auto pp_dmat = createDMatrix(param.n_rows, param.n_cols, param.sparsity, 0);
    DMatrixHandle dmat[1];
    std::string path {"/home/fis/Others/datasets/resources/HIGGS/HIGGS"};
    XGDMatrixCreateFromFile(path.c_str(), 0, &dmat[0]);
    auto pp_dmat = static_cast<std::shared_ptr<DMatrix>*>(dmat[0]);
    monitor_.Stop("Create dmatrix");

    index_matrix_.Init((*pp_dmat).get(), param.bins);
    column_matrix_.Init(index_matrix_, kSparsityThreshold);
    std::vector<std::pair<std::string, std::string>> args;
    tparam_.InitAllowUnknown(args);

    // {
    //   LOG(INFO) << "Init";
    //   monitor_.Start("block matrix initialization");
    //   common::GHistIndexBlockMatrix block_matrix_;
    //   block_matrix_.Init(index_matrix_, column_matrix_, tparam_);
    //   monitor_.Stop("block matrix initialization");
    // }

    {
      LOG(INFO) << "Build";
      common::GHistIndexBlockMatrix block_matrix_;
      monitor_.Start("block matrix Build");
      block_matrix_.Build(index_matrix_, column_matrix_, tparam_);
      monitor_.Stop("block matrix Build");
    }
  }
};

constexpr float Benchmark::kSparsityThreshold;

}  // namespace xgboost

int main(int argc, char const* argv[]) {
  std::vector<std::pair<std::string, std::string>> args {
    {"verbosity", "3"}};
  xgboost::ConsoleLogger::Configure(args.begin(), args.end());
  args.clear();
  omp_set_num_threads(4);

  for (size_t i = 1; i < argc; i++) {
    std::pair<std::string, std::string> arg;
    std::string kv_str {argv[i]};
    auto kv = xgboost::common::Split(kv_str, '=');
    arg.first = kv.at(0);
    arg.second = kv.at(1);
    args.emplace_back(arg);
  }

  xgboost::BenchmarkParameter param;
  param.Init(args);
  xgboost::Benchmark bm;
  bm.Run(param);
  return 0;
}
