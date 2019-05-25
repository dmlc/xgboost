#include <xgboost/c_api.h>
#include <xgboost/logging.h>
#include <dmlc/parameter.h>
#include "../../src/common/common.h"
#include "../../src/common/feature_bundling.h"
#include "../../src/common/hist_util.h"
#include "../../src/common/timer.h"
#include "../cpp/helpers.h"

namespace xgboost {

struct BenchmarkParameter : public dmlc::Parameter<BenchmarkParameter> {
  int32_t n_rows;
  int32_t n_cols;
  float sparsity;
  int32_t bins;
  std::string data_path;
  DMLC_DECLARE_PARAMETER(BenchmarkParameter) {
    DMLC_DECLARE_FIELD(n_rows)
        .set_default(100);
    DMLC_DECLARE_FIELD(n_cols)
        .set_default(100);
    DMLC_DECLARE_FIELD(sparsity)
        .set_default(0.7);
    DMLC_DECLARE_FIELD(bins)
        .set_default(100);
    DMLC_DECLARE_FIELD(data_path)
        .set_default("");
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
    DMatrixHandle dmat[1];
    std::string path {param.data_path};
    XGDMatrixCreateFromFile(path.c_str(), 0, &dmat[0]);
    auto pp_dmat = static_cast<std::shared_ptr<DMatrix>*>(dmat[0]);
    monitor_.Stop("Create dmatrix");

    index_matrix_.Init((*pp_dmat).get(), param.bins);
    column_matrix_.Init(index_matrix_, kSparsityThreshold);
    std::vector<std::pair<std::string, std::string>> args;
    tparam_.InitAllowUnknown(args);

    // common::GHistIndexBlockMatrix block_matrix_old;
    // {
    //   LOG(INFO) << "Init";
    //   monitor_.Start("block matrix initialization");
    //   block_matrix_old.Init(index_matrix_, column_matrix_, tparam_);
    //   monitor_.Stop("block matrix initialization");
    // }

    common::GHistIndexBlockMatrix block_matrix_new;
    {
      LOG(INFO) << "Build";
      monitor_.Start("block matrix Build");
      block_matrix_new.Build(index_matrix_, column_matrix_, tparam_);
      monitor_.Stop("block matrix Build");
    }

    // check(block_matrix_old, block_matrix_new);
  }

  void check(common::GHistIndexBlockMatrix const& block_matrix_old,
             common::GHistIndexBlockMatrix const& block_matrix_new) {
    CHECK_EQ(block_matrix_old.index_.size(), block_matrix_new.index_.size());
    CHECK_EQ(block_matrix_old.row_ptr_.size(), block_matrix_new.row_ptr_.size());
    CHECK_EQ(block_matrix_old.blocks_.size(), block_matrix_new.blocks_.size());

    for (size_t i = 0; i < block_matrix_new.index_.size(); ++i) {
      CHECK_EQ(block_matrix_old.index_[i],
               block_matrix_new.index_[i]) << " i: " << i << ", "
                                           << "size: " << block_matrix_new.index_.size();
    }

    for (size_t i = 0; i < block_matrix_old.row_ptr_.size(); ++i) {
      CHECK_EQ(block_matrix_old.row_ptr_[i],
               block_matrix_new.row_ptr_[i]) << " i: " << i << ", "
                                             << "size: " << block_matrix_new.row_ptr_.size();
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
