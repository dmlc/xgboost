#include <dmlc/parameter.h>
#include <dmlc/registry.h>

#include <xgboost/c_api.h>
#include <xgboost/logging.h>

#include "../../src/common/common.h"
#include "../../src/common/column_matrix.h"
#include "../../src/common/hist_util.h"
#include "../../src/common/timer.h"

namespace xgboost {

struct BenchmarkParameter : public dmlc::Parameter<BenchmarkParameter> {
  int32_t bins;
  std::string data_path;

  DMLC_DECLARE_PARAMETER(BenchmarkParameter) {
    DMLC_DECLARE_FIELD(bins)
        .set_default(100)
        .describe("Maximum number of bins.");
    DMLC_DECLARE_FIELD(data_path)
        .set_default("")
        .describe("Path to dataset for benchmarking.");
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

    common::GHistIndexBlockMatrix block_matrix_new;
    {
      LOG(INFO) << "Build";
      monitor_.Start("block matrix Initialization");
      block_matrix_new.Init(index_matrix_, column_matrix_, tparam_);
      monitor_.Stop("block matrix Initialization");
    }
  }
};

constexpr float Benchmark::kSparsityThreshold;

}  // namespace xgboost

void Help(xgboost::BenchmarkParameter const& param) {
  std::cerr << "Usage: [OPTION]=[ARGUMENT]\n\n";
  for (auto const& field :   param.__FIELDS__()) {
    std::cerr << "\t" << field.name << ":\t" << field.description << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char const* argv[]) {
  std::vector<std::pair<std::string, std::string>> args {
    {"verbosity", "3"}};
  xgboost::ConsoleLogger::Configure(args.begin(), args.end());
  args.clear();
  omp_set_num_threads(1);

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

  if (param.data_path.length() == 0) {
    std::cerr << "Please provide path to dataset.\n" << std::endl;
    Help(param);
    return 1;
  }

  xgboost::Benchmark bm;
  bm.Run(param);
  return 0;
}
