#include <xgboost/c_api.h>

#include "helpers.h"
#include "../../src/data/device_adapter.cuh"
#include "../../src/data/iterative_device_dmatrix.h"

namespace xgboost {

CudaArrayIterForTest::CudaArrayIterForTest(float sparsity, size_t rows,
                                           size_t cols, size_t batches)
    : ArrayIterForTest{sparsity, rows, cols, batches} {
  rng_->Device(0);
  std::tie(batches_, interface_) =
      rng_->GenerateArrayInterfaceBatch(&data_, n_batches_);
  this->Reset();
}

size_t constexpr CudaArrayIterForTest::kRows;
size_t constexpr CudaArrayIterForTest::kCols;
size_t constexpr CudaArrayIterForTest::kBatches;

int CudaArrayIterForTest::Next() {
  if (iter_ == n_batches_) {
    return 0;
  }
  XGProxyDMatrixSetDataCudaArrayInterface(proxy_, batches_[iter_].c_str());
  iter_++;
  return 1;
}


std::shared_ptr<DMatrix> RandomDataGenerator::GenerateDeviceDMatrix(bool with_label,
                                                                    bool float_label,
                                                                    size_t classes) {
  CudaArrayIterForTest iter{this->sparsity_, this->rows_, this->cols_, 1};
  auto m = std::make_shared<data::IterativeDeviceDMatrix>(
      &iter, iter.Proxy(), Reset, Next, std::numeric_limits<float>::quiet_NaN(),
      0, bins_);
  return m;
}
}  // namespace xgboost
