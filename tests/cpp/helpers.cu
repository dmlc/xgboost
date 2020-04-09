#include "helpers.h"
#include "../../src/data/device_adapter.cuh"
#include "../../src/data/device_dmatrix.h"

namespace xgboost {
std::shared_ptr<DMatrix> RandomDataGenerator::GenerateDeviceDMatrix(bool with_label,
                                                                    bool float_label,
                                                                    size_t classes) {
  std::vector<HostDeviceVector<float>> storage(cols_);
  std::string arr = this->GenerateColumnarArrayInterface(&storage);
  auto adapter = data::CudfAdapter(arr);
  std::shared_ptr<DMatrix> m {
    new data::DeviceDMatrix{&adapter,
          std::numeric_limits<float>::quiet_NaN(), 1, 256}};
  return m;
}
}  // namespace xgboost
