#include <gtest/gtest.h>
#include <xgboost/linalg.h>
#include <numeric>

namespace xgboost {

auto MakeMatrixFromTest(HostDeviceVector<float> *storage, size_t n_rows, size_t n_cols) {
  storage->Resize(n_rows * n_cols);
  auto& h_storage = storage->HostVector();

  std::iota(h_storage.begin(), h_storage.end(), 0);

  auto m = MatrixView<float>{storage, {n_cols, 1}, {n_rows, n_cols}, -1};
  return m;

}

TEST(Linalg, Matrix) {
  size_t kRows = 31, kCols = 77;
  HostDeviceVector<float> storage;
  auto m = MakeMatrixFromTest(&storage, kRows, kCols);
  ASSERT_EQ(m.DeviceIdx(), GenericParameter::kCpuId);
  ASSERT_EQ(m(0, 0), 0);
  ASSERT_EQ(m(kRows - 1, kCols - 1), storage.Size() - 1);
}

TEST(Linalg, Vector) {
  size_t kRows = 31, kCols = 77;
  HostDeviceVector<float> storage;
  auto m = MakeMatrixFromTest(&storage, kRows, kCols);
  auto v = VectorView<float>(m, 3);
  for (size_t i = 0; i < v.Size(); ++i) {
    ASSERT_EQ(v[i], m(i, 3));
  }

  ASSERT_EQ(v[0], 3);
}
} // namespace xgboost
