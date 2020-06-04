#include <gtest/gtest.h>
#include <algorithm>

#include "helpers.h"
#include "../../src/data/array_interface.h"
namespace xgboost {

TEST(RandomDataGenerator, DMatrix) {
  size_t constexpr kRows { 16 }, kCols { 32 };
  float constexpr kSparsity { 0.4f };
  auto p_dmatrix = RandomDataGenerator{kRows, kCols, kSparsity}.GenerateDMatrix();

  HostDeviceVector<float> csr_value;
  HostDeviceVector<bst_row_t> csr_rptr;
  HostDeviceVector<bst_feature_t> csr_cidx;
  RandomDataGenerator{kRows, kCols, kSparsity}.GenerateCSR(&csr_value, &csr_rptr, &csr_cidx);

  HostDeviceVector<float> dense_data;
  RandomDataGenerator{kRows, kCols, kSparsity}.GenerateDense(&dense_data);

  auto it = std::copy_if(
      dense_data.HostVector().begin(), dense_data.HostVector().end(),
      dense_data.HostVector().begin(), [](float v) { return !std::isnan(v); });

  CHECK_EQ(p_dmatrix->Info().num_row_, kRows);
  CHECK_EQ(p_dmatrix->Info().num_col_, kCols);

  for (auto const& page : p_dmatrix->GetBatches<SparsePage>()) {
    size_t n_elements = page.data.Size();
    CHECK_EQ(n_elements, it - dense_data.HostVector().begin());
    CHECK_EQ(n_elements, csr_value.Size());

    for (size_t i = 0; i < n_elements; ++i) {
      CHECK_EQ(dense_data.HostVector()[i], csr_value.HostVector()[i]);
      CHECK_EQ(dense_data.HostVector()[i], page.data.HostVector()[i].fvalue);
      CHECK_EQ(page.data.HostVector()[i].index, csr_cidx.HostVector()[i]);
    }
    CHECK_EQ(page.offset.Size(), csr_rptr.Size());
    for (size_t i = 0; i < p_dmatrix->Info().num_row_; ++i) {
      CHECK_EQ(page.offset.HostVector()[i], csr_rptr.HostVector()[i]);
    }
  }
}

TEST(RandomDataGenerator, GenerateArrayInterfaceBatch) {
  size_t constexpr kRows { 937 }, kCols { 100 }, kBatches { 13 };
  float constexpr kSparsity { 0.4f };

  HostDeviceVector<float> storage;
  std::string array;
  std::vector<std::string> batches;
  std::tie(batches, array) =
      RandomDataGenerator{kRows, kCols, kSparsity}.GenerateArrayInterfaceBatch(
          &storage, kBatches);
  CHECK_EQ(batches.size(), kBatches);

  size_t rows = 0;
  for (auto const &interface_str : batches) {
    Json j_interface =
        Json::Load({interface_str.c_str(), interface_str.size()});
    ArrayInterfaceHandler::Validate(get<Object const>(j_interface));
    CHECK_EQ(get<Integer>(j_interface["shape"][1]), kCols);
    rows += get<Integer>(j_interface["shape"][0]);
  }
  CHECK_EQ(rows, kRows);
  auto j_array = Json::Load({array.c_str(), array.size()});
  CHECK_EQ(get<Integer>(j_array["shape"][0]), kRows);
  CHECK_EQ(get<Integer>(j_array["shape"][1]), kCols);
}
}  // namespace xgboost
