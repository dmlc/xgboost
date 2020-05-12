// Copyright by Contributors

#include <dmlc/filesystem.h>
#include "../helpers.h"
#include "../../../src/common/compressed_iterator.h"
#include "../../../src/data/ellpack_page.cuh"

namespace xgboost {

TEST(SparsePageDMatrix, EllpackPage) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  DMatrix* dmat = DMatrix::Load(tmp_file + "#" + tmp_file + ".cache", true, false);

  // Loop over the batches and assert the data is as expected
  for (const auto& batch : dmat->GetBatches<EllpackPage>({0, 256, 64})) {
    EXPECT_EQ(batch.Size(), dmat->Info().num_row_);
  }

  EXPECT_TRUE(FileExists(tmp_file + ".cache"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.ellpack.page"));

  delete dmat;
}

TEST(SparsePageDMatrix, MultipleEllpackPages) {
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(12, 64, filename);

  // Loop over the batches and count the records
  int64_t batch_count = 0;
  int64_t row_count = 0;
  for (const auto& batch : dmat->GetBatches<EllpackPage>({0, 256, 7UL})) {
    EXPECT_LT(batch.Size(), dmat->Info().num_row_);
    batch_count++;
    row_count += batch.Size();
  }
  EXPECT_GE(batch_count, 2);
  EXPECT_EQ(row_count, dmat->Info().num_row_);

  EXPECT_TRUE(FileExists(filename + ".cache.ellpack.page"));
}

TEST(SparsePageDMatrix, EllpackPageContent) {
  constexpr size_t kRows = 6;
  constexpr size_t kCols = 2;
  constexpr size_t kPageSize = 1;

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat_ext(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));

  BatchParam param{0, 2, 0};
  auto impl = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();
  EXPECT_EQ(impl->base_rowid, 0);
  EXPECT_EQ(impl->n_rows, kRows);
  EXPECT_FALSE(impl->is_dense);
  EXPECT_EQ(impl->row_stride, 2);
  EXPECT_EQ(impl->Cuts().TotalBins(), 4);

  auto impl_ext = (*dmat_ext->GetBatches<EllpackPage>(param).begin()).Impl();
  EXPECT_EQ(impl_ext->base_rowid, 0);
  EXPECT_EQ(impl_ext->n_rows, kRows);
  EXPECT_FALSE(impl_ext->is_dense);
  EXPECT_EQ(impl_ext->row_stride, 2);
  EXPECT_EQ(impl_ext->Cuts().TotalBins(), 4);

  std::vector<common::CompressedByteT> buffer(impl->gidx_buffer.HostVector());
  std::vector<common::CompressedByteT> buffer_ext(impl_ext->gidx_buffer.HostVector());
  EXPECT_EQ(buffer, buffer_ext);
}

struct ReadRowFunction {
  EllpackDeviceAccessor matrix;
  int row;
  bst_float* row_data_d;
  ReadRowFunction(EllpackDeviceAccessor matrix, int row, bst_float* row_data_d)
      : matrix(std::move(matrix)), row(row), row_data_d(row_data_d) {}

  __device__ void operator()(size_t col) {
    auto value = matrix.GetFvalue(row, col);
    if (isnan(value)) {
      value = -1;
    }
    row_data_d[col] = value;
  }
};

TEST(SparsePageDMatrix, MultipleEllpackPageContent) {
  constexpr size_t kRows = 6;
  constexpr size_t kCols = 2;
  constexpr int kMaxBins = 256;
  constexpr size_t kPageSize = 1;

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat_ext(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));

  BatchParam param{0, kMaxBins, kPageSize};
  auto impl = (*dmat->GetBatches<EllpackPage>(param).begin()).Impl();
  EXPECT_EQ(impl->base_rowid, 0);
  EXPECT_EQ(impl->n_rows, kRows);

  size_t current_row = 0;
  thrust::device_vector<bst_float> row_d(kCols);
  thrust::device_vector<bst_float> row_ext_d(kCols);
  std::vector<bst_float> row(kCols);
  std::vector<bst_float> row_ext(kCols);
  for (auto& page : dmat_ext->GetBatches<EllpackPage>(param)) {
    auto impl_ext = page.Impl();
    EXPECT_EQ(impl_ext->base_rowid, current_row);

    for (size_t i = 0; i < impl_ext->Size(); i++) {
      dh::LaunchN(0, kCols, ReadRowFunction(impl->GetDeviceAccessor(0), current_row, row_d.data().get()));
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(0, kCols, ReadRowFunction(impl_ext->GetDeviceAccessor(0), current_row, row_ext_d.data().get()));
      thrust::copy(row_ext_d.begin(), row_ext_d.end(), row_ext.begin());

      EXPECT_EQ(row, row_ext);
      current_row++;
    }
  }
}

TEST(SparsePageDMatrix, EllpackPageMultipleLoops) {
  constexpr size_t kRows = 1024;
  constexpr size_t kCols = 16;
  constexpr int kMaxBins = 256;
  constexpr size_t kPageSize = 4096;

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat_ext(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));

  BatchParam param{0, kMaxBins, kPageSize};

  size_t current_row = 0;
  for (auto& page : dmat_ext->GetBatches<EllpackPage>(param)) {
    auto impl_ext = page.Impl();
    EXPECT_EQ(impl_ext->base_rowid, current_row);
    current_row += impl_ext->n_rows;
  }
}

}  // namespace xgboost
