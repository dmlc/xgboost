/**
 * Copyright 2019-2023 by XGBoost Contributors
 */
#include <xgboost/data.h>  // for DMatrix

#include "../../../src/common/compressed_iterator.h"
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/sparse_page_dmatrix.h"
#include "../../../src/tree/param.h"  // TrainParam
#include "../filesystem.h"            // dmlc::TemporaryDirectory
#include "../helpers.h"

namespace xgboost {

TEST(SparsePageDMatrix, EllpackPage) {
  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  DMatrix* dmat = DMatrix::Load(tmp_file + "?format=libsvm" + "#" + tmp_file + ".cache");

  // Loop over the batches and assert the data is as expected
  size_t n = 0;
  for (const auto& batch : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    n += batch.Size();
  }
  EXPECT_EQ(n, dmat->Info().num_row_);

  auto path =
      data::MakeId(tmp_file + ".cache",
                   dynamic_cast<data::SparsePageDMatrix *>(dmat)) +
      ".row.page";
  EXPECT_TRUE(FileExists(path));
  path =
      data::MakeId(tmp_file + ".cache",
                   dynamic_cast<data::SparsePageDMatrix *>(dmat)) +
      ".ellpack.page";
  EXPECT_TRUE(FileExists(path));

  delete dmat;
}

TEST(SparsePageDMatrix, MultipleEllpackPages) {
  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  size_t constexpr kPageSize = 64, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(kEntries, filename);

  // Loop over the batches and count the records
  int64_t batch_count = 0;
  int64_t row_count = 0;
  for (const auto& batch : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    EXPECT_LT(batch.Size(), dmat->Info().num_row_);
    batch_count++;
    row_count += batch.Size();
  }
  EXPECT_GE(batch_count, 2);
  EXPECT_EQ(row_count, dmat->Info().num_row_);

  auto path =
      data::MakeId(filename,
                   dynamic_cast<data::SparsePageDMatrix *>(dmat.get())) +
      ".ellpack.page";
}

TEST(SparsePageDMatrix, RetainEllpackPage) {
  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{32, tree::TrainParam::DftSparseThreshold()};
  auto m = CreateSparsePageDMatrix(10000);

  auto batches = m->GetBatches<EllpackPage>(&ctx, param);
  auto begin = batches.begin();
  auto end = batches.end();

  std::vector<HostDeviceVector<common::CompressedByteT>> gidx_buffers;
  std::vector<std::shared_ptr<EllpackPage const>> iterators;
  for (auto it = begin; it != end; ++it) {
    iterators.push_back(it.Page());
    gidx_buffers.emplace_back();
    gidx_buffers.back().Resize((*it).Impl()->gidx_buffer.Size());
    gidx_buffers.back().Copy((*it).Impl()->gidx_buffer);
  }
  ASSERT_GE(iterators.size(), 2);

  for (size_t i = 0; i < iterators.size(); ++i) {
    ASSERT_EQ((*iterators[i]).Impl()->gidx_buffer.HostVector(), gidx_buffers.at(i).HostVector());
    if (i != iterators.size() - 1) {
      ASSERT_EQ(iterators[i].use_count(), 1);
    } else {
      // The last batch is still being held by sparse page DMatrix.
      ASSERT_EQ(iterators[i].use_count(), 2);
    }
  }

  // make sure it's const and the caller can not modify the content of page.
  for (auto& page : m->GetBatches<EllpackPage>(&ctx, param)) {
    static_assert(std::is_const<std::remove_reference_t<decltype(page)>>::value);
  }

  // The above iteration clears out all references inside DMatrix.
  for (auto const& ptr : iterators) {
    ASSERT_TRUE(ptr.unique());
  }
}

TEST(SparsePageDMatrix, EllpackPageContent) {
  auto ctx = CreateEmptyGenericParam(0);
  constexpr size_t kRows = 6;
  constexpr size_t kCols = 2;
  constexpr size_t kPageSize = 1;

  // Create an in-memory DMatrix.
  std::unique_ptr<DMatrix> dmat(CreateSparsePageDMatrixWithRC(kRows, kCols, 0, true));

  // Create a DMatrix with multiple batches.
  dmlc::TemporaryDirectory tmpdir;
  std::unique_ptr<DMatrix>
      dmat_ext(CreateSparsePageDMatrixWithRC(kRows, kCols, kPageSize, true, tmpdir));

  auto param = BatchParam{2, tree::TrainParam::DftSparseThreshold()};
  auto impl = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();
  EXPECT_EQ(impl->base_rowid, 0);
  EXPECT_EQ(impl->n_rows, kRows);
  EXPECT_FALSE(impl->is_dense);
  EXPECT_EQ(impl->row_stride, 2);
  EXPECT_EQ(impl->Cuts().TotalBins(), 4);

  std::unique_ptr<EllpackPageImpl> impl_ext;
  size_t offset = 0;
  for (auto& batch : dmat_ext->GetBatches<EllpackPage>(&ctx, param)) {
    if (!impl_ext) {
      impl_ext.reset(new EllpackPageImpl(
          batch.Impl()->gidx_buffer.DeviceIdx(), batch.Impl()->Cuts(),
          batch.Impl()->is_dense, batch.Impl()->row_stride, kRows));
    }
    auto n_elems = impl_ext->Copy(0, batch.Impl(), offset);
    offset += n_elems;
  }
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

  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{kMaxBins, tree::TrainParam::DftSparseThreshold()};
  auto impl = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();
  EXPECT_EQ(impl->base_rowid, 0);
  EXPECT_EQ(impl->n_rows, kRows);

  size_t current_row = 0;
  thrust::device_vector<bst_float> row_d(kCols);
  thrust::device_vector<bst_float> row_ext_d(kCols);
  std::vector<bst_float> row(kCols);
  std::vector<bst_float> row_ext(kCols);
  for (auto& page : dmat_ext->GetBatches<EllpackPage>(&ctx, param)) {
    auto impl_ext = page.Impl();
    EXPECT_EQ(impl_ext->base_rowid, current_row);

    for (size_t i = 0; i < impl_ext->Size(); i++) {
      dh::LaunchN(kCols, ReadRowFunction(impl->GetDeviceAccessor(0), current_row, row_d.data().get()));
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(kCols, ReadRowFunction(impl_ext->GetDeviceAccessor(0), current_row, row_ext_d.data().get()));
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

  Context ctx{MakeCUDACtx(0)};
  auto param = BatchParam{kMaxBins, tree::TrainParam::DftSparseThreshold()};

  size_t current_row = 0;
  for (auto& page : dmat_ext->GetBatches<EllpackPage>(&ctx, param)) {
    auto impl_ext = page.Impl();
    EXPECT_EQ(impl_ext->base_rowid, current_row);
    current_row += impl_ext->n_rows;
  }
}

}  // namespace xgboost
