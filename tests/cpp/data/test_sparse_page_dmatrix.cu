/**
 * Copyright 2019-2024, XGBoost Contributors
 */
#include <xgboost/data.h>  // for DMatrix

#include "../../../src/common/compressed_iterator.h"
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/ellpack_page.h"
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
      data::MakeId(tmp_file + ".cache", dynamic_cast<data::SparsePageDMatrix*>(dmat)) + ".row.page";
  EXPECT_TRUE(FileExists(path));
  path = data::MakeId(tmp_file + ".cache", dynamic_cast<data::SparsePageDMatrix*>(dmat)) +
         ".ellpack.page";
  EXPECT_TRUE(FileExists(path));

  delete dmat;
}

TEST(SparsePageDMatrix, EllpackSkipSparsePage) {
  // Test Ellpack can avoid loading sparse page after the initialization.
  dmlc::TemporaryDirectory tmpdir;
  std::size_t n_batches = 6;
  auto Xy = RandomDataGenerator{180, 12, 0.0}.Batches(n_batches).GenerateSparsePageDMatrix(
      tmpdir.path + "/", true);
  auto ctx = MakeCUDACtx(0);
  auto cpu = ctx.MakeCPU();
  bst_bin_t n_bins{256};
  double sparse_thresh{0.8};
  BatchParam batch_param{n_bins, sparse_thresh};

  auto check_ellpack = [&]() {
    std::int32_t k = 0;
    for (auto const& page : Xy->GetBatches<EllpackPage>(&ctx, batch_param)) {
      auto impl = page.Impl();
      ASSERT_EQ(page.Size(), 30);
      ASSERT_EQ(k, impl->base_rowid);
      k += page.Size();
    }
  };

  auto casted = std::dynamic_pointer_cast<data::SparsePageDMatrix>(Xy);
  CHECK(casted);
  check_ellpack();

  // Make the number of fetches don't change (no new fetch)
  auto n_fetches = casted->SparsePageFetchCount();
  for (std::size_t i = 0; i < 3; ++i) {
    for ([[maybe_unused]] auto const& page : Xy->GetBatches<EllpackPage>(&ctx, batch_param)) {
    }
    auto casted = std::dynamic_pointer_cast<data::SparsePageDMatrix>(Xy);
    ASSERT_EQ(casted->SparsePageFetchCount(), n_fetches);
  }
  check_ellpack();

  dh::device_vector<float> hess(Xy->Info().num_row_, 1.0f);
  for (std::size_t i = 0; i < 4; ++i) {
    for ([[maybe_unused]] auto const& page : Xy->GetBatches<SparsePage>(&ctx)) {
    }
    for ([[maybe_unused]] auto const& page : Xy->GetBatches<SortedCSCPage>(&cpu)) {
    }
    for ([[maybe_unused]] auto const& page : Xy->GetBatches<EllpackPage>(&ctx, batch_param)) {
    }
    // Approx tree method pages
    {
      BatchParam regen{n_bins, dh::ToSpan(hess), false};
      for ([[maybe_unused]] auto const& page : Xy->GetBatches<EllpackPage>(&ctx, regen)) {
      }
    }
    {
      BatchParam regen{n_bins, dh::ToSpan(hess), true};
      for ([[maybe_unused]] auto const& page : Xy->GetBatches<EllpackPage>(&ctx, regen)) {
      }
    }

    check_ellpack();
  }

  // half the pages
  {
    auto it = Xy->GetBatches<SparsePage>(&ctx).begin();
    for (std::size_t i = 0; i < n_batches / 2; ++i) {
      ++it;
    }
    check_ellpack();
  }
  {
    auto it = Xy->GetBatches<EllpackPage>(&ctx, batch_param).begin();
    for (std::size_t i = 0; i < n_batches / 2; ++i) {
      ++it;
    }
    check_ellpack();
  }
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
  std::int64_t batch_count = 0;
  bst_idx_t row_count = 0;
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
    ASSERT_EQ(iterators[i].use_count(), 1);
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

namespace {
// Test comparing external DMatrix with in-core DMatrix
class TestEllpackPageExt : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  void Run(bool on_host, bool is_dense) {
    float sparsity = is_dense ? 0.0 : 0.2;

    auto ctx = MakeCUDACtx(0);
    constexpr bst_idx_t kRows = 64;
    constexpr size_t kCols = 2;

    // Create an in-memory DMatrix.
    auto p_fmat = RandomDataGenerator{kRows, kCols, sparsity}.GenerateDMatrix(true);

    // Create a DMatrix with multiple batches.
    dmlc::TemporaryDirectory tmpdir;
    auto prefix = tmpdir.path + "/cache";

    auto p_ext_fmat = RandomDataGenerator{kRows, kCols, sparsity}
                          .Batches(4)
                          .OnHost(on_host)
                          .GenerateSparsePageDMatrix(prefix, true);

    auto param = BatchParam{2, tree::TrainParam::DftSparseThreshold()};
    auto impl = (*p_fmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();
    ASSERT_EQ(impl->base_rowid, 0);
    ASSERT_EQ(impl->n_rows, kRows);
    ASSERT_EQ(impl->is_dense, is_dense);
    ASSERT_EQ(impl->row_stride, 2);
    ASSERT_EQ(impl->Cuts().TotalBins(), 4);

    std::unique_ptr<EllpackPageImpl> impl_ext;
    size_t offset = 0;
    for (auto& batch : p_ext_fmat->GetBatches<EllpackPage>(&ctx, param)) {
      if (!impl_ext) {
        impl_ext = std::make_unique<EllpackPageImpl>(
            batch.Impl()->gidx_buffer.Device(), batch.Impl()->CutsShared(), batch.Impl()->is_dense,
            batch.Impl()->row_stride, kRows);
      }
      auto n_elems = impl_ext->Copy(ctx.Device(), batch.Impl(), offset);
      offset += n_elems;
    }
    ASSERT_EQ(impl_ext->base_rowid, 0);
    ASSERT_EQ(impl_ext->n_rows, kRows);
    ASSERT_EQ(impl_ext->is_dense, is_dense);
    ASSERT_EQ(impl_ext->row_stride, 2);
    ASSERT_EQ(impl_ext->Cuts().TotalBins(), 4);

    std::vector<common::CompressedByteT> buffer(impl->gidx_buffer.HostVector());
    std::vector<common::CompressedByteT> buffer_ext(impl_ext->gidx_buffer.HostVector());
    ASSERT_EQ(buffer, buffer_ext);
  }
};
}  // anonymous namespace

TEST_P(TestEllpackPageExt, Data) {
  auto [on_host, is_dense] = this->GetParam();
  this->Run(on_host, is_dense);
}

INSTANTIATE_TEST_SUITE_P(EllpackPageExt, TestEllpackPageExt, ::testing::ValuesIn([]() {
                           std::vector<std::tuple<bool, bool>> values;
                           for (auto on_host : {true, false}) {
                             for (auto is_dense : {true, false}) {
                               values.emplace_back(on_host, is_dense);
                             }
                           }
                           return values;
                         }()),
                         [](::testing::TestParamInfo<TestEllpackPageExt::ParamType> const& info) {
                           auto on_host = std::get<0>(info.param);
                           auto is_dense = std::get<1>(info.param);
                           std::stringstream ss;
                           ss << (on_host ? "host" : "ext");
                           ss << "_";
                           ss << (is_dense ? "dense" : "sparse");
                           return ss.str();
                         });

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
      dh::LaunchN(kCols, ReadRowFunction(impl->GetDeviceAccessor(ctx.Device()), current_row,
                                         row_d.data().get()));
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(kCols, ReadRowFunction(impl_ext->GetDeviceAccessor(ctx.Device()), current_row,
                                         row_ext_d.data().get()));
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
