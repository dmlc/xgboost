/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <xgboost/base.h>

#include <utility>

#include "../../../src/common/categorical.h"          // for AsCat
#include "../../../src/common/compressed_iterator.h"  // for CompressedByteT
#include "../../../src/common/hist_util.h"
#include "../../../src/data/device_adapter.cuh"  // for CupyAdapter
#include "../../../src/data/ellpack_page.cuh"
#include "../../../src/data/ellpack_page.h"
#include "../../../src/data/gradient_index.h"  // for GHistIndexMatrix
#include "../../../src/tree/param.h"           // for TrainParam
#include "../helpers.h"
#include "../histogram_helpers.h"
#include "gtest/gtest.h"

namespace xgboost {
TEST(EllpackPage, EmptyDMatrix) {
  constexpr int kNRows = 0, kNCols = 0, kMaxBin = 256;
  constexpr float kSparsity = 0;
  auto dmat = RandomDataGenerator(kNRows, kNCols, kSparsity).GenerateDMatrix();
  auto ctx = MakeCUDACtx(0);
  auto& page = *dmat->GetBatches<EllpackPage>(
                        &ctx, BatchParam{kMaxBin, tree::TrainParam::DftSparseThreshold()})
                    .begin();
  auto impl = page.Impl();
  ASSERT_EQ(impl->info.row_stride, 0);
  ASSERT_EQ(impl->Cuts().TotalBins(), 0);
  ASSERT_EQ(impl->gidx_buffer.size(), 4);
}

TEST(EllpackPage, BuildGidxDense) {
  bst_idx_t n_samples = 16, n_features = 8;
  auto ctx = MakeCUDACtx(0);
  auto page = BuildEllpackPage(&ctx, n_samples, n_features);
  std::vector<common::CompressedByteT> h_gidx_buffer;
  auto h_accessor = page->GetHostAccessor(&ctx, &h_gidx_buffer);

  ASSERT_EQ(page->info.row_stride, n_features);

  std::vector<uint32_t> solution = {
    0, 3, 8,  9, 14, 17, 20, 21,
    0, 4, 7, 10, 14, 16, 19, 22,
    1, 3, 7, 11, 14, 15, 19, 21,
    2, 3, 7,  9, 13, 16, 20, 22,
    2, 3, 6,  9, 12, 16, 20, 21,
    1, 5, 6, 10, 13, 16, 20, 21,
    2, 5, 8,  9, 13, 17, 19, 22,
    2, 4, 6, 10, 14, 17, 19, 21,
    2, 5, 7,  9, 13, 16, 19, 22,
    0, 3, 8, 10, 12, 16, 19, 22,
    1, 3, 7, 10, 13, 16, 19, 21,
    1, 3, 8, 10, 13, 17, 20, 22,
    2, 4, 6,  9, 14, 15, 19, 22,
    1, 4, 6,  9, 13, 16, 19, 21,
    2, 4, 8, 10, 14, 15, 19, 22,
    1, 4, 7, 10, 14, 16, 19, 21,
  };
  for (size_t i = 0; i < n_samples * n_features; ++i) {
    auto fidx = i % n_features;
    ASSERT_EQ(solution[i], h_accessor.gidx_iter[i] + h_accessor.feature_segments[fidx]);
  }
  ASSERT_EQ(page->NumSymbols(), 3);
  ASSERT_EQ(page->NumNonMissing(&ctx, {}), n_samples * n_features);
  ASSERT_EQ(page->NumSymbols(), h_accessor.NullValue());
}

TEST(EllpackPage, BuildGidxSparse) {
  int constexpr kNRows = 16, kNCols = 8;
  auto ctx = MakeCUDACtx(0);
  auto page = BuildEllpackPage(&ctx, kNRows, kNCols, 0.9f);

  std::vector<common::CompressedByteT> h_gidx_buffer;
  auto h_acc = page->GetHostAccessor(&ctx, &h_gidx_buffer);

  ASSERT_EQ(page->info.row_stride, 3);

  // row_stride = 3, 16 rows, 48 entries for ELLPack
  std::vector<uint32_t> solution = {
    15, 24, 24,  0, 24, 24, 24, 24, 24, 24, 24, 24, 20, 24, 24, 24,
    24, 24, 24, 24, 24,  5, 24, 24,  0, 16, 24, 15, 24, 24, 24, 24,
    24,  7, 14, 16,  4, 24, 24, 24, 24, 24,  9, 24, 24,  1, 24, 24
  };
  for (size_t i = 0; i < kNRows * page->info.row_stride; ++i) {
    ASSERT_EQ(solution[i], h_acc.gidx_iter[i]);
  }
}

TEST(EllpackPage, FromCategoricalBasic) {
  using common::AsCat;
  size_t constexpr kRows = 1000, kCats = 13, kCols = 1;
  int32_t max_bins = 8;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, kCats);
  auto m = GetDMatrixFromData(x, kRows, 1);
  auto& h_ft = m->Info().feature_types.HostVector();
  h_ft.resize(kCols, FeatureType::kCategorical);

  auto ctx = MakeCUDACtx(0);
  auto p = BatchParam{max_bins, tree::TrainParam::DftSparseThreshold()};
  auto ellpack = EllpackPage(&ctx, m.get(), p);
  auto accessor = ellpack.Impl()->GetDeviceAccessor(&ctx);
  ASSERT_EQ(kCats, accessor.NumBins());

  auto x_copy = x;
  std::sort(x_copy.begin(), x_copy.end());
  auto n_uniques = std::unique(x_copy.begin(), x_copy.end()) - x_copy.begin();
  ASSERT_EQ(n_uniques, kCats);

  std::vector<uint32_t> h_cuts_ptr(accessor.NumFeatures() + 1);
  dh::safe_cuda(cudaMemcpyAsync(h_cuts_ptr.data(), accessor.feature_segments,
                                sizeof(bst_feature_t) * h_cuts_ptr.size(), cudaMemcpyDefault));
  std::vector<float> h_cuts_values(accessor.gidx_fvalue_map.size());
  dh::CopyDeviceSpanToVector(&h_cuts_values, accessor.gidx_fvalue_map);

  ASSERT_EQ(h_cuts_ptr.size(), 2);
  ASSERT_EQ(h_cuts_values.size(), kCats);

  std::vector<common::CompressedByteT> h_gidx_buffer;
  auto h_accessor = ellpack.Impl()->GetHostAccessor(&ctx, &h_gidx_buffer);

  for (size_t i = 0; i < x.size(); ++i) {
    auto bin = h_accessor.gidx_iter[i];
    auto bin_value = h_cuts_values.at(bin);
    ASSERT_EQ(AsCat(x[i]), AsCat(bin_value));
  }
}

TEST(EllpackPage, FromCategoricalMissing) {
  auto ctx = MakeCUDACtx(0);

  std::shared_ptr<common::HistogramCuts> cuts;
  auto nan = std::numeric_limits<float>::quiet_NaN();
  // 2 rows and 3 columns. The second column is nan, row_stride is 2.
  std::vector<float> data{{0.1, nan, 1, 0.2, nan, 0}};
  auto p_fmat = GetDMatrixFromData(data, 2, 3);
  p_fmat->Info().feature_types.HostVector() = {FeatureType::kNumerical, FeatureType::kNumerical,
                                               FeatureType::kCategorical};
  p_fmat->Info().feature_types.SetDevice(ctx.Device());

  auto p = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(&ctx, p)) {
    cuts = std::make_shared<common::HistogramCuts>(page.Cuts());
  }
  cuts->SetDevice(ctx.Device());
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, p)) {
    std::vector<common::CompressedByteT> h_buffer;
    auto h_acc = page.Impl()->GetHostAccessor(&ctx, &h_buffer,
                                              p_fmat->Info().feature_types.ConstDeviceSpan());
    ASSERT_EQ(h_acc.n_rows, 2);
    ASSERT_EQ(cuts->NumFeatures(), 3);
    ASSERT_EQ(h_acc.row_stride, 2);
    ASSERT_EQ(h_acc.gidx_iter[0], 0);
    ASSERT_EQ(h_acc.gidx_iter[1], 4);  // cat 1
    ASSERT_EQ(h_acc.gidx_iter[2], 1);
    ASSERT_EQ(h_acc.gidx_iter[3], 3);  // cat 0
  }
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

TEST(EllpackPage, Copy) {
  constexpr size_t kRows = 1024;
  constexpr size_t kCols = 16;

  // Create a DMatrix with multiple batches.
  auto dmat =
      RandomDataGenerator{kRows, kCols, 0.0f}.Batches(4).GenerateSparsePageDMatrix("temp", true);
  auto ctx = MakeCUDACtx(0);
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  auto page = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();

  // Create an empty result page.
  EllpackPageImpl result(&ctx, page->CutsShared(), page->is_dense, page->info.row_stride, kRows);

  // Copy batch pages into the result page.
  size_t offset = 0;
  for (auto& batch : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    size_t num_elements = result.Copy(&ctx, batch.Impl(), offset);
    offset += num_elements;
  }

  size_t current_row = 0;
  thrust::device_vector<bst_float> row_d(kCols);
  thrust::device_vector<bst_float> row_result_d(kCols);
  std::vector<bst_float> row(kCols);
  std::vector<bst_float> row_result(kCols);
  for (auto& page : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    auto impl = page.Impl();
    EXPECT_EQ(impl->base_rowid, current_row);

    for (size_t i = 0; i < impl->Size(); i++) {
      dh::LaunchN(kCols,
                  ReadRowFunction(impl->GetDeviceAccessor(&ctx), current_row, row_d.data().get()));
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(kCols, ReadRowFunction(result.GetDeviceAccessor(&ctx), current_row,
                                         row_result_d.data().get()));
      thrust::copy(row_result_d.begin(), row_result_d.end(), row_result.begin());

      EXPECT_EQ(row, row_result);
      current_row++;
    }
  }
}

TEST(EllpackPage, Compact) {
  constexpr size_t kRows = 16;
  constexpr size_t kCols = 2;
  constexpr size_t kCompactedRows = 8;

  // Create a DMatrix with multiple batches.
  auto dmat =
      RandomDataGenerator{kRows, kCols, 0.0f}.Batches(2).GenerateSparsePageDMatrix("temp", true);
  auto ctx = MakeCUDACtx(0);
  auto param = BatchParam{256, tree::TrainParam::DftSparseThreshold()};
  auto page = (*dmat->GetBatches<EllpackPage>(&ctx, param).begin()).Impl();

  // Create an empty result page.
  EllpackPageImpl result(&ctx, page->CutsShared(), page->is_dense, page->info.row_stride,
                         kCompactedRows);

  // Compact batch pages into the result page.
  std::vector<size_t> row_indexes_h {
    SIZE_MAX, 0, 1, 2, SIZE_MAX, 3, SIZE_MAX, 4, 5, SIZE_MAX, 6, SIZE_MAX, 7, SIZE_MAX, SIZE_MAX,
    SIZE_MAX};
  thrust::device_vector<size_t> row_indexes_d = row_indexes_h;
  common::Span<size_t> row_indexes_span(row_indexes_d.data().get(), kRows);
  for (auto& batch : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    result.Compact(&ctx, batch.Impl(), row_indexes_span);
  }

  size_t current_row = 0;
  thrust::device_vector<bst_float> row_d(kCols);
  thrust::device_vector<bst_float> row_result_d(kCols);
  std::vector<bst_float> row(kCols);
  std::vector<bst_float> row_result(kCols);
  for (auto& page : dmat->GetBatches<EllpackPage>(&ctx, param)) {
    auto impl = page.Impl();
    ASSERT_EQ(impl->base_rowid, current_row);

    for (size_t i = 0; i < impl->Size(); i++) {
      size_t compacted_row = row_indexes_h[current_row];
      if (compacted_row == SIZE_MAX) {
        current_row++;
        continue;
      }

      dh::LaunchN(kCols,
                  ReadRowFunction(impl->GetDeviceAccessor(&ctx), current_row, row_d.data().get()));
      dh::safe_cuda(cudaDeviceSynchronize());
      thrust::copy(row_d.begin(), row_d.end(), row.begin());

      dh::LaunchN(kCols, ReadRowFunction(result.GetDeviceAccessor(&ctx), compacted_row,
                                         row_result_d.data().get()));
      thrust::copy(row_result_d.begin(), row_result_d.end(), row_result.begin());

      EXPECT_EQ(row, row_result);
      current_row++;
    }
  }
}

namespace {
// Test for treating sparse ellpack as a dense
class CompressedDense : public ::testing::TestWithParam<std::size_t> {
  auto InitSparsePage(std::size_t null_column) const {
    bst_idx_t n_samples = 16, n_features = 8;
    std::vector<float> data(n_samples * n_features);

    std::iota(data.begin(), data.end(), 0.0f);
    for (std::size_t i = 0; i < data.size(); i += n_features) {
      data[i + null_column] = std::numeric_limits<float>::quiet_NaN();
    }
    data[null_column] = null_column;  // keep the first sample full.
    auto p_fmat = GetDMatrixFromData(data, n_samples, n_features);
    return p_fmat;
  }

  void CheckBasic(Context const* ctx, BatchParam batch, std::size_t null_column,
                  EllpackPageImpl const& impl) {
    ASSERT_FALSE(impl.IsDense());
    ASSERT_TRUE(impl.IsDenseCompressed());
    ASSERT_EQ(impl.NumSymbols(), batch.max_bin + 1);

    std::vector<common::CompressedByteT> h_gidx;
    auto h_acc = impl.GetHostAccessor(ctx, &h_gidx);
    ASSERT_EQ(h_acc.row_stride, h_acc.NumFeatures());
    ASSERT_EQ(h_acc.NullValue(), batch.max_bin);
    for (std::size_t i = 0; i < h_acc.row_stride * h_acc.n_rows; ++i) {
      auto [m, n] = linalg::UnravelIndex(i, h_acc.n_rows, h_acc.row_stride);
      if (n == null_column && m != 0) {
        ASSERT_EQ(static_cast<std::int32_t>(h_acc.gidx_iter[i]), h_acc.NullValue());
      } else {
        ASSERT_EQ(static_cast<std::int32_t>(h_acc.gidx_iter[i]), m);
      }
    }
  }

 public:
  void CheckFromSparsePage(std::size_t null_column) {
    auto p_fmat = this->InitSparsePage(null_column);
    auto ctx = MakeCUDACtx(0);
    auto batch = BatchParam{static_cast<bst_bin_t>(p_fmat->Info().num_row_),
                            std::numeric_limits<float>::quiet_NaN()};

    for (auto const& ellpack : p_fmat->GetBatches<EllpackPage>(&ctx, batch)) {
      auto impl = ellpack.Impl();
      this->CheckBasic(&ctx, batch, null_column, *impl);
    }
  }

  void CheckFromAdapter(std::size_t null_column) {
    bst_idx_t n_samples = 16, n_features = 8;

    auto ctx = MakeCUDACtx(0);
    HostDeviceVector<float> data(n_samples * n_features, 0.0f, ctx.Device());
    auto& h_data = data.HostVector();
    std::iota(h_data.begin(), h_data.end(), 0.0f);
    for (std::size_t i = 0; i < h_data.size(); i += n_features) {
      h_data[i + null_column] = std::numeric_limits<float>::quiet_NaN();
    }
    h_data[null_column] = null_column;  // Keep the first sample full.
    auto p_fmat = GetDMatrixFromData(h_data, n_samples, n_features);

    data.ConstDeviceSpan();  // Pull to device
    auto arri = GetArrayInterface(&data, n_samples, n_features);
    auto sarri = Json::Dump(arri);
    data::CupyAdapter adapter{StringView{sarri}};

    Context cpu_ctx;
    auto batch = BatchParam{static_cast<bst_bin_t>(p_fmat->Info().num_row_), 0.8};

    std::shared_ptr<common::HistogramCuts> cuts;
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(&cpu_ctx, batch)) {
      cuts = std::make_shared<common::HistogramCuts>(page.Cuts());
    }
    dh::device_vector<bst_idx_t> row_counts(n_samples, n_features - 1);
    row_counts[0] = n_features;
    auto d_row_counts = dh::ToSpan(row_counts);
    ASSERT_EQ(adapter.NumColumns(), n_features);
    auto impl =
        EllpackPageImpl{&ctx,       adapter.Value(), std::numeric_limits<float>::quiet_NaN(),
                        false,      d_row_counts,    {},
                        n_features, n_samples,       cuts};
    this->CheckBasic(&ctx, batch, null_column, impl);
    dh::DefaultStream().Sync();
  }

  void CheckFromToGHist(std::size_t null_column) {
    Context cpu_ctx;
    auto ctx = MakeCUDACtx(0);
    std::vector<std::uint8_t> orig;
    {
      // Test from GHist
      auto p_fmat = this->InitSparsePage(null_column);
      auto batch = BatchParam{static_cast<bst_bin_t>(p_fmat->Info().num_row_), 0.8};
      for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(&cpu_ctx, batch)) {
        orig = {page.data.cbegin(), page.data.cend()};
        auto impl = EllpackPageImpl{&ctx, page, {}};
        this->CheckBasic(&ctx, batch, null_column, impl);
      }
    }

    {
      // Test to GHist
      auto p_fmat = this->InitSparsePage(null_column);
      auto batch = BatchParam{static_cast<bst_bin_t>(p_fmat->Info().num_row_), 0.8};
      for (auto const& page : p_fmat->GetBatches<EllpackPage>(&ctx, batch)) {
        auto gidx = GHistIndexMatrix{&ctx, p_fmat->Info(), page, batch};
        ASSERT_EQ(gidx.Size(), p_fmat->Info().num_row_);
        for (std::size_t ridx = 0; ridx < gidx.Size(); ++ridx) {
          auto rbegin = gidx.row_ptr[ridx];
          auto rend = gidx.row_ptr[ridx + 1];
          if (ridx == 0) {
            ASSERT_EQ(rend - rbegin, p_fmat->Info().num_col_);
          } else {
            ASSERT_EQ(rend - rbegin, p_fmat->Info().num_col_ - 1);
          }
        }
        // GHist can't compress a dataset with missing values
        ASSERT_FALSE(gidx.index.Offset());
        ASSERT_TRUE(std::equal(gidx.data.cbegin(), gidx.data.cend(), orig.cbegin()));
      }
    }
  }
};

TEST_P(CompressedDense, FromSparsePage) { this->CheckFromSparsePage(this->GetParam()); }

TEST_P(CompressedDense, FromAdapter) { this->CheckFromAdapter(this->GetParam()); }

TEST_P(CompressedDense, FromToGHist) { this->CheckFromToGHist(this->GetParam()); }
}  // anonymous namespace

INSTANTIATE_TEST_SUITE_P(EllpackPage, CompressedDense, testing::Values(0ul, 1ul, 7ul));

namespace {
class SparseEllpack : public testing::TestWithParam<float> {
 protected:
  void TestFromGHistIndex(float sparsity) const {
    // Only testing with small sample size as the cuts might be different between host and
    // device.
    size_t n_samples{128}, n_features{13};
    Context ctx;
    Context gpu_ctx{MakeCUDACtx(0)};
    auto Xy = RandomDataGenerator{n_samples, n_features, sparsity}.GenerateDMatrix(true);
    std::unique_ptr<EllpackPageImpl> from_ghist;
    ASSERT_TRUE(Xy->SingleColBlock());

    for (auto const& page : Xy->GetBatches<GHistIndexMatrix>(&ctx, BatchParam{17, 0.6})) {
      from_ghist.reset(new EllpackPageImpl{&gpu_ctx, page, {}});
    }

    for (auto const& page : Xy->GetBatches<EllpackPage>(
             &gpu_ctx, BatchParam{17, tree::TrainParam::DftSparseThreshold()})) {
      auto from_sparse_page = page.Impl();
      ASSERT_EQ(from_sparse_page->is_dense, from_ghist->is_dense);
      ASSERT_EQ(from_sparse_page->base_rowid, 0);
      ASSERT_EQ(from_sparse_page->base_rowid, from_ghist->base_rowid);
      ASSERT_EQ(from_sparse_page->n_rows, from_ghist->n_rows);
      ASSERT_EQ(from_sparse_page->gidx_buffer.size(), from_ghist->gidx_buffer.size());
      ASSERT_EQ(from_sparse_page->NumSymbols(), from_ghist->NumSymbols());
      std::vector<common::CompressedByteT> h_gidx_from_sparse, h_gidx_from_ghist;
      auto from_ghist_acc = from_ghist->GetHostAccessor(&gpu_ctx, &h_gidx_from_ghist);
      auto from_sparse_acc = from_sparse_page->GetHostAccessor(&gpu_ctx, &h_gidx_from_sparse);
      for (size_t i = 0; i < from_ghist->n_rows * from_ghist->info.row_stride; ++i) {
        ASSERT_EQ(from_ghist_acc.gidx_iter[i], from_sparse_acc.gidx_iter[i]);
      }
    }
  }

  void TestNumNonMissing(float sparsity) const {
    size_t n_samples{1024}, n_features{13};
    auto ctx = MakeCUDACtx(0);
    auto p_fmat = RandomDataGenerator{n_samples, n_features, sparsity}.GenerateDMatrix(true);
    auto nnz = p_fmat->Info().num_nonzero_;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(
             &ctx, BatchParam{17, tree::TrainParam::DftSparseThreshold()})) {
      auto ellpack_nnz =
          page.Impl()->NumNonMissing(&ctx, p_fmat->Info().feature_types.ConstDeviceSpan());
      ASSERT_EQ(nnz, ellpack_nnz);
    }
  }
};
}  // namespace

TEST_P(SparseEllpack, FromGHistIndex) { this->TestFromGHistIndex(GetParam()); }

TEST_P(SparseEllpack, NumNonMissing) { this->TestNumNonMissing(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(EllpackPage, SparseEllpack, ::testing::Values(.0f, .2f, .4f, .8f));
}  // namespace xgboost
