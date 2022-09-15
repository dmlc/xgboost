/*!
 * Copyright 2021-2022 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include "../../../src/common/column_matrix.h"
#include "../../../src/common/io.h"  // MemoryBufferStream
#include "../../../src/data/gradient_index.h"
#include "../helpers.h"

namespace xgboost {
namespace data {
TEST(GradientIndex, ExternalMemory) {
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(10000);
  std::vector<size_t> base_rowids;
  std::vector<float> hessian(dmat->Info().num_row_, 1);
  for (auto const &page : dmat->GetBatches<GHistIndexMatrix>({64, hessian, true})) {
    base_rowids.push_back(page.base_rowid);
  }
  size_t i = 0;
  for (auto const &page : dmat->GetBatches<SparsePage>()) {
    ASSERT_EQ(base_rowids[i], page.base_rowid);
    ++i;
  }


  base_rowids.clear();
  for (auto const &page : dmat->GetBatches<GHistIndexMatrix>({64, hessian, false})) {
    base_rowids.push_back(page.base_rowid);
  }
  i = 0;
  for (auto const &page : dmat->GetBatches<SparsePage>()) {
    ASSERT_EQ(base_rowids[i], page.base_rowid);
    ++i;
  }
}

TEST(GradientIndex, FromCategoricalBasic) {
  size_t constexpr kRows = 1000, kCats = 13, kCols = 1;
  size_t max_bins = 8;
  auto x = GenerateRandomCategoricalSingleColumn(kRows, kCats);
  auto m = GetDMatrixFromData(x, kRows, 1);

  auto &h_ft = m->Info().feature_types.HostVector();
  h_ft.resize(kCols, FeatureType::kCategorical);

  BatchParam p(max_bins, 0.8);
  GHistIndexMatrix gidx(m.get(), max_bins, p.sparse_thresh, false, common::OmpGetNumThreads(0), {});

  auto x_copy = x;
  std::sort(x_copy.begin(), x_copy.end());
  auto n_uniques = std::unique(x_copy.begin(), x_copy.end()) - x_copy.begin();
  ASSERT_EQ(n_uniques, kCats);

  auto const &h_cut_ptr = gidx.cut.Ptrs();
  auto const &h_cut_values = gidx.cut.Values();

  ASSERT_EQ(h_cut_ptr.size(), 2);
  ASSERT_EQ(h_cut_values.size(), kCats);

  auto const &index = gidx.index;

  for (size_t i = 0; i < x.size(); ++i) {
    auto bin = index[i];
    auto bin_value = h_cut_values.at(bin);
    ASSERT_EQ(common::AsCat(x[i]), common::AsCat(bin_value));
  }
}

TEST(GradientIndex, PushBatch) {
  size_t constexpr kRows = 64, kCols = 4;
  bst_bin_t max_bins = 64;
  float st = 0.5;

  auto test = [&](float sparisty) {
    auto m = RandomDataGenerator{kRows, kCols, sparisty}.GenerateDMatrix(true);
    auto cuts = common::SketchOnDMatrix(m.get(), max_bins, common::OmpGetNumThreads(0), false, {});
    common::HistogramCuts copy_cuts = cuts;

    ASSERT_EQ(m->Info().num_row_, kRows);
    ASSERT_EQ(m->Info().num_col_, kCols);
    GHistIndexMatrix gmat{m->Info(), std::move(copy_cuts), max_bins};

    for (auto const &page : m->GetBatches<SparsePage>()) {
      SparsePageAdapterBatch batch{page.GetView()};
      gmat.PushAdapterBatch(m->Ctx(), 0, 0, batch, std::numeric_limits<float>::quiet_NaN(), {}, st,
                            m->Info().num_row_);
      gmat.PushAdapterBatchColumns(m->Ctx(), batch, std::numeric_limits<float>::quiet_NaN(), 0);
    }
    for (auto const &page : m->GetBatches<GHistIndexMatrix>(BatchParam{max_bins, st})) {
      for (size_t i = 0; i < kRows; ++i) {
        for (size_t j = 0; j < kCols; ++j) {
          auto v0 = gmat.GetFvalue(i, j, false);
          auto v1 = page.GetFvalue(i, j, false);
          if (sparisty == 0.0) {
            ASSERT_FALSE(std::isnan(v0));
          }
          if (!std::isnan(v0)) {
            ASSERT_EQ(v0, v1);
          }
        }
      }
    }
  };

  test(0.0f);
  test(0.5f);
  test(0.9f);
}

#if defined(XGBOOST_USE_CUDA)

namespace {
class GHistIndexMatrixTest : public testing::TestWithParam<std::tuple<float, float>> {
 protected:
  void Run(float density, double threshold) {
    // Only testing with small sample size as the cuts might be different between host and
    // device.
    size_t n_samples{128}, n_features{13};
    Context ctx;
    ctx.gpu_id = 0;
    auto Xy = RandomDataGenerator{n_samples, n_features, 1 - density}.GenerateDMatrix(true);
    std::unique_ptr<GHistIndexMatrix> from_ellpack;
    ASSERT_TRUE(Xy->SingleColBlock());
    bst_bin_t constexpr kBins{17};
    auto p = BatchParam{kBins, threshold};
    for (auto const &page : Xy->GetBatches<EllpackPage>(BatchParam{0, kBins})) {
      from_ellpack.reset(new GHistIndexMatrix{&ctx, Xy->Info(), page, p});
    }

    for (auto const &from_sparse_page : Xy->GetBatches<GHistIndexMatrix>(p)) {
      ASSERT_EQ(from_sparse_page.IsDense(), from_ellpack->IsDense());
      ASSERT_EQ(from_sparse_page.base_rowid, 0);
      ASSERT_EQ(from_sparse_page.base_rowid, from_ellpack->base_rowid);
      ASSERT_EQ(from_sparse_page.Size(), from_ellpack->Size());
      ASSERT_EQ(from_sparse_page.index.Size(), from_ellpack->index.Size());

      auto const &gidx_from_sparse = from_sparse_page.index;
      auto const &gidx_from_ellpack = from_ellpack->index;

      for (size_t i = 0; i < gidx_from_sparse.Size(); ++i) {
        ASSERT_EQ(gidx_from_sparse[i], gidx_from_ellpack[i]);
      }

      auto const &columns_from_sparse = from_sparse_page.Transpose();
      auto const &columns_from_ellpack = from_ellpack->Transpose();
      ASSERT_EQ(columns_from_sparse.AnyMissing(), columns_from_ellpack.AnyMissing());
      ASSERT_EQ(columns_from_sparse.GetTypeSize(), columns_from_ellpack.GetTypeSize());
      ASSERT_EQ(columns_from_sparse.GetNumFeature(), columns_from_ellpack.GetNumFeature());
      for (size_t i = 0; i < n_features; ++i) {
        ASSERT_EQ(columns_from_sparse.GetColumnType(i), columns_from_ellpack.GetColumnType(i));
      }

      std::string from_sparse_buf;
      {
        common::MemoryBufferStream fo{&from_sparse_buf};
        columns_from_sparse.Write(&fo);
      }
      std::string from_ellpack_buf;
      {
        common::MemoryBufferStream fo{&from_ellpack_buf};
        columns_from_sparse.Write(&fo);
      }
      ASSERT_EQ(from_sparse_buf, from_ellpack_buf);
    }
  }
};
}  // anonymous namespace

TEST_P(GHistIndexMatrixTest, FromEllpack) {
  float sparsity;
  double thresh;
  std::tie(sparsity, thresh) = GetParam();
  this->Run(sparsity, thresh);
}

INSTANTIATE_TEST_SUITE_P(GHistIndexMatrix, GHistIndexMatrixTest,
                         testing::Values(std::make_tuple(1.f, .0),    // no missing
                                         std::make_tuple(.2f, .8),    // sparse columns
                                         std::make_tuple(.8f, .2),    // dense columns
                                         std::make_tuple(1.f, .2),    // no missing
                                         std::make_tuple(.5f, .6),    // sparse columns
                                         std::make_tuple(.6f, .4)));  // dense columns

#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace data
}  // namespace xgboost
