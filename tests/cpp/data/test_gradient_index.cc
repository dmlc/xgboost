/*!
 * Copyright 2021-2022 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

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
  GHistIndexMatrix gidx;

  gidx.Init(m.get(), max_bins, p.sparse_thresh, false, common::OmpGetNumThreads(0), {});

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
}  // namespace data
}  // namespace xgboost
