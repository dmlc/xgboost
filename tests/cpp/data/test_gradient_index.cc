/*!
 * Copyright 2021 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include "../helpers.h"
#include "../../../src/data/gradient_index.h"

namespace xgboost {
namespace data {
TEST(GradientIndex, ExternalMemory) {
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(10000);
  std::vector<size_t> base_rowids;
  for (auto const& page : dmat->GetBatches<GHistIndexMatrix>({0, 64})) {
    base_rowids.push_back(page.base_rowid);
  }
  size_t i = 0;
  for (auto const& page : dmat->GetBatches<SparsePage>()) {
    ASSERT_EQ(base_rowids[i], page.base_rowid);
    ++i;
  }
}
}  // namespace data
}  // namespace xgboost
