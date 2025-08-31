/**
 * Copyright 2023-2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>

#include "../../../../src/collective/communicator-inl.h"
#include "../../filesystem.h"  // for test_federated_data
#include "../../helpers.h"
#include "test_worker.h"

namespace xgboost {

void VerifyLoadUri() {
  auto const rank = collective::GetRank();

  size_t constexpr kRows{16};
  size_t const kCols = 8 + rank;

  common::TemporaryDirectory tmpdir;
  auto path = tmpdir.Path() / ("small" + std::to_string(rank) + ".csv");
  CreateTestCSV(path.string(), kRows, kCols);

  std::unique_ptr<DMatrix> dmat;
  std::string uri = path.string() + "?format=csv";
  dmat.reset(DMatrix::Load(uri, false, DataSplitMode::kCol));

  ASSERT_EQ(dmat->Info().num_col_, 8 * collective::GetWorldSize() + 1);
  ASSERT_EQ(dmat->Info().num_row_, kRows);

  for (auto const& page : dmat->GetBatches<SparsePage>()) {
    auto entries = page.GetView().data;
    auto index = 0;
    int offsets[] = {0, 8, 17};
    int offset = offsets[rank];
    for (std::size_t row = 0; row < kRows; row++) {
      for (std::size_t col = 0; col < kCols; col++) {
        EXPECT_EQ(entries[index].index, col + offset);
        index++;
      }
    }
  }
}

TEST(FederatedDataTest, LoadUri) {
  static int constexpr kWorldSize{2};
  collective::TestFederatedGlobal(kWorldSize, [] { VerifyLoadUri(); });
}
}  // namespace xgboost
