#include <gtest/gtest.h>
#include <vector>

#include "xgboost/data.h"
#include "../helpers.h"

namespace xgboost {
TEST(SparsePage, PushCSC) {
  std::vector<size_t> offset {0};
  std::vector<Entry> data;
  SparsePage page;
  page.offset.HostVector() = offset;
  page.data.HostVector() = data;

  offset = {0, 1, 4};
  for (size_t i = 0; i < offset.back(); ++i) {
    data.emplace_back(Entry(i, 0.1f));
  }

  SparsePage other;
  other.offset.HostVector() = offset;
  other.data.HostVector() = data;

  page.PushCSC(other);

  ASSERT_EQ(page.offset.HostVector().size(), offset.size());
  ASSERT_EQ(page.data.HostVector().size(), data.size());
  for (size_t i = 0; i < offset.size(); ++i) {
    ASSERT_EQ(page.offset.HostVector()[i], offset[i]);
  }
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(page.data.HostVector()[i].index, data[i].index);
  }

  page.PushCSC(other);
  ASSERT_EQ(page.offset.HostVector().size(), offset.size());
  ASSERT_EQ(page.data.Size(), data.size() * 2);

  for (size_t i = 0; i < offset.size(); ++i) {
    ASSERT_EQ(page.offset.HostVector()[i], offset[i] * 2);
  }

  auto inst = page[0];
  ASSERT_EQ(inst.size(), 2);
  for (auto entry : inst) {
    ASSERT_EQ(entry.index, 0);
  }

  inst = page[1];
  ASSERT_EQ(inst.size(), 6);
  std::vector<size_t> indices_sol {1, 2, 3};
  for (size_t i = 0; i < inst.size(); ++i) {
    ASSERT_EQ(inst[i].index, indices_sol[i % 3]);
  }
}

TEST(SparsePage, PushCSCAfterTranspose) {
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(9, 64UL);
  int ncols = dmat->Info().num_col_;
  SparsePage page; // Consolidated sparse page
  for (auto& batch : dmat->GetRowBatches()) {
    // Transpose each batch and push
    SparsePage tmp = batch.GetTranspose(ncols);
    page.PushCSC(tmp);
  }
}
}  // namespace xgboost
