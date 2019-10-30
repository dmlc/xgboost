// Copyright by Contributors

#include <dmlc/filesystem.h>
#include "../helpers.h"

namespace xgboost {

TEST(GPUSparsePageDMatrix, EllpackPage) {
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

TEST(GPUSparsePageDMatrix, MultipleEllpackPages) {
  dmlc::TemporaryDirectory tmpdir;
  std::string filename = tmpdir.path + "/big.libsvm";
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(12, 64, filename);

  // Loop over the batches and count the records
  int64_t batch_count = 0;
  int64_t row_count = 0;
  for (const auto& batch : dmat->GetBatches<EllpackPage>({0, 256, 0, 7UL})) {
    batch_count++;
    row_count += batch.Size();
  }
  EXPECT_GE(batch_count, 2);
  EXPECT_EQ(row_count, dmat->Info().num_row_);

  EXPECT_TRUE(FileExists(filename + ".cache.ellpack.page"));
}

}  // namespace xgboost
