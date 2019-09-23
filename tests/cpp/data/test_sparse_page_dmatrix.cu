// Copyright by Contributors

#include <dmlc/filesystem.h>
#include "../helpers.h"

namespace xgboost {

TEST(GPUSparsePageDMatrix, DISABLED_EllpackPage) {
//TEST(GPUSparsePageDMatrix, EllpackPage) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  DMatrix* dmat = DMatrix::Load(tmp_file + "#" + tmp_file + ".cache", true, false);

  // Loop over the batches and assert the data is as expected
  for (const auto& batch : dmat->GetBatches<EllpackPage>()) {
    EXPECT_EQ(batch.Size(), dmat->Info().num_row_);
  }

  EXPECT_TRUE(FileExists(tmp_file + ".cache"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.row.page"));
  EXPECT_TRUE(FileExists(tmp_file + ".cache.ellpack.page"));

  delete dmat;
}

}  // namespace xgboost
