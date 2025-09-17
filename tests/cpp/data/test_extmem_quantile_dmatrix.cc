/**
 * Copyright 2024, XGBoost Contributors
 */
#include "test_extmem_quantile_dmatrix.h"  // for TestExtMemQdmBasic

#include <gtest/gtest.h>
#include <xgboost/data.h>  // for BatchParam

#include <algorithm>  // for equal

#include "../../../src/common/column_matrix.h"  // for ColumnMatrix
#include "../../../src/data/gradient_index.h"   // for GHistIndexMatrix
#include "../../../src/tree/param.h"            // for TrainParam

namespace xgboost::data {
namespace {
class ExtMemQuantileDMatrixCpu : public ::testing::TestWithParam<float> {
 public:
  void Run(float sparsity) {
    auto equal = [](Context const*, GHistIndexMatrix const& orig, GHistIndexMatrix const& sparse) {
      // Check the CSR matrix
      auto orig_cuts = orig.Cuts();
      auto sparse_cuts = sparse.Cuts();
      ASSERT_EQ(orig_cuts.Values(), sparse_cuts.Values());
      ASSERT_EQ(orig_cuts.MinValues(), sparse_cuts.MinValues());
      ASSERT_EQ(orig_cuts.Ptrs(), sparse_cuts.Ptrs());

      auto orig_ptr = orig.data.data();
      auto sparse_ptr = sparse.data.data();
      ASSERT_EQ(orig.data.size(), sparse.data.size());

      auto equal = std::equal(orig_ptr, orig_ptr + orig.data.size(), sparse_ptr);
      ASSERT_TRUE(equal);

      // Check the column matrix
      common::ColumnMatrix const& orig_columns = orig.Transpose();
      common::ColumnMatrix const& sparse_columns = sparse.Transpose();

      std::string str_orig, str_sparse;
      common::AlignedMemWriteStream fo_orig{&str_orig}, fo_sparse{&str_sparse};
      auto n_bytes_orig = orig_columns.Write(&fo_orig);
      auto n_bytes_sparse = sparse_columns.Write(&fo_sparse);
      ASSERT_EQ(n_bytes_orig, n_bytes_sparse);
      ASSERT_EQ(str_orig, str_sparse);
    };

    Context ctx;
    TestExtMemQdmBasic<GHistIndexMatrix>(
        &ctx, false, sparsity, equal, [](GHistIndexMatrix const& page) { return page.IsDense(); });
  }
};
}  // anonymous namespace

TEST_P(ExtMemQuantileDMatrixCpu, Basic) { this->Run(this->GetParam()); }

INSTANTIATE_TEST_SUITE_P(ExtMemQuantileDMatrix, ExtMemQuantileDMatrixCpu, ::testing::ValuesIn([] {
                           std::vector<float> sparsities{
                               0.0f, tree::TrainParam::DftSparseThreshold(), 0.4f, 0.8f};
                           return sparsities;
                         }()));
}  // namespace xgboost::data
