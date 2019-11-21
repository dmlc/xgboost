/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "simple_csr_source.h"
#include "../common/group_data.h"
#include "../common/math.h"

namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  explicit SimpleDMatrix(std::unique_ptr<DataSource<SparsePage>>&& source)
      : source_(std::move(source)) {}

  template <typename AdapterT>
  explicit SimpleDMatrix(const AdapterT& adapter, float missing, int nthread) {
    const int nthreadmax = omp_get_max_threads();
    if (nthread <= 0) nthread = nthreadmax;
    int nthread_orig = omp_get_max_threads();
    omp_set_num_threads(nthread);
    source_.reset(new SimpleCSRSource());
    SimpleCSRSource& mat = *reinterpret_cast<SimpleCSRSource*>(source_.get());
    auto& offset_vec = mat.page_.offset.HostVector();
    auto& data_vec = mat.page_.data.HostVector();
    common::ParallelGroupBuilder<
        Entry, std::remove_reference<decltype(offset_vec)>::type::value_type>
        builder(&offset_vec, &data_vec);
    builder.InitBudget(adapter.GetNumRows(), nthread);
    size_t num_batches = adapter.Size();
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_batches);
         ++i) {  // NOLINT(*)
      int tid = omp_get_thread_num();
      auto batch = adapter[i];
      for (auto j = 0ull; j < batch.Size(); j++) {
        auto element = batch.GetElement(j);
        if (!common::CheckNAN(element.value) && element.value != missing) {
          builder.AddBudget(element.row_idx, tid);
        }
      }
    }
    builder.InitStorage();
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_batches);
         ++i) {  // NOLINT(*)
      int tid = omp_get_thread_num();
      auto batch = adapter[i];
      for (auto j = 0ull; j < batch.Size(); j++) {
        auto element = batch.GetElement(j);
        if (!common::CheckNAN(element.value) && element.value != missing) {
          builder.Push(element.row_idx, Entry(element.column_idx, element.value),
            tid);
        }
      }
    }
    mat.info.num_row_ = mat.page_.offset.Size() - 1;
    if (adapter.GetNumRows() > 0) {
      CHECK_LE(mat.info.num_row_, adapter.GetNumRows());
      // provision for empty rows at the bottom of matrix
      for (uint64_t i = mat.info.num_row_;
           i < static_cast<uint64_t>(adapter.GetNumRows()); ++i) {
        offset_vec.push_back(offset_vec.back());
      }
      mat.info.num_row_ = adapter.GetNumRows();
      CHECK_EQ(mat.info.num_row_, offset_vec.size() - 1);  // sanity check
    }
    mat.info.num_col_ = adapter.GetNumFeatures();
    mat.info.num_nonzero_ = offset_vec.back();
    omp_set_num_threads(nthread_orig);
  }

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  // source data pointer.
  std::unique_ptr<DataSource<SparsePage>> source_;

  std::unique_ptr<CSCPage> column_page_;
  std::unique_ptr<SortedCSCPage> sorted_column_page_;
  std::unique_ptr<EllpackPage> ellpack_page_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
