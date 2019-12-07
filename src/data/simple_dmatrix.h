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
#include <memory>
#include <limits>
#include <utility>
#include <vector>

#include "simple_csr_source.h"
#include "../common/group_data.h"
#include "../common/math.h"
#include "adapter.h"

namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  explicit SimpleDMatrix(std::unique_ptr<DataSource<SparsePage>>&& source)
      : source_(std::move(source)) {}

  template <typename AdapterT>
  explicit SimpleDMatrix(AdapterT* adapter, float missing, int nthread) {
    // Set number of threads but keep old value so we can reset it after
    const int nthreadmax = omp_get_max_threads();
    if (nthread <= 0) nthread = nthreadmax;
    int nthread_original = omp_get_max_threads();
    omp_set_num_threads(nthread);

    source_.reset(new SimpleCSRSource());
    SimpleCSRSource& mat = *reinterpret_cast<SimpleCSRSource*>(source_.get());
    std::vector<uint64_t> qids;
    uint64_t default_max = std::numeric_limits<uint64_t>::max();
    uint64_t last_group_id = default_max;
    bst_uint group_size = 0;
    auto& offset_vec = mat.page_.offset.HostVector();
    auto& data_vec = mat.page_.data.HostVector();
    uint64_t inferred_num_columns = 0;

    size_t num_rows_processed = 0;
    size_t rows_from_last_batch = 0;

    using OffsetVecValueT = std::remove_reference<decltype(offset_vec)>::type::value_type;
    std::vector<std::vector<OffsetVecValueT>> tmp_thread_ptr;

    adapter->BeforeFirst();
    // Iterate over batches of input data
    while (adapter->Next()) {
      auto &batch = adapter->Value();

      // Batch data and its offsets
      std::vector<OffsetVecValueT> batch_offset_vec;
      std::vector<Entry> batch_data_vec;
      common::ParallelGroupBuilder<Entry, OffsetVecValueT> builder(&batch_offset_vec,
                                                                   &batch_data_vec,
                                                                   &tmp_thread_ptr);

      // First-pass over the batch counting valid elements
      size_t num_lines = batch.Size();
      if (!rows_from_last_batch) rows_from_last_batch = num_lines;
      // The first parameter is merely a hint. Hence, if the adapter is a CSC styled matrix
      // this will be the number of columns for the *initial* batch. However, this will
      // be dynamically resized (when there are more rows than columns) as and when the
      // builder adds the budget for the elements from the first batch.
      // For the subsequent batches, the initial budget will be max of rows across
      // batches seen thus far, *if* the CSC adapter feeds in more batches, and will be a good
      // heuristic for the initial size of the thread vector (to reduce the number of dynamic
      // rellocations).
      // Note: Large number of dynamic reallocations can occur, if this is a CSC type adapter,
      // that feeds in a large number of rows with few features in a *single* batch.
      builder.InitBudget(rows_from_last_batch, nthread);
#pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_lines); ++i) {  // NOLINT(*)
        int tid = omp_get_thread_num();
        auto line = batch.GetLine(i);
        for (auto j = 0ull; j < line.Size(); j++) {
          auto element = line.GetElement(j);
          inferred_num_columns =
              std::max(inferred_num_columns,
                       static_cast<uint64_t>(element.column_idx + 1));
          if (!common::CheckNAN(element.value) && element.value != missing) {
            builder.AddBudget(element.row_idx - num_rows_processed, tid);
          }
        }
      }
      builder.InitStorage();

      // Second pass over batch, placing elements in correct position
#pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_lines); ++i) {  // NOLINT(*)
        int tid = omp_get_thread_num();
        auto line = batch.GetLine(i);
        for (auto j = 0ull; j < line.Size(); j++) {
          auto element = line.GetElement(j);
          if (!common::CheckNAN(element.value) && element.value != missing) {
            builder.Push(element.row_idx - num_rows_processed,
                         Entry(element.column_idx, element.value), tid);
          }
        }
      }

      // Append meta information if available
      if (batch.Labels() != nullptr) {
        auto& labels = mat.info.labels_.HostVector();
        labels.insert(labels.end(), batch.Labels(), batch.Labels() + batch.Size());
      }
      if (batch.Weights() != nullptr) {
        auto& weights = mat.info.weights_.HostVector();
        weights.insert(weights.end(), batch.Weights(), batch.Weights() + batch.Size());
      }
      if (batch.Qid() != nullptr) {
        qids.insert(qids.end(), batch.Qid(), batch.Qid() + batch.Size());
        // get group
        for (size_t i = 0; i < batch.Size(); ++i) {
          const uint64_t cur_group_id = batch.Qid()[i];
          if (last_group_id == default_max || last_group_id != cur_group_id) {
            mat.info.group_ptr_.push_back(group_size);
          }
          last_group_id = cur_group_id;
          ++group_size;
        }
      }

      // Remove all trailing empty rows, as we have now converted the data fed to us via the
      // adapter into a CSR style dmatrix. This is required as InitBudget may oversize the thread
      // vector with more number of columns/rows. The offset vector for the CSR matrix will now
      // have trailing entries for the rows/columns previously forecasted but not present. Sweep
      // through those and remove them.
      if (batch_offset_vec.size() > 1) {
        auto last_row_ptr_val = batch_offset_vec.back();
        auto k =  batch_offset_vec.size() - 1;
        for (; k; --k) {
          auto cur_row_ptr_val = batch_offset_vec[k - 1];
          if (cur_row_ptr_val != last_row_ptr_val) break;
          last_row_ptr_val = cur_row_ptr_val;
        }
        batch_offset_vec.resize(k + 1);
      }

      rows_from_last_batch = std::max(batch_offset_vec.size() - 1, rows_from_last_batch);

      if (data_vec.empty()) {
        data_vec.swap(batch_data_vec);
        offset_vec.swap(batch_offset_vec);
      } else {
        CHECK(batch_offset_vec.size() && batch_offset_vec.front() == 0);

        // Recompute the dmatrix row offsets based on the data already present in it
        auto last_val = offset_vec.back();
        auto prev_size = offset_vec.size();
        auto batch_offset_vec_idx = 1;  // Skip the leading 0
        offset_vec.resize(prev_size + batch_offset_vec.size() - 1);
        std::for_each(
          offset_vec.begin() + prev_size, offset_vec.end(),
          [&](OffsetVecValueT &v) { v = last_val + batch_offset_vec[batch_offset_vec_idx++]; });

        // Append the batch data with the dmatrix data
        data_vec.insert(data_vec.end(), batch_data_vec.begin(), batch_data_vec.end());
      }

      num_rows_processed = offset_vec.size() - 1;
    }

    if (last_group_id != default_max) {
      if (group_size > mat.info.group_ptr_.back()) {
        mat.info.group_ptr_.push_back(group_size);
      }
    }

    // Deal with empty rows/columns if necessary
    if (adapter->NumColumns() == kAdapterUnknownSize) {
      mat.info.num_col_ = inferred_num_columns;
    } else {
      mat.info.num_col_ = adapter->NumColumns();
    }
    // Synchronise worker columns
    rabit::Allreduce<rabit::op::Max>(&mat.info.num_col_, 1);

    if (adapter->NumRows() == kAdapterUnknownSize) {
      mat.info.num_row_ = offset_vec.size() - 1;
    } else {
      if (offset_vec.empty()) {
        offset_vec.emplace_back(0);
      }

      while (offset_vec.size() - 1 < adapter->NumRows()) {
        offset_vec.emplace_back(offset_vec.back());
      }
      mat.info.num_row_ = adapter->NumRows();
    }
    mat.info.num_nonzero_ = data_vec.size();
    omp_set_num_threads(nthread_original);
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
