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

    adapter->BeforeFirst();
    // Iterate over batches of input data
    while (adapter->Next()) {
      auto &batch = adapter->Value();
      common::ParallelGroupBuilder<
        Entry, std::remove_reference<decltype(offset_vec)>::type::value_type>
        builder(&offset_vec, &data_vec);
      builder.InitBudget(0, nthread);

      // First-pass over the batch counting valid elements
      size_t num_lines = batch.Size();
#pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_lines);
        ++i) {  // NOLINT(*)
        int tid = omp_get_thread_num();
        auto line = batch.GetLine(i);
        for (auto j = 0ull; j < line.Size(); j++) {
          auto element = line.GetElement(j);
          mat.info.num_col_ =
              std::max(mat.info.num_col_, static_cast<uint64_t>(element.column_idx+ 1));
          if (!common::CheckNAN(element.value) && element.value != missing) {
            builder.AddBudget(element.row_idx, tid);
          }
        }
      }
      builder.InitStorage();

      // Second pass over batch, placing elements in correct position
#pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < static_cast<omp_ulong>(num_lines);
        ++i) {  // NOLINT(*)
        int tid = omp_get_thread_num();
        auto line = batch.GetLine(i);
        for (auto j = 0ull; j < line.Size(); j++) {
          auto element = line.GetElement(j);
          if (!common::CheckNAN(element.value) && element.value != missing) {
            builder.Push(element.row_idx, Entry(element.column_idx, element.value),
              tid);
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
    }
    mat.info.num_row_ = mat.page_.offset.Size() - 1;
    mat.info.num_nonzero_ = offset_vec.back();
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
