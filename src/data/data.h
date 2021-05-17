/*!
 * Copyright (c) 2015-2021 by Contributors
 * \file data.h
 * \brief Dispatching for input data.
 */
#include <xgboost/data.h>
#include "simple_dmatrix.h"
#include "sparse_page_dmatrix.h"
#include "../common/group_data.h"

namespace xgboost {
template <typename AdapterT>
DMatrix* DMatrix::Create(AdapterT* adapter, float missing, int nthread,
                         const std::string& cache_prefix,  size_t page_size) {
  if (cache_prefix.length() == 0) {
    // Data split mode is fixed to be row right now.
    return data::SimpleDMatrix::FromCPUData(adapter, missing, nthread);
  } else {
#if DMLC_ENABLE_STD_THREAD
    return new data::SparsePageDMatrix(adapter, missing, nthread, cache_prefix,
                                       page_size);
#else
    LOG(FATAL) << "External memory is not enabled in mingw";
    return nullptr;
#endif  // DMLC_ENABLE_STD_THREAD
  }
}

template <typename AdapterBatchT>
uint64_t SparsePage::Push(const AdapterBatchT& batch, float missing, int nthread) {
  constexpr bool kIsRowMajor = AdapterBatchT::kIsRowMajor;
  // Allow threading only for row-major case as column-major requires O(nthread*batch_size) memory
  nthread = kIsRowMajor ? nthread : 1;
  // Set number of threads but keep old value so we can reset it after
  int nthread_original = common::OmpSetNumThreadsWithoutHT(&nthread);
  if (!kIsRowMajor) {
    CHECK_EQ(nthread, 1);
  }
  auto& offset_vec = offset.HostVector();
  auto& data_vec = data.HostVector();

  size_t builder_base_row_offset = this->Size();
  common::ParallelGroupBuilder<
      Entry, std::remove_reference<decltype(offset_vec)>::type::value_type, kIsRowMajor>
      builder(&offset_vec, &data_vec, builder_base_row_offset);
  // Estimate expected number of rows by using last element in batch
  // This is not required to be exact but prevents unnecessary resizing
  size_t expected_rows = 0;
  if (batch.Size() > 0) {
    auto last_line = batch.GetLine(batch.Size() - 1);
    if (last_line.Size() > 0) {
      expected_rows =
          last_line.GetElement(last_line.Size() - 1).row_idx - base_rowid;
    }
  }
  size_t batch_size = batch.Size();
  expected_rows = kIsRowMajor ? batch_size : expected_rows;
  uint64_t max_columns = 0;
  if (batch_size == 0) {
    omp_set_num_threads(nthread_original);
    return max_columns;
  }
  const size_t thread_size = batch_size / nthread;

  builder.InitBudget(expected_rows, nthread);
  std::vector<std::vector<uint64_t>> max_columns_vector(nthread);
  dmlc::OMPException exec;
  std::atomic<bool> valid{true};
  // First-pass over the batch counting valid elements
#pragma omp parallel num_threads(nthread)
  {
    exec.Run([&]() {
      int tid = omp_get_thread_num();
      size_t begin = tid*thread_size;
      size_t end = tid != (nthread-1) ? (tid+1)*thread_size : batch_size;
      max_columns_vector[tid].resize(1, 0);
      uint64_t& max_columns_local = max_columns_vector[tid][0];

      for (size_t i = begin; i < end; ++i) {
        auto line = batch.GetLine(i);
        for (auto j = 0ull; j < line.Size(); j++) {
          data::COOTuple const& element = line.GetElement(j);
          if (!std::isinf(missing) && std::isinf(element.value)) {
            valid = false;
          }
          const size_t key = element.row_idx - base_rowid;
          CHECK_GE(key,  builder_base_row_offset);
          max_columns_local =
              std::max(max_columns_local, static_cast<uint64_t>(element.column_idx + 1));

          if (!common::CheckNAN(element.value) && element.value != missing) {
            // Adapter row index is absolute, here we want it relative to
            // current page
            builder.AddBudget(key, tid);
          }
        }
      }
    });
  }
  exec.Rethrow();
  CHECK(valid) << "Input data contains `inf` or `nan`";
  for (const auto & max : max_columns_vector) {
    max_columns = std::max(max_columns, max[0]);
  }

  builder.InitStorage();

  // Second pass over batch, placing elements in correct position

#pragma omp parallel num_threads(nthread)
  {
    exec.Run([&]() {
      int tid = omp_get_thread_num();
      size_t begin = tid*thread_size;
      size_t end = tid != (nthread-1) ? (tid+1)*thread_size : batch_size;
      for (size_t i = begin; i < end; ++i) {
        auto line = batch.GetLine(i);
        for (auto j = 0ull; j < line.Size(); j++) {
          auto element = line.GetElement(j);
          const size_t key = (element.row_idx - base_rowid);
          if (!common::CheckNAN(element.value) && element.value != missing) {
            builder.Push(key, Entry(element.column_idx, element.value), tid);
          }
        }
      }
    });
  }
  exec.Rethrow();
  omp_set_num_threads(nthread_original);

  return max_columns;
}
}  // namespace xgboost
