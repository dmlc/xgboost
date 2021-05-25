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

#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

#include "rabit/rabit.h"
#include "adapter.h"

namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
  template <typename AdapterT>
  explicit SimpleDMatrix(AdapterT *adapter, float missing, int nthread) {
    std::vector<uint64_t> qids;
    uint64_t default_max = std::numeric_limits<uint64_t>::max();
    uint64_t last_group_id = default_max;
    bst_uint group_size = 0;
    auto &offset_vec = sparse_page_.offset.HostVector();
    auto &data_vec = sparse_page_.data.HostVector();
    uint64_t inferred_num_columns = 0;
    uint64_t total_batch_size = 0;
    // batch_size is either number of rows or cols, depending on data layout

    adapter->BeforeFirst();
    // Iterate over batches of input data
    while (adapter->Next()) {
      auto &batch = adapter->Value();
      auto batch_max_columns = sparse_page_.Push(batch, missing, nthread);
      inferred_num_columns = std::max(batch_max_columns, inferred_num_columns);
      total_batch_size += batch.Size();
      // Append meta information if available
      if (batch.Labels() != nullptr) {
        auto &labels = info_.labels_.HostVector();
        labels.insert(labels.end(), batch.Labels(),
                      batch.Labels() + batch.Size());
      }
      if (batch.Weights() != nullptr) {
        auto &weights = info_.weights_.HostVector();
        weights.insert(weights.end(), batch.Weights(),
                       batch.Weights() + batch.Size());
      }
      if (batch.BaseMargin() != nullptr) {
        auto &base_margin = info_.base_margin_.HostVector();
        base_margin.insert(base_margin.end(), batch.BaseMargin(),
                           batch.BaseMargin() + batch.Size());
      }
      if (batch.Qid() != nullptr) {
        qids.insert(qids.end(), batch.Qid(), batch.Qid() + batch.Size());
        // get group
        for (size_t i = 0; i < batch.Size(); ++i) {
          const uint64_t cur_group_id = batch.Qid()[i];
          if (last_group_id == default_max || last_group_id != cur_group_id) {
            info_.group_ptr_.push_back(group_size);
          }
          last_group_id = cur_group_id;
          ++group_size;
        }
      }
    }

    if (last_group_id != default_max) {
      if (group_size > info_.group_ptr_.back()) {
        info_.group_ptr_.push_back(group_size);
      }
    }

    // Deal with empty rows/columns if necessary
    if (adapter->NumColumns() == kAdapterUnknownSize) {
      info_.num_col_ = inferred_num_columns;
    } else {
      info_.num_col_ = adapter->NumColumns();
    }

    // Synchronise worker columns
    rabit::Allreduce<rabit::op::Max>(&info_.num_col_, 1);

    if (adapter->NumRows() == kAdapterUnknownSize) {
      using IteratorAdapterT =
          IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext,
                          XGBoostBatchCSR>;
      // If AdapterT is either IteratorAdapter or FileAdapter type, use the
      // total batch size to determine the correct number of rows, as offset_vec
      // may be too short
      if (std::is_same<AdapterT, IteratorAdapterT>::value ||
          std::is_same<AdapterT, FileAdapter>::value) {
        info_.num_row_ = total_batch_size;
        // Ensure offset_vec.size() - 1 == [number of rows]
        while (offset_vec.size() - 1 < total_batch_size) {
          offset_vec.emplace_back(offset_vec.back());
        }
      } else {
        CHECK((std::is_same<AdapterT, CSCAdapter>::value))
            << "Expecting CSCAdapter";
        info_.num_row_ = offset_vec.size() - 1;
      }
    } else {
      if (offset_vec.empty()) {
        offset_vec.emplace_back(0);
      }
      while (offset_vec.size() - 1 < adapter->NumRows()) {
        offset_vec.emplace_back(offset_vec.back());
      }
      info_.num_row_ = adapter->NumRows();
    }
    info_.num_nonzero_ = data_vec.size();
  }

  template <typename AdapterT>
  void LoadFromGPU(AdapterT* adapter, float missing, int nthread);

 public:
  SimpleDMatrix() = default;

  template <typename AdapterT>
  static SimpleDMatrix* FromGPUData(AdapterT* adapter, float missing, int nthread) {
    auto x = new SimpleDMatrix();
    x->LoadFromGPU(adapter, missing, nthread);
    return x;
  }

  template <typename AdapterT>
  static SimpleDMatrix* FromCPUData(AdapterT* adapter, float missing, int nthread) {
    return new SimpleDMatrix(adapter, missing, nthread);
  }

  explicit SimpleDMatrix(dmlc::Stream* in_stream);
  ~SimpleDMatrix() override = default;

  void SaveToLocalFile(const std::string& fname);

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  bool SingleColBlock() const override { return true; }
  DMatrix* Slice(common::Span<int32_t const> ridxs) override;

  /*! \brief magic number used to identify SimpleDMatrix binary files */
  static const int kMagic = 0xffffab01;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  MetaInfo info_;
  SparsePage sparse_page_;  // Primary storage type
  std::unique_ptr<CSCPage> column_page_;
  std::unique_ptr<SortedCSCPage> sorted_column_page_;
  std::unique_ptr<EllpackPage> ellpack_page_;
  BatchParam batch_param_;

  bool EllpackExists() const override {
    return static_cast<bool>(ellpack_page_);
  }
  bool SparsePageExists() const override {
    return true;
  }
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
