/*!
 * Copyright 2014~2020 by Contributors
 * \file simple_dmatrix.cc
 * \brief the input data structure for gradient boosting
 * \author Tianqi Chen
 */
#include <vector>
#include <limits>
#include <type_traits>
#include <algorithm>

#include "xgboost/data.h"
#include "xgboost/c_api.h"

#include "simple_dmatrix.h"
#include "./simple_batch_iterator.h"
#include "../common/random.h"
#include "../common/threading_utils.h"
#include "adapter.h"

namespace xgboost {
namespace data {
MetaInfo& SimpleDMatrix::Info() { return info_; }

const MetaInfo& SimpleDMatrix::Info() const { return info_; }

DMatrix* SimpleDMatrix::Slice(common::Span<int32_t const> ridxs) {
  auto out = new SimpleDMatrix;
  SparsePage& out_page = out->sparse_page_;
  for (auto const &page : this->GetBatches<SparsePage>()) {
    page.data.HostVector();
    page.offset.HostVector();
    auto& h_data = out_page.data.HostVector();
    auto& h_offset = out_page.offset.HostVector();
    size_t rptr{0};
    for (auto ridx : ridxs) {
      auto inst = page[ridx];
      rptr += inst.size();
      std::copy(inst.begin(), inst.end(), std::back_inserter(h_data));
      h_offset.emplace_back(rptr);
    }
    out->Info() = this->Info().Slice(ridxs);
    out->Info().num_nonzero_ = h_offset.back();
  }
  return out;
}

BatchSet<SparsePage> SimpleDMatrix::GetRowBatches() {
  // since csr is the default data structure so `source_` is always available.
  auto begin_iter = BatchIterator<SparsePage>(
      new SimpleBatchIteratorImpl<SparsePage>(&sparse_page_));
  return BatchSet<SparsePage>(begin_iter);
}

BatchSet<CSCPage> SimpleDMatrix::GetColumnBatches() {
  // column page doesn't exist, generate it
  if (!column_page_) {
    column_page_.reset(new CSCPage(sparse_page_.GetTranspose(info_.num_col_)));
  }
  auto begin_iter =
      BatchIterator<CSCPage>(new SimpleBatchIteratorImpl<CSCPage>(column_page_.get()));
  return BatchSet<CSCPage>(begin_iter);
}

BatchSet<SortedCSCPage> SimpleDMatrix::GetSortedColumnBatches() {
  // Sorted column page doesn't exist, generate it
  if (!sorted_column_page_) {
    sorted_column_page_.reset(
        new SortedCSCPage(sparse_page_.GetTranspose(info_.num_col_)));
    sorted_column_page_->SortRows();
  }
  auto begin_iter = BatchIterator<SortedCSCPage>(
      new SimpleBatchIteratorImpl<SortedCSCPage>(sorted_column_page_.get()));
  return BatchSet<SortedCSCPage>(begin_iter);
}

BatchSet<EllpackPage> SimpleDMatrix::GetEllpackBatches(const BatchParam& param) {
  // ELLPACK page doesn't exist, generate it
  if (!(batch_param_ != BatchParam{})) {
    CHECK(param != BatchParam{}) << "Batch parameter is not initialized.";
  }
  if (!ellpack_page_  || (batch_param_ != param && param != BatchParam{})) {
    CHECK_GE(param.gpu_id, 0);
    CHECK_GE(param.max_bin, 2);
    ellpack_page_.reset(new EllpackPage(this, param));
    batch_param_ = param;
  }
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_page_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}

template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, int nthread) {
  // Set number of threads but keep old value so we can reset it after
  int nthread_original = common::OmpSetNumThreadsWithoutHT(&nthread);

  std::vector<uint64_t> qids;
  uint64_t default_max = std::numeric_limits<uint64_t>::max();
  uint64_t last_group_id = default_max;
  bst_uint group_size = 0;
  auto& offset_vec = sparse_page_.offset.HostVector();
  auto& data_vec = sparse_page_.data.HostVector();
  uint64_t inferred_num_columns = 0;
  uint64_t total_batch_size = 0;
    // batch_size is either number of rows or cols, depending on data layout

  adapter->BeforeFirst();
  // Iterate over batches of input data
  while (adapter->Next()) {
    auto& batch = adapter->Value();
    auto batch_max_columns = sparse_page_.Push(batch, missing, nthread);
    inferred_num_columns = std::max(batch_max_columns, inferred_num_columns);
    total_batch_size += batch.Size();
    // Append meta information if available
    if (batch.Labels() != nullptr) {
      auto& labels = info_.labels_.HostVector();
      labels.insert(labels.end(), batch.Labels(),
                    batch.Labels() + batch.Size());
    }
    if (batch.Weights() != nullptr) {
      auto& weights = info_.weights_.HostVector();
      weights.insert(weights.end(), batch.Weights(),
                     batch.Weights() + batch.Size());
    }
    if (batch.BaseMargin() != nullptr) {
      auto& base_margin = info_.base_margin_.HostVector();
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
    using IteratorAdapterT
      = IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>;
    // If AdapterT is either IteratorAdapter or FileAdapter type, use the total batch size to
    // determine the correct number of rows, as offset_vec may be too short
    if (std::is_same<AdapterT, IteratorAdapterT>::value
        || std::is_same<AdapterT, FileAdapter>::value) {
      info_.num_row_ = total_batch_size;
      // Ensure offset_vec.size() - 1 == [number of rows]
      while (offset_vec.size() - 1 < total_batch_size) {
        offset_vec.emplace_back(offset_vec.back());
      }
    } else {
      CHECK((std::is_same<AdapterT, CSCAdapter>::value)) << "Expecting CSCAdapter";
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
  omp_set_num_threads(nthread_original);
}

SimpleDMatrix::SimpleDMatrix(dmlc::Stream* in_stream) {
  int tmagic;
  CHECK(in_stream->Read(&tmagic)) << "invalid input file format";
  CHECK_EQ(tmagic, kMagic) << "invalid format, magic number mismatch";
  info_.LoadBinary(in_stream);
  in_stream->Read(&sparse_page_.offset.HostVector());
  in_stream->Read(&sparse_page_.data.HostVector());
}

void SimpleDMatrix::SaveToLocalFile(const std::string& fname) {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    int tmagic = kMagic;
    fo->Write(tmagic);
    info_.SaveBinary(fo.get());
    fo->Write(sparse_page_.offset.HostVector());
    fo->Write(sparse_page_.data.HostVector());
}

template SimpleDMatrix::SimpleDMatrix(DenseAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(CSRAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(CSCAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(DataTableAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(FileAdapter* adapter, float missing,
                                     int nthread);
template SimpleDMatrix::SimpleDMatrix(
    IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>
        *adapter,
    float missing, int nthread);
}  // namespace data
}  // namespace xgboost
