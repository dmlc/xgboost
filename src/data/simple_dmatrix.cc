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
    auto batch = page.GetView();
    auto& h_data = out_page.data.HostVector();
    auto& h_offset = out_page.offset.HostVector();
    size_t rptr{0};
    for (auto ridx : ridxs) {
      auto inst = batch[ridx];
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
}  // namespace data
}  // namespace xgboost
