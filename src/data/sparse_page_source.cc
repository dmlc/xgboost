/*!
 *  Copyright (c) 2020 by XGBoost Contributors
 */
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
void DataPool::Slice(std::shared_ptr<SparsePage> out, size_t offset,
                     size_t n_rows, size_t entry_offset) const {
  auto const &in_offset = pool_.offset.HostVector();
  auto const &in_data = pool_.data.HostVector();
  auto &h_offset = out->offset.HostVector();
  CHECK_LE(offset + n_rows + 1, in_offset.size());
  h_offset.resize(n_rows + 1, 0);
  std::transform(in_offset.cbegin() + offset,
                 in_offset.cbegin() + offset + n_rows + 1, h_offset.begin(),
                 [=](size_t ptr) { return ptr - entry_offset; });

  auto &h_data = out->data.HostVector();
  CHECK_GT(h_offset.size(), 0);
  size_t n_entries = h_offset.back();
  h_data.resize(n_entries);

  CHECK_EQ(n_entries, in_offset.at(offset + n_rows) - in_offset.at(offset));
  std::copy_n(in_data.cbegin() + in_offset.at(offset), n_entries,
              h_data.begin());
}

void DataPool::SplitWritePage() {
  size_t total = pool_.Size();
  size_t offset = 0;
  size_t entry_offset = 0;
  do {
    size_t n_rows = std::min(page_size_, total - offset);
    std::shared_ptr<SparsePage> out;
    writer_->Alloc(&out);
    out->Clear();
    out->SetBaseRowId(inferred_num_rows_);
    this->Slice(out, offset, n_rows, entry_offset);
    inferred_num_rows_ += out->Size();
    offset += n_rows;
    entry_offset += out->data.Size();
    writer_->PushWrite(std::move(out));
  } while (total - offset >= page_size_);

  auto out = std::make_shared<SparsePage>();
  this->Slice(out, offset, total - offset, entry_offset);
  pool_.Clear();
  pool_.Push(*out);
}
}  // namespace data
}  // namespace xgboost
