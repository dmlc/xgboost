/*!
 *  Copyright (c) 2020 by XGBoost Contributors
 */
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
void SplitWritePage(SparsePage const &page, size_t page_size,
               SparsePageWriter<SparsePage> *writer, size_t *rows,
               MetaInfo *info) {
  std::shared_ptr<SparsePage> out;
  auto& inferred_num_rows = *rows;
  size_t total = page.Size();

  writer->Alloc(&out);
  out->Clear();
  out->SetBaseRowId(page.base_rowid);
  auto const& in_offset = page.offset.HostVector();
  auto const& in_data = page.data.HostVector();

  size_t n_pages = common::DivRoundUp(total, page_size);
  size_t offset = 0;
  size_t entry_offset = 0;

  size_t rows_written = 0;
  for (size_t page_id = 0; page_id < n_pages; ++page_id) {
    size_t n_rows = std::min(page_size, total - offset);
    auto& h_offset = out->offset.HostVector();
    CHECK_LE(offset + n_rows + 1, in_offset.size());
    h_offset.resize(n_rows + 1, 0);
    std::transform(in_offset.cbegin() + offset,
                   in_offset.cbegin() + offset + n_rows + 1, h_offset.begin(),
                   [=](size_t ptr) { return ptr - entry_offset; });

    auto& h_data = out->data.HostVector();
    CHECK_GT(h_offset.size(), 0);
    size_t n_entries = h_offset.back();
    h_data.resize(n_entries);

    CHECK_EQ(n_entries, in_offset.at(offset + n_rows) - in_offset.at(offset));
    std::copy_n(in_data.cbegin() + in_offset.at(offset), n_entries, h_data.begin());

    offset += n_rows;
    entry_offset += n_entries;

    info->num_nonzero_ += h_data.size();
    inferred_num_rows += out->Size();

    writer->PushWrite(std::move(out));

    // Don't allocate unnecessary page, otherwise the concurrent queue might run into dead
    // lock.
    if (page_id != n_pages - 1) {
      writer->Alloc(&out);
      out->Clear();
      out->SetBaseRowId(inferred_num_rows);
    }

    rows_written += n_rows;
  }
  CHECK_EQ(rows_written, total);
}
}  // namespace data
}  // namespace xgboost
