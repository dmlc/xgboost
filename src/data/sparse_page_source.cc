/*!
 *  Copyright (c) 2020 by XGBoost Contributors
 */
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
void SplitWritePage(std::shared_ptr<SparsePage> page, size_t page_size,
                    SparsePageWriter<SparsePage> *writer, size_t *rows,
                    MetaInfo *info, bool last) {
  auto out = std::make_shared<SparsePage>();
  auto& inferred_num_rows = *rows;
  size_t total = page->Size();

  out->Clear();
  out->SetBaseRowId(page->base_rowid);
  auto const& in_offset = page->offset.HostVector();
  auto const& in_data = page->data.HostVector();

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

    // Don't allocate unnecessary page, otherwise the concurrent queue might run into dead
    // lock.
    if (page_id != n_pages - 1) {
      info->num_nonzero_ += h_data.size();
      inferred_num_rows += out->Size();

      std::shared_ptr<SparsePage> tmp;
      writer->Alloc(&tmp);
      tmp->Push(std::move(*out));
      writer->PushWrite(std::move(tmp));

      out->Clear();
      out->SetBaseRowId(inferred_num_rows);
    } else {
      if (last) {
        inferred_num_rows += out->Size();
        info->num_nonzero_ += h_data.size();

        std::shared_ptr<SparsePage> tmp;
        writer->Alloc(&tmp);
        tmp->Push(std::move(*out));
        writer->PushWrite(std::move(out));
      } else {
        // Return the remaining entries.
        page->Clear();
        page->Push(*out);
        page->SetBaseRowId(inferred_num_rows);
      }
    }

    rows_written += n_rows;
  }
  CHECK_EQ(rows_written, total);
}

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
