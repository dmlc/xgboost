/*!
 *  Copyright (c) 2020 by XGBoost Contributors
 */
#include "adapter.h"
#include "xgboost/data.h"

namespace xgboost {
namespace data {

template <typename T> auto CopyMeta(std::vector<T> *p_out, std::vector<T> const &in,
                                    size_t offset, size_t n_rows) {
  auto &out = *p_out;
  out.clear();
  if (!in.empty()) {
    out.resize(n_rows);
    std::copy_n(in.begin() + offset, n_rows, out.begin());
  }
}

void FileAdapter::Block::CopySlice(Block const &that, size_t n_rows, size_t offset) {
  auto &in_offset = that.offset;
  auto &in_data = that.value;
  auto& in_index = that.index;

  size_t entry_offset = *(in_offset.cbegin() + offset);

  Block &out = *this;
  auto &h_offset = out.offset;
  CHECK_LE(offset + n_rows + 1, in_offset.size());
  h_offset.resize(n_rows + 1, 0);
  std::transform(in_offset.cbegin() + offset, in_offset.cbegin() + offset + n_rows + 1,
                 h_offset.begin(), [=](size_t ptr) { return ptr - entry_offset; });

  auto &h_data = out.value;
  CHECK_GT(h_offset.size(), 0);
  size_t n_entries = h_offset.back();
  h_data.resize(n_entries);
  auto& h_index = out.index;
  h_index.resize(n_entries);

  CHECK_EQ(n_entries, in_offset.at(offset + n_rows) - in_offset.at(offset));
  std::copy_n(in_data.cbegin() + in_offset.at(offset), n_entries, h_data.begin());
  std::copy_n(in_index.cbegin() + in_offset.at(offset), n_entries, h_index.begin());

  CopyMeta(&label, that.label, offset, n_rows);
  CopyMeta(&weight, that.weight, offset, n_rows);
  CopyMeta(&qid, that.qid, offset, n_rows);
  CopyMeta(&field, that.field, offset, n_rows);
}

dmlc::RowBlock<uint32_t> FileAdapter::DataPool::Value() {
  if (Full()) {
    size_t i = page_size_;
    staging_.Clear();
    staging_.CopySlice(block_, i, 0);
    CHECK_EQ(staging_.Size(), page_size_);
    Block remaining;
    remaining.CopySlice(block_, block_.Size() - i, i);
    block_ = std::move(remaining);
  } else {
    staging_ = std::move(block_);
    CHECK(!staging_.value.empty());
    CHECK(!staging_.index.empty());
    block_.Clear();
  }
  return dmlc::RowBlock<uint32_t>(staging_);  // NOLINT
}

bool FileAdapter::DataPool::Push(dmlc::RowBlock<uint32_t> const *that) {
  CHECK_EQ(that->offset[0], 0);
  CHECK(!block_.offset.empty());

  auto before_push = block_.Size();

  size_t back = block_.offset.size();
  CHECK_NE(that->size, 0);
  size_t last = block_.offset.back();
  block_.offset.resize(that->size + back);
  std::transform(that->offset + 1, that->offset + 1 + that->size,
                 block_.offset.begin() + back, [last](size_t ptr) { return ptr + last; });
  if (that->weight) {
    back = block_.weight.size();
    block_.weight.resize(that->size + back);
    std::copy_n(that->weight, that->size, this->block_.weight.begin() + back);
  }
  if (that->qid) {
    back = block_.qid.size();
    block_.qid.resize(that->size + back);
    std::copy_n(that->qid, that->size, this->block_.qid.begin() + back);
  }
  if (that->label) {
    back = block_.label.size();
    block_.label.resize(that->size + back);
    std::copy_n(that->label, that->size, this->block_.label.begin() + back);
  }
  CHECK(!that->field);

  CHECK(that->value || that->size == 0);
  size_t n_entries = that->offset[that->size];
  back = block_.value.size();
  block_.value.resize(n_entries + back);
  std::copy_n(that->value, n_entries, this->block_.value.begin() + back);
  block_.index.resize(n_entries + back);
  std::copy_n(that->index, n_entries, this->block_.index.begin() + back);

  auto after_push = block_.Size();
  CHECK_GT(after_push, before_push);
  return Full();
}

bool FileAdapter::Next() {
  if (pool_.Full()) {
    auto block = pool_.Value();
    batch_.reset(new FileAdapterBatch(block, row_offset_));
    row_offset_ += block.size;
    return true;
  }

  bool next = false;
  while ((next = parser_->Next()) && !(pool_.Push(&parser_->Value()))) {
  }

  // Must return before calling Value, which moves data to staging.
  if (pool_.Available() == 0) {
    return false;
  }

  auto block = pool_.Value();
  CHECK_NE(block.size, 0);
  CHECK(block.value) << block.size;
  CHECK(block.index);

  batch_.reset(new FileAdapterBatch(block, row_offset_));
  row_offset_ += block.size;
  CHECK_NE(batch_->Size(), 0);
  return true;
}
}  // namespace data
}  // namespace xgboost
