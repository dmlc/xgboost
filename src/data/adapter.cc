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
  size_t entry_offset = 0;

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

  CHECK_EQ(n_entries, in_offset.at(offset + n_rows) - in_offset.at(offset));
  std::copy_n(in_data.cbegin() + in_offset.at(offset), n_entries, h_data.begin());

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
    Block remaining;
    remaining.CopySlice(block_, block_.Size() - i, i);
    block_ = std::move(remaining);
  } else {
    staging_ = std::move(block_);
    block_.Clear();
  }
  return dmlc::RowBlock<uint32_t>(staging_);  // NOLINT
}

bool FileAdapter::DataPool::Push(dmlc::RowBlock<uint32_t> const *block) {
  CHECK_EQ(block->offset[0], 0);
  size_t back = block_.offset.size();
  block_.offset.resize(block->size + back);
  std::copy_n(block->offset + 1, block->size, this->block_.offset.begin() + back);

  if (block->weight) {
    back = block_.weight.size();
    std::copy_n(block->weight, block->size, this->block_.weight.begin() + back);
  }
  if (block->qid) {
    back = block_.qid.size();
    std::copy_n(block->qid, block->size, this->block_.qid.begin() + back);
  }
  if (block->field) {
    back = block_.field.size();
    std::copy_n(block->field, block->size, this->block_.field.begin() + back);
  }
  if (block->index) {
    back = block_.index.size();
    std::copy_n(block->index, block->size, this->block_.index.begin() + back);
  }
  if (block->value) {
    back = block_.value.size();
    std::copy_n(block->value, block->size, this->block_.value.begin() + back);
  }

  CHECK(block->value || block->size == 0);
  back = block_.value.size();
  block_.value.resize(block->size + back);
  std::copy_n(block->value, block->size, this->block_.value.begin() + back);
  return Full();
}

bool FileAdapter::Next() {
  if (pool_.Full()) {
    auto block = pool_.Value();
    batch_.reset(new FileAdapterBatch(block, row_offset_));
    return true;
  }
  bool next = false;
  while ((next = parser_->Next()) && pool_.Push(&parser_->Value())) {
  }
  auto block = pool_.Value();
  batch_.reset(new FileAdapterBatch(block, row_offset_));
  row_offset_ += block.size;
  return next;
}
}  // namespace data
}  // namespace xgboost
