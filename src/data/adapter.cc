/*!
 *  Copyright (c) 2020 by XGBoost Contributors
 */
#include "adapter.h"

namespace xgboost {
namespace data {
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
}  // namespace data
}  // namespace xgboost
