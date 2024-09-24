/**
 * Copyright 2019-2024, XGBoost contributors
 */
#ifndef XGBOOST_DATA_SIMPLE_BATCH_ITERATOR_H_
#define XGBOOST_DATA_SIMPLE_BATCH_ITERATOR_H_

#include <memory>   // for shared_ptr
#include <utility>  // for move

#include "xgboost/data.h"  // for BatchIteratorImpl

namespace xgboost::data {
template <typename T>
class SimpleBatchIteratorImpl : public BatchIteratorImpl<T> {
 public:
  explicit SimpleBatchIteratorImpl(std::shared_ptr<T const> page) : page_(std::move(page)) {}
  const T& operator*() const override {
    CHECK(page_ != nullptr);
    return *page_;
  }
  SimpleBatchIteratorImpl& operator++() override {
    page_ = nullptr;
    return *this;
  }
  bool AtEnd() const override { return page_ == nullptr; }

  std::shared_ptr<T const> Page() const override { return page_; }

 private:
  std::shared_ptr<T const> page_{nullptr};
};
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SIMPLE_BATCH_ITERATOR_H_
