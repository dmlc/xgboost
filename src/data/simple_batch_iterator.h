/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_DATA_SIMPLE_BATCH_ITERATOR_H_
#define XGBOOST_DATA_SIMPLE_BATCH_ITERATOR_H_

#include <xgboost/data.h>

namespace xgboost {
namespace data {

template<typename T>
class SimpleBatchIteratorImpl : public BatchIteratorImpl<T> {
 public:
  explicit SimpleBatchIteratorImpl(T* page) : page_(page) {}
  T& operator*() override {
    CHECK(page_ != nullptr);
    return *page_;
  }
  const T& operator*() const override {
    CHECK(page_ != nullptr);
    return *page_;
  }
  void operator++() override { page_ = nullptr; }
  bool AtEnd() const override { return page_ == nullptr; }

 private:
  T* page_{nullptr};
};

}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_BATCH_ITERATOR_H_
