/*!
 * Copyright 2018 XGBoost contributors
 */
#ifndef XGBOOST_TEST_SPAN_H_
#define XGBOOST_TEST_SPAN_H_

#include "../../include/xgboost/base.h"
#include "../../../src/common/span.h"

template <typename Iter>
XGBOOST_DEVICE void InitializeRange(Iter _begin, Iter _end) {
  float j = 0;
  for (Iter i = _begin; i != _end; ++i, ++j) {
    *i = j;
  }
}

namespace xgboost {
namespace common {

#define SPAN_ASSERT_TRUE(cond, status)          \
  if (!(cond)) {                                \
    *(status) = -1;                             \
  }

#define SPAN_ASSERT_FALSE(cond, status)         \
  if ((cond)) {                                 \
    *(status) = -1;                             \
  }

struct TestTestStatus {
  int * status_;

  TestTestStatus(int* _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    SPAN_ASSERT_TRUE(false, status_);
  }
};

struct TestAssignment {
  int* status_;

  TestAssignment(int* _status) : status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    Span<float> s1;

    float arr[] = {3, 4, 5};

    Span<const float> s2 = arr;
    SPAN_ASSERT_TRUE(s2.size() == 3, status_);
    SPAN_ASSERT_TRUE(s2.data() == &arr[0], status_);

    s2 = s1;
    SPAN_ASSERT_TRUE(s2.empty(), status_);
  }
};

struct TestBeginEnd {
  int* status_;

  TestBeginEnd(int* _status) : status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);

    Span<float> s (arr);
    Span<float>::iterator beg { s.begin() };
    Span<float>::iterator end { s.end() };

    SPAN_ASSERT_TRUE(end ==  beg + 16, status_);
    SPAN_ASSERT_TRUE(*beg == arr[0], status_);
    SPAN_ASSERT_TRUE(*(end - 1) == arr[15], status_);
  }
};

struct TestRBeginREnd {
  int * status_;

  TestRBeginREnd(int* _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);

    Span<float> s (arr);
    Span<float>::iterator rbeg { s.rbegin() };
    Span<float>::iterator rend { s.rend() };

    SPAN_ASSERT_TRUE(rbeg == rend + 16, status_);
    SPAN_ASSERT_TRUE(*(rbeg - 1) == arr[15], status_);
    SPAN_ASSERT_TRUE(*rend == arr[0], status_);
  }
};

struct TestObservers {
  int * status_;

  TestObservers(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    // empty
    {
      float *arr = nullptr;
      Span<float> s(arr, static_cast<Span<float>::index_type>(0));
      SPAN_ASSERT_TRUE(s.empty(), status_);
    }

    // size, size_types
    {
      float* arr = new float[16];
      Span<float> s (arr, 16);
      SPAN_ASSERT_TRUE(s.size() == 16, status_);
      SPAN_ASSERT_TRUE(s.size_bytes() == 16 * sizeof(float), status_);
      delete [] arr;
    }
  }
};

struct TestCompare {
  int * status_;

  TestCompare(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float lhs_arr[16], rhs_arr[16];
    InitializeRange(lhs_arr, lhs_arr + 16);
    InitializeRange(rhs_arr, rhs_arr + 16);

    Span<float> lhs(lhs_arr);
    Span<float> rhs(rhs_arr);

    SPAN_ASSERT_TRUE(lhs == rhs, status_);
    SPAN_ASSERT_FALSE(lhs != rhs, status_);

    SPAN_ASSERT_TRUE(lhs <= rhs, status_);
    SPAN_ASSERT_TRUE(lhs >= rhs, status_);

    lhs[2] -= 1;

    SPAN_ASSERT_FALSE(lhs == rhs, status_);
    SPAN_ASSERT_TRUE(lhs < rhs, status_);
    SPAN_ASSERT_FALSE(lhs > rhs, status_);
  }
};

struct TestIterConstruct {
  int * status_;

  TestIterConstruct(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    Span<float>::iterator it1;
    Span<float>::iterator it2;
    SPAN_ASSERT_TRUE(it1 == it2, status_);

    Span<float>::const_iterator cit1;
    Span<float>::const_iterator cit2;
    SPAN_ASSERT_TRUE(cit1 == cit2, status_);
  }
};

struct TestIterRef {
  int * status_;

  TestIterRef(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);

    Span<float> s (arr);
    SPAN_ASSERT_TRUE(*(s.begin()) == s[0], status_);
    SPAN_ASSERT_TRUE(*(s.end() - 1) == s[15], status_);
  }
};

struct TestIterCalculate {
  int * status_;

  TestIterCalculate(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);

    Span<float> s (arr);
    Span<float>::iterator beg { s.begin() };

    beg += 4;
    SPAN_ASSERT_TRUE(*beg == 4, status_);

    beg -= 2;
    SPAN_ASSERT_TRUE(*beg == 2, status_);

    ++beg;
    SPAN_ASSERT_TRUE(*beg == 3, status_);

    --beg;
    SPAN_ASSERT_TRUE(*beg == 2, status_);

    beg++;
    beg--;
    SPAN_ASSERT_TRUE(*beg == 2, status_);
  }
};

struct TestIterCompare {
  int * status_;

  TestIterCompare(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);
    Span<float> s (arr);
    Span<float>::iterator left { s.begin() };
    Span<float>::iterator right { s.end() };

    left += 1;
    right -= 15;

    SPAN_ASSERT_TRUE(left == right, status_);

    SPAN_ASSERT_TRUE(left >= right, status_);
    SPAN_ASSERT_TRUE(left <= right, status_);

    ++right;
    SPAN_ASSERT_TRUE(right > left, status_);
    SPAN_ASSERT_TRUE(left < right, status_);
    SPAN_ASSERT_TRUE(left <= right, status_);
  }
};

struct TestAsBytes {
  int * status_;

  TestAsBytes(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);

    {
      const Span<const float> s {arr};
      const Span<const byte> bs = as_bytes(s);
      SPAN_ASSERT_TRUE(bs.size() == s.size_bytes(), status_);
      SPAN_ASSERT_TRUE(static_cast<const void*>(bs.data()) ==
                       static_cast<const void*>(s.data()),
                       status_);
    }

    {
      Span<float> s;
      const Span<const byte> bs = as_bytes(s);
      SPAN_ASSERT_TRUE(bs.size() == s.size(), status_);
      SPAN_ASSERT_TRUE(bs.size() == 0, status_);
      SPAN_ASSERT_TRUE(bs.size_bytes() == 0, status_);
      SPAN_ASSERT_TRUE(static_cast<const void*>(bs.data()) ==
                       static_cast<const void*>(s.data()),
                       status_);
      SPAN_ASSERT_TRUE(bs.data() == nullptr, status_);
    }
  }
};

struct TestAsWritableBytes {
  int * status_;

  TestAsWritableBytes(int * _status): status_(_status) {}

  XGBOOST_DEVICE void operator()() {
    this->operator()(0);
  }
  XGBOOST_DEVICE void operator()(int _idx) {
    float arr[16];
    InitializeRange(arr, arr + 16);

    {
      Span<float> s;
      Span<byte> bs = as_writable_bytes(s);
      SPAN_ASSERT_TRUE(bs.size() == s.size(), status_);
      SPAN_ASSERT_TRUE(bs.size_bytes() == s.size_bytes(), status_);
      SPAN_ASSERT_TRUE(bs.size() == 0, status_);
      SPAN_ASSERT_TRUE(bs.size_bytes() == 0, status_);
      SPAN_ASSERT_TRUE(bs.data() == nullptr, status_);
      SPAN_ASSERT_TRUE(static_cast<void*>(bs.data()) ==
                       static_cast<void*>(s.data()), status_);
    }

    {
      Span<float> s { arr };
      Span<byte> bs { as_writable_bytes(s) };
      SPAN_ASSERT_TRUE(s.size_bytes() == bs.size_bytes(), status_);
      SPAN_ASSERT_TRUE(static_cast<void*>(bs.data()) ==
                       static_cast<void*>(s.data()), status_);
    }
  }
};

}  // namespace common
}  // namespace xgboost

#endif
