#ifndef XGBOOST_TEST_COMMON_H_
#define XGBOOST_TEST_COMMON_H_

namespace xgboost {
namespace common {

template <typename Iter>
void InitializeRange(Iter _begin, Iter _end) {
  float j = 0;
  for (Iter i = _begin; i != _end; ++i, ++j) {
    *i = j;
  }
}

}  // namespace common
}  // namespace xgboost

#endif
