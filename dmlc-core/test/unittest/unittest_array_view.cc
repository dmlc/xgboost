#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <dmlc/array_view.h>

void ArrayViewTest(dmlc::array_view<int> view, int base) {
  int cnt = base;
  for (int v : view) {
    CHECK_EQ(v, cnt);
    ++cnt;
  }
}

TEST(ArrayView, Basic) {
  std::vector<int> vec{0, 1, 2};
  ArrayViewTest(vec, 0);
  int arr[] = {1, 2, 3};
  ArrayViewTest(dmlc::array_view<int>(arr, arr + 3), 1);
  dmlc::array_view<int> a = vec;
  CHECK_EQ(a.size(), vec.size());
}
