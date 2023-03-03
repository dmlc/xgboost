#include <gtest/gtest.h>
#include <xgboost/intrusive_ptr.h>

namespace xgboost {
namespace {
class NotCopyConstructible {
 public:
  float data;

  explicit NotCopyConstructible(float d) : data{d} {}
  NotCopyConstructible(NotCopyConstructible const &that) = delete;
  NotCopyConstructible &operator=(NotCopyConstructible const &that) = delete;
  NotCopyConstructible(NotCopyConstructible&& that) = default;
};
static_assert(
    !std::is_trivially_copy_constructible<NotCopyConstructible>::value);
static_assert(
    !std::is_trivially_copy_assignable<NotCopyConstructible>::value);

class ForIntrusivePtrTest {
 public:
  mutable class IntrusivePtrCell ref;
  float data { 0 };

  friend IntrusivePtrCell &
  IntrusivePtrRefCount(ForIntrusivePtrTest const *t) noexcept {  // NOLINT
    return t->ref;
  }

  ForIntrusivePtrTest() = default;
  ForIntrusivePtrTest(float a, int32_t b) : data{a + static_cast<float>(b)} {}

  explicit ForIntrusivePtrTest(NotCopyConstructible a) : data{a.data} {}
};
}  // anonymous namespace

TEST(IntrusivePtr, Basic) {
  IntrusivePtr<ForIntrusivePtrTest> ptr {new ForIntrusivePtrTest};
  auto p = ptr.get();

  // Copy ctor
  IntrusivePtr<ForIntrusivePtrTest> ptr_1 { ptr };
  ASSERT_EQ(ptr_1.get(), p);

  ASSERT_EQ((*ptr_1).data, ptr_1->data);
  ASSERT_EQ(ptr.use_count(), 2);

  // hash
  ASSERT_EQ(std::hash<IntrusivePtr<ForIntrusivePtrTest>>{}(ptr_1),
            std::hash<ForIntrusivePtrTest*>{}(ptr_1.get()));

  // Raw ptr comparison
  ASSERT_EQ(ptr, p);
  ASSERT_EQ(ptr_1, ptr);

  ForIntrusivePtrTest* raw_ptr {nullptr};
  ASSERT_NE(ptr_1, raw_ptr);
  ASSERT_NE(raw_ptr, ptr_1);

  // Reset with raw ptr.
  auto p_1 = new ForIntrusivePtrTest;
  ptr.reset(p_1);

  ASSERT_EQ(ptr_1.use_count(), 1);
  ASSERT_EQ(ptr.use_count(), 1);

  ASSERT_TRUE(ptr);
  ASSERT_TRUE(ptr_1);

  // Swap
  std::swap(ptr, ptr_1);
  ASSERT_NE(ptr, p_1);
  ASSERT_EQ(ptr_1, p_1);

  // Reset
  ptr.reset();
  ASSERT_FALSE(ptr);
  ASSERT_EQ(ptr.use_count(), 0);

  // Comparison operators
  ASSERT_EQ(ptr < ptr_1, ptr.get() < ptr_1.get());
  ASSERT_EQ(ptr > ptr_1, ptr.get() > ptr_1.get());

  ASSERT_LE(ptr, ptr);
  ASSERT_GE(ptr, ptr);

  // Copy assign
  IntrusivePtr<ForIntrusivePtrTest> ptr_2;
  ptr_2 = ptr_1;
  ASSERT_EQ(ptr_2, ptr_1);
  ASSERT_EQ(ptr_2.use_count(), 2);

  // Move assign
  IntrusivePtr<ForIntrusivePtrTest> ptr_3;
  ptr_3 = std::move(ptr_2);
  ASSERT_EQ(ptr_2.use_count(), 0);  // NOLINT
  ASSERT_EQ(ptr_3.use_count(), 2);

  // Move ctor
  IntrusivePtr<ForIntrusivePtrTest> ptr_4 { std::move(ptr_3) };
  ASSERT_EQ(ptr_3.use_count(), 0);  // NOLINT
  ASSERT_EQ(ptr_4.use_count(), 2);

  // Comparison
  ASSERT_EQ(ptr_1 > ptr_2, ptr_1.get() > ptr_2.get());
  ASSERT_EQ(ptr_1, ptr_1);
  ASSERT_EQ(ptr_1 < ptr_2, ptr_1.get() < ptr_2.get());
}
} // namespace xgboost
