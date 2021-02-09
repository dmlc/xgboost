/*!
 * Copyright 2018 XGBoost contributors
 * \brief span class based on ISO++20 span
 *
 * About NOLINTs in this file:
 *
 *   If we want Span to work with std interface, like range for loop, the
 *   naming must be consistant with std, not XGBoost. Also, the interface also
 *   conflicts with XGBoost coding style, specifically, the use of `explicit'
 *   keyword.
 *
 *
 * Some of the code is copied from Guidelines Support Library, here is the
 * license:
 *
 * Copyright (c) 2015 Microsoft Corporation. All rights reserved.
 *
 * This code is licensed under the MIT License (MIT).
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef XGBOOST_SPAN_H_
#define XGBOOST_SPAN_H_

#include <xgboost/base.h>
#include <xgboost/logging.h>

#include <cinttypes>          // size_t
#include <limits>             // numeric_limits
#include <iterator>
#include <type_traits>
#include <cstdio>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif  // defined(__CUDACC__)

/*!
 * The version number 1910 is picked up from GSL.
 *
 * We might want to use MOODYCAMEL_NOEXCEPT from dmlc/concurrentqueue.h. But
 * there are a lot more definitions in that file would cause warnings/troubles
 * in MSVC 2013. Currently we try to keep the closure of Span as minimal as
 * possible.
 *
 * There are other workarounds for MSVC, like _Unwrapped, _Verify_range ...
 * Some of these are hiden magics of MSVC and I tried to avoid them. Should any
 * of them become needed, please consult the source code of GSL, and possibily
 * some explanations from this thread:
 *
 *   https://github.com/Microsoft/GSL/pull/664
 *
 * TODO(trivialfis): Group these MSVC workarounds into a manageable place.
 */
#if defined(_MSC_VER) && _MSC_VER < 1910

#define __span_noexcept

#pragma push_macro("constexpr")
#define constexpr /*constexpr*/

#else

#define __span_noexcept noexcept

#endif  // defined(_MSC_VER) && _MSC_VER < 1910

namespace xgboost {
namespace common {

#if defined(__CUDA_ARCH__)
// Usual logging facility is not available inside device code.

#if defined(_MSC_VER)

// Windows CUDA doesn't have __assert_fail.
#define KERNEL_CHECK(cond)                                                     \
  do {                                                                         \
    if (XGBOOST_EXPECT(!(cond), false)) {                                      \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)

#else  // defined(_MSC_VER)

#define __ASSERT_STR_HELPER(x) #x

#define KERNEL_CHECK(cond)                                                     \
  (XGBOOST_EXPECT((cond), true)                                                \
       ? static_cast<void>(0)                                                  \
       : __assert_fail(__ASSERT_STR_HELPER(e), __FILE__, __LINE__,             \
                       __PRETTY_FUNCTION__))

#endif  // defined(_MSC_VER)

#define SPAN_CHECK KERNEL_CHECK

#else  // not CUDA

#define KERNEL_CHECK(cond)                                                     \
  (XGBOOST_EXPECT((cond), true) ? static_cast<void>(0) : std::terminate())

#define SPAN_CHECK(cond) KERNEL_CHECK(cond)

#endif  // __CUDA_ARCH__

#if defined(__CUDA_ARCH__)
#define SPAN_LT(lhs, rhs) KERNEL_CHECK((lhs) < (rhs))
#else
#define SPAN_LT(lhs, rhs) KERNEL_CHECK((lhs) < (rhs))
#endif  // defined(__CUDA_ARCH__)

namespace detail {
/*!
 * By default, XGBoost uses uint32_t for indexing data. int64_t covers all
 *   values uint32_t can represent. Also, On x86-64 Linux, GCC uses long int to
 *   represent ptrdiff_t, which is just int64_t. So we make it determinstic
 *   here.
 */
using ptrdiff_t = typename std::conditional<  // NOLINT
    std::is_same<std::ptrdiff_t, std::int64_t>::value,
    std::ptrdiff_t, std::int64_t>::type;
}  // namespace detail

#if defined(_MSC_VER) && _MSC_VER < 1910
constexpr const std::size_t
dynamic_extent = std::numeric_limits<std::size_t>::max();  // NOLINT
#else
constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();  // NOLINT
#endif  // defined(_MSC_VER) && _MSC_VER < 1910

enum class byte : unsigned char {};  // NOLINT

template <class ElementType, std::size_t Extent>
class Span;

namespace detail {

template <typename SpanType, bool IsConst>
class SpanIterator {
  using ElementType = typename SpanType::element_type;

 public:
  using iterator_category = std::random_access_iterator_tag;      // NOLINT
  using value_type = typename SpanType::value_type;  // NOLINT
  using difference_type = detail::ptrdiff_t;             // NOLINT

  using reference = typename std::conditional<                    // NOLINT
    IsConst, const ElementType, ElementType>::type&;
  using pointer = typename std::add_pointer<reference>::type;     // NOLINT

  constexpr SpanIterator() = default;

  XGBOOST_DEVICE constexpr SpanIterator(
      const SpanType* _span,
      typename SpanType::index_type _idx) __span_noexcept :
                                           span_(_span), index_(_idx) {}

  friend SpanIterator<SpanType, true>;
  template <bool B, typename std::enable_if<!B && IsConst>::type* = nullptr>
  XGBOOST_DEVICE constexpr SpanIterator(                         // NOLINT
      const SpanIterator<SpanType, B>& other_) __span_noexcept
      : SpanIterator(other_.span_, other_.index_) {}

  XGBOOST_DEVICE reference operator*() const {
    SPAN_CHECK(index_ < span_->size());
    return *(span_->data() + index_);
  }
  XGBOOST_DEVICE reference operator[](difference_type n) const {
    return *(*this + n);
  }

  XGBOOST_DEVICE pointer operator->() const {
    SPAN_CHECK(index_ != span_->size());
    return span_->data() + index_;
  }

  XGBOOST_DEVICE SpanIterator& operator++() {
    SPAN_CHECK(index_ != span_->size());
    index_++;
    return *this;
  }

  XGBOOST_DEVICE SpanIterator operator++(int) {
    auto ret = *this;
    ++(*this);
    return ret;
  }

  XGBOOST_DEVICE SpanIterator& operator--() {
    SPAN_CHECK(index_ != 0 && index_ <= span_->size());
    index_--;
    return *this;
  }

  XGBOOST_DEVICE SpanIterator operator--(int) {
    auto ret = *this;
    --(*this);
    return ret;
  }

  XGBOOST_DEVICE SpanIterator operator+(difference_type n) const {
    auto ret = *this;
    return ret += n;
  }

  XGBOOST_DEVICE SpanIterator& operator+=(difference_type n) {
    SPAN_CHECK((index_ + n) <= span_->size());
    index_ += n;
    return *this;
  }

  XGBOOST_DEVICE difference_type operator-(SpanIterator rhs) const {
    SPAN_CHECK(span_ == rhs.span_);
    return index_ - rhs.index_;
  }

  XGBOOST_DEVICE SpanIterator operator-(difference_type n) const {
    auto ret = *this;
    return ret -= n;
  }

  XGBOOST_DEVICE SpanIterator& operator-=(difference_type n) {
    return *this += -n;
  }

  // friends
  XGBOOST_DEVICE constexpr friend bool operator==(
      SpanIterator _lhs, SpanIterator _rhs) __span_noexcept {
    return _lhs.span_ == _rhs.span_ && _lhs.index_ == _rhs.index_;
  }

  XGBOOST_DEVICE constexpr friend bool operator!=(
      SpanIterator _lhs, SpanIterator _rhs) __span_noexcept {
    return !(_lhs == _rhs);
  }

  XGBOOST_DEVICE constexpr friend bool operator<(
      SpanIterator _lhs, SpanIterator _rhs) __span_noexcept {
    return _lhs.index_ < _rhs.index_;
  }

  XGBOOST_DEVICE constexpr friend bool operator<=(
      SpanIterator _lhs, SpanIterator _rhs) __span_noexcept {
    return !(_rhs < _lhs);
  }

  XGBOOST_DEVICE constexpr friend bool operator>(
      SpanIterator _lhs, SpanIterator _rhs) __span_noexcept {
    return _rhs < _lhs;
  }

  XGBOOST_DEVICE constexpr friend bool operator>=(
      SpanIterator _lhs, SpanIterator _rhs) __span_noexcept {
    return !(_rhs > _lhs);
  }

 protected:
  const SpanType *span_ { nullptr };
  typename SpanType::index_type index_ { 0 };
};


// It's tempting to use constexpr instead of structs to do the following meta
// programming. But remember that we are supporting MSVC 2013 here.

/*!
 * The extent E of the span returned by subspan is determined as follows:
 *
 *   - If Count is not dynamic_extent, Count;
 *   - Otherwise, if Extent is not dynamic_extent, Extent - Offset;
 *   - Otherwise, dynamic_extent.
 */
template <std::size_t Extent, std::size_t Offset, std::size_t Count>
struct ExtentValue : public std::integral_constant<
  std::size_t, Count != dynamic_extent ?
  Count : (Extent != dynamic_extent ? Extent - Offset : Extent)> {};

/*!
 * If N is dynamic_extent, the extent of the returned span E is also
 * dynamic_extent; otherwise it is std::size_t(sizeof(T)) * N.
 */
template <typename T, std::size_t Extent>
struct ExtentAsBytesValue : public std::integral_constant<
  std::size_t,
  Extent == dynamic_extent ?
  Extent : sizeof(T) * Extent> {};

template <std::size_t From, std::size_t To>
struct IsAllowedExtentConversion : public std::integral_constant<
  bool, From == To || From == dynamic_extent || To == dynamic_extent> {};

template <class From, class To>
struct IsAllowedElementTypeConversion : public std::integral_constant<
  bool, std::is_convertible<From(*)[], To(*)[]>::value> {};

template <class T>
struct IsSpanOracle : std::false_type {};

template <class T, std::size_t Extent>
struct IsSpanOracle<Span<T, Extent>> : std::true_type {};

template <class T>
struct IsSpan : public IsSpanOracle<typename std::remove_cv<T>::type> {};

// Re-implement std algorithms here to adopt CUDA.
template <typename T>
struct Less {
  XGBOOST_DEVICE constexpr bool operator()(const T& _x, const T& _y) const {
    return _x < _y;
  }
};

template <typename T>
struct Greater {
  XGBOOST_DEVICE constexpr bool operator()(const T& _x, const T& _y) const {
    return _x > _y;
  }
};

template <class InputIt1, class InputIt2,
          class Compare =
          detail::Less<decltype(std::declval<InputIt1>().operator*())>>
XGBOOST_DEVICE bool LexicographicalCompare(InputIt1 first1, InputIt1 last1,
                                            InputIt2 first2, InputIt2 last2) {
  Compare comp;
  for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
    if (comp(*first1, *first2)) {
      return true;
    }
    if (comp(*first2, *first1)) {
      return false;
    }
  }
  return first1 == last1 && first2 != last2;
}

}  // namespace detail


/*!
 * \brief span class implementation, based on ISO++20 span<T>. The interface
 *      should be the same.
 *
 * What's different from span<T> in Guidelines Support Library (GSL)
 *
 *    Interface might be slightly different, we stick with ISO.
 *
 *    GSL uses C++14/17 features, which are not available here.
 *    GSL uses constexpr extensively, which is not possibile with limitation
 *      of C++11.
 *    GSL doesn't concern about CUDA.
 *
 *    GSL is more thoroughly implemented and tested.
 *    GSL is more optimized, especially for static extent.
 *
 *    GSL uses __buildin_unreachable() when error, Span<T> uses dmlc LOG and
 *      customized CUDA logging.
 *
 *
 * What's different from span<T> in ISO++20 (ISO)
 *
 *    ISO uses functions/structs from std library, which might be not available
 *      in CUDA.
 *    Initializing from std::array is not supported.
 *
 *    ISO uses constexpr extensively, which is not possibile with limitation
 *      of C++11.
 *    ISO uses C++14/17 features, which is not available here.
 *    ISO doesn't concern about CUDA.
 *
 *    ISO uses std::terminate(), Span<T> uses dmlc LOG and customized CUDA
 *      logging.
 *
 *
 * Limitations:
 *    With thrust:
 *       It's not adviced to initialize Span with host_vector directly, since
 *         host_vector::data() is a host function.
 *       It's not possible to initialize Span with device_vector directly, since
 *         device_vector::data() returns a wrapped pointer.
 *       It's unclear that what kind of thrust algorithm can be used without
 *         memory error. See the test case "GPUSpan.WithTrust"
 *
 *    Pass iterator to kernel:
 *       Not possible. Use subspan instead.
 *
 *       The underlying Span in SpanIterator is a pointer, but CUDA pass kernel
 *       parameter by value.  If we were to hold a Span value instead of a
 *       pointer, the following snippet will crash, violating the safety
 *       purpose of Span:
 *
 *       \code{.cpp}
 *       Span<float> span {arr_a};
 *       auto beg = span.begin();
 *
 *       Span<float> span_b = arr_b;
 *       span = span_b;
 *
 *       delete arr_a;
 *       beg++;                 // crash
 *       \endcode
 *
 *       While hoding a pointer or reference should avoid the problem, its a
 *       compromise. Since we have subspan, it's acceptable not to support
 *       passing iterator.
 */
template <typename T,
          std::size_t Extent = dynamic_extent>
class Span {
 public:
  using element_type = T;                               // NOLINT
  using value_type = typename std::remove_cv<T>::type;  // NOLINT
  using index_type = std::size_t;                       // NOLINT
  using difference_type = detail::ptrdiff_t;            // NOLINT
  using pointer = T*;                                   // NOLINT
  using reference = T&;                                 // NOLINT

  using iterator = detail::SpanIterator<Span<T, Extent>, false>;             // NOLINT
  using const_iterator = const detail::SpanIterator<Span<T, Extent>, true>;  // NOLINT
  using reverse_iterator = detail::SpanIterator<Span<T, Extent>, false>;     // NOLINT
  using const_reverse_iterator = const detail::SpanIterator<Span<T, Extent>, true>;  // NOLINT

  // constructors
  constexpr Span() __span_noexcept = default;

  XGBOOST_DEVICE Span(pointer _ptr, index_type _count) :
      size_(_count), data_(_ptr) {
    SPAN_CHECK(!(Extent != dynamic_extent && _count != Extent));
    SPAN_CHECK(_ptr || _count == 0);
  }

  XGBOOST_DEVICE Span(pointer _first, pointer _last) :
      size_(_last - _first), data_(_first) {
    SPAN_CHECK(data_ || size_ == 0);
  }

  template <std::size_t N>
  XGBOOST_DEVICE constexpr Span(element_type (&arr)[N])  // NOLINT
      __span_noexcept : size_(N), data_(&arr[0]) {}

  template <class Container,
            class = typename std::enable_if<
              !std::is_const<element_type>::value &&
              !detail::IsSpan<Container>::value &&
              std::is_convertible<typename Container::pointer, pointer>::value &&
              std::is_convertible<typename Container::pointer,
                                  decltype(std::declval<Container>().data())>::value>::type>
  Span(Container& _cont) :  // NOLINT
      size_(_cont.size()), data_(_cont.data()) {
    static_assert(!detail::IsSpan<Container>::value, "Wrong constructor of Span is called.");
  }

  template <class Container,
            class = typename std::enable_if<
              std::is_const<element_type>::value &&
              !detail::IsSpan<Container>::value &&
              std::is_convertible<typename Container::pointer, pointer>::value &&
              std::is_convertible<typename Container::pointer,
                                  decltype(std::declval<Container>().data())>::value>::type>
  Span(const Container& _cont) : size_(_cont.size()),  // NOLINT
                                 data_(_cont.data()) {
    static_assert(!detail::IsSpan<Container>::value, "Wrong constructor of Span is called.");
  }

  template <class U, std::size_t OtherExtent,
            class = typename std::enable_if<
              detail::IsAllowedElementTypeConversion<U, T>::value &&
              detail::IsAllowedExtentConversion<OtherExtent, Extent>::value>>
  XGBOOST_DEVICE constexpr Span(const Span<U, OtherExtent>& _other)   // NOLINT
      __span_noexcept : size_(_other.size()), data_(_other.data()) {}

  XGBOOST_DEVICE constexpr Span(const Span& _other)
      __span_noexcept : size_(_other.size()), data_(_other.data()) {}

  XGBOOST_DEVICE Span& operator=(const Span& _other) __span_noexcept {
    size_ = _other.size();
    data_ = _other.data();
    return *this;
  }

  XGBOOST_DEVICE ~Span() __span_noexcept {};  // NOLINT

  XGBOOST_DEVICE constexpr iterator begin() const __span_noexcept {  // NOLINT
    return {this, 0};
  }

  XGBOOST_DEVICE constexpr iterator end() const __span_noexcept {    // NOLINT
    return {this, size()};
  }

  XGBOOST_DEVICE constexpr const_iterator cbegin() const __span_noexcept {  // NOLINT
    return {this, 0};
  }

  XGBOOST_DEVICE constexpr const_iterator cend() const __span_noexcept {    // NOLINT
    return {this, size()};
  }

  XGBOOST_DEVICE constexpr reverse_iterator rbegin() const __span_noexcept {  // NOLINT
    return reverse_iterator{end()};
  }

  XGBOOST_DEVICE constexpr reverse_iterator rend() const __span_noexcept {    // NOLINT
    return reverse_iterator{begin()};
  }

  XGBOOST_DEVICE constexpr const_reverse_iterator crbegin() const __span_noexcept {  // NOLINT
    return const_reverse_iterator{cend()};
  }

  XGBOOST_DEVICE constexpr const_reverse_iterator crend() const __span_noexcept {    // NOLINT
    return const_reverse_iterator{cbegin()};
  }

  // element access

  XGBOOST_DEVICE reference front() const {  // NOLINT
    return (*this)[0];
  }

  XGBOOST_DEVICE reference back() const {  // NOLINT
    return (*this)[size() - 1];
  }

  XGBOOST_DEVICE reference operator[](index_type _idx) const {
    SPAN_LT(_idx, size());
    return data()[_idx];
  }

  XGBOOST_DEVICE reference operator()(index_type _idx) const {
    return this->operator[](_idx);
  }

  XGBOOST_DEVICE constexpr pointer data() const __span_noexcept {   // NOLINT
    return data_;
  }

  // Observers
  XGBOOST_DEVICE constexpr index_type size() const __span_noexcept {  // NOLINT
    return size_;
  }
  XGBOOST_DEVICE constexpr index_type size_bytes() const __span_noexcept {  // NOLINT
    return size() * sizeof(T);
  }

  XGBOOST_DEVICE constexpr bool empty() const __span_noexcept {  // NOLINT
    return size() == 0;
  }

  // Subviews
  template <std::size_t Count>
  XGBOOST_DEVICE Span<element_type, Count> first() const {  // NOLINT
    SPAN_CHECK(Count <= size());
    return {data(), Count};
  }

  XGBOOST_DEVICE Span<element_type, dynamic_extent> first(  // NOLINT
      std::size_t _count) const {
    SPAN_CHECK(_count <= size());
    return {data(), _count};
  }

  template <std::size_t Count>
  XGBOOST_DEVICE Span<element_type, Count> last() const {  // NOLINT
    SPAN_CHECK(Count <= size());
    return {data() + size() - Count, Count};
  }

  XGBOOST_DEVICE Span<element_type, dynamic_extent> last(  // NOLINT
      std::size_t _count) const {
    SPAN_CHECK(_count <= size());
    return subspan(size() - _count, _count);
  }

  /*!
   * If Count is std::dynamic_extent, r.size() == this->size() - Offset;
   * Otherwise r.size() == Count.
   */
  template <std::size_t Offset,
            std::size_t Count = dynamic_extent>
  XGBOOST_DEVICE auto subspan() const ->                   // NOLINT
      Span<element_type,
           detail::ExtentValue<Extent, Offset, Count>::value> {
    SPAN_CHECK((Count == dynamic_extent) ?
               (Offset <= size()) : (Offset + Count <= size()));
    return {data() + Offset, Count == dynamic_extent ? size() - Offset : Count};
  }

  XGBOOST_DEVICE Span<element_type, dynamic_extent> subspan(  // NOLINT
      index_type _offset,
      index_type _count = dynamic_extent) const {
    SPAN_CHECK((_count == dynamic_extent) ?
               (_offset <= size()) : (_offset + _count <= size()));
    return {data() + _offset, _count ==
            dynamic_extent ? size() - _offset : _count};
  }

 private:
  index_type size_ { 0 };
  pointer data_ { nullptr };
};

template <class T, std::size_t X, class U, std::size_t Y>
XGBOOST_DEVICE bool operator==(Span<T, X> l, Span<U, Y> r) {
  if (l.size() != r.size()) {
    return false;
  }
  for (auto l_beg = l.cbegin(), r_beg = r.cbegin(); l_beg != l.cend();
       ++l_beg, ++r_beg) {
    if (*l_beg != *r_beg) {
      return false;
    }
  }
  return true;
}

template <class T, std::size_t X, class U, std::size_t Y>
XGBOOST_DEVICE constexpr bool operator!=(Span<T, X> l, Span<U, Y> r) {
  return !(l == r);
}

template <class T, std::size_t X, class U, std::size_t Y>
XGBOOST_DEVICE constexpr bool operator<(Span<T, X> l, Span<U, Y> r) {
  return detail::LexicographicalCompare(l.begin(), l.end(),
                                         r.begin(), r.end());
}

template <class T, std::size_t X, class U, std::size_t Y>
XGBOOST_DEVICE constexpr bool operator<=(Span<T, X> l, Span<U, Y> r) {
  return !(l > r);
}

template <class T, std::size_t X, class U, std::size_t Y>
XGBOOST_DEVICE constexpr bool operator>(Span<T, X> l, Span<U, Y> r) {
  return detail::LexicographicalCompare<
    typename Span<T, X>::iterator, typename Span<U, Y>::iterator,
    detail::Greater<typename Span<T, X>::element_type>>(l.begin(), l.end(),
                                                        r.begin(), r.end());
}

template <class T, std::size_t X, class U, std::size_t Y>
XGBOOST_DEVICE constexpr bool operator>=(Span<T, X> l, Span<U, Y> r) {
  return !(l < r);
}

template <class T, std::size_t E>
XGBOOST_DEVICE auto as_bytes(Span<T, E> s) __span_noexcept ->           // NOLINT
    Span<const byte, detail::ExtentAsBytesValue<T, E>::value> {
  return {reinterpret_cast<const byte*>(s.data()), s.size_bytes()};
}

template <class T, std::size_t E>
XGBOOST_DEVICE auto as_writable_bytes(Span<T, E> s) __span_noexcept ->  // NOLINT
    Span<byte, detail::ExtentAsBytesValue<T, E>::value> {
  return {reinterpret_cast<byte*>(s.data()), s.size_bytes()};
}

}  // namespace common
}  // namespace xgboost

#if defined(_MSC_VER) &&_MSC_VER < 1910
#undef constexpr
#pragma pop_macro("constexpr")
#undef __span_noexcept
#endif  // _MSC_VER < 1910

#endif  // XGBOOST_SPAN_H_
