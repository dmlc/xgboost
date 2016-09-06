/*!
 *  Copyright (c) 2016 by Contributors
 * \file array_view.h
 * \brief Read only data structure to reference array
 */
#ifndef DMLC_ARRAY_VIEW_H_
#define DMLC_ARRAY_VIEW_H_

#include <vector>
#include <array>

namespace dmlc {

/*!
 * \brief Read only data structure to reference continuous memory region of array.
 * Provide unified view for vector, array and C style array.
 * This data structure do not guarantee aliveness of referenced array.
 *
 * Make sure do not use array_view to record data in async function closures.
 * Also do not use array_view to create reference to temporary data structure.
 *
 * \tparam ValueType The value
 *
 * \code
 *  std::vector<int> myvec{1,2,3};
 *  dmlc::array_view<int> view(myvec);
 *  // indexed visit to the view.
 *  LOG(INFO) << view[0];
 *
 *  for (int v : view) {
 *     // visit each element in the view
 *  }
 * \endcode
 */
template<typename ValueType>
class array_view {
 public:
  /*! \brief default constructor */
  array_view() = default;
  /*!
   * \brief default copy constructor
   * \param other another array view.
   */
  array_view(const array_view<ValueType> &other) = default;  // NOLINT(*)
  /*!
   * \brief default move constructor
   * \param other another array view.
   */
  array_view(array_view<ValueType>&& other) = default; // NOLINT(*)
  /*!
   * \brief default assign constructor
   * \param other another array view.
   * \return self.
   */
  array_view<ValueType>& operator=(const array_view<ValueType>& other) = default; // NOLINT(*)
  /*!
   * \brief construct array view std::vector
   * \param other vector container
   */
  array_view(const std::vector<ValueType>& other) {  // NOLINT(*)
    if (other.size() != 0) {
      begin_ = &other[0]; size_ = other.size();
    }
  }
  /*!
   * \brief construct array std::array
   * \param other another array view.
   */
  template<std::size_t size>
  array_view(const std::array<ValueType, size>& other) {  // NOLINT(*)
    if (size != 0) {
      begin_ = &other[0]; size_ = size;
    }
  }
  /*!
   * \brief construct array view from continuous segment
   * \param begin beginning pointre
   * \param end end pointer
   */
  array_view(const ValueType* begin, const ValueType* end) {
    if (begin < end) {
      begin_ = begin;
      size_ = end - begin;
    }
  }
  /*! \return size of the array */
  inline size_t size() const {
    return size_;
  }
  /*! \return begin of the array */
  inline const ValueType* begin() const {
    return begin_;
  }
  /*! \return end point of the array */
  inline const ValueType* end() const {
    return begin_ + size_;
  }
  /*!
   * \brief get i-th element from the view
   * \param i The index.
   * \return const reference to i-th element.
   */
  inline const ValueType& operator[](size_t i) const {
    return begin_[i];
  }

 private:
  /*! \brief the begin of the view */
  const ValueType* begin_{nullptr};
  /*! \brief The size of the view */
  size_t size_{0};
};

}  // namespace dmlc

#endif  // DMLC_ARRAY_VIEW_H_
