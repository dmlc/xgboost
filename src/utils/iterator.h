#ifndef XGBOOST_UTILS_ITERATOR_H
#define XGBOOST_UTILS_ITERATOR_H
#include <cstdio>
/*!
 * \file iterator.h
 * \brief itertator interface
 * \author Tianqi Chen
 */
namespace xgboost {
namespace utils {
/*!
 * \brief iterator interface
 * \tparam DType data type
 */
template<typename DType>
class IIterator {
 public:
  /*!
   * \brief set the parameter 
   * \param name name of parameter
   * \param val value of parameter
   */
  virtual void SetParam(const char *name, const char *val) {}
  /*! \brief initalize the iterator so that we can use the iterator */
  virtual void Init(void) {}
  /*! \brief set before first of the item */
  virtual void BeforeFirst(void) = 0;
  /*! \brief move to next item */
  virtual bool Next(void) = 0;
  /*! \brief get current data */
  virtual const DType &Value(void) const = 0;
 public:
  /*! \brief constructor */
  virtual ~IIterator(void) {}
};

}  // namespace utils
}  // namespace xgboost
#endif

