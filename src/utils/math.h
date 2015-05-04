#ifndef XGBOOST_UTILS_MATH_H_
#define XGBOOST_UTILS_MATH_H_
/*!
 * \file math.h
 * \brief support additional math
 * \author Tianqi Chen
 */
#include <cmath>
#ifdef _MSC_VER
extern "C" {
#include <amp_math.h>
}
#endif
namespace xgboost {
namespace utils {
#ifdef XGBOOST_STRICT_CXX98_
// check nan
bool CheckNAN(double v);
double LogGamma(double v);
#else
template<typename T>
inline bool CheckNAN(T v) {
#ifdef _MSC_VER
  return (_isnan(x) != 0);
#else
  return isnan(v);
#endif
}
template<typename T>
inline T LogGamma(T v) {
  return lgamma(v);
}
#endif
}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_MATH_H_
