#ifndef XGBOOST_UTILS_RANDOM_H_
#define XGBOOST_UTILS_RANDOM_H_
/*!
 * \file xgboost_random.h
 * \brief PRNG to support random number generation
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 *
 * Use standard PRNG from stdlib
 */
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "./utils.h"

/*! namespace of PRNG */
namespace xgboost {
namespace random {
#ifndef XGBOOST_CUSTOMIZE_PRNG_
/*! \brief seed the PRNG */
inline void Seed(unsigned seed) {
  srand(seed);
}
/*! \brief basic function, uniform */
inline double Uniform(void) {
  return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX)+1.0);
}
/*! \brief return a real numer uniform in (0,1) */
inline double NextDouble2(void) {
  return (static_cast<double>(rand()) + 1.0) / (static_cast<double>(RAND_MAX)+2.0);
}
/*! \brief return  x~N(0,1) */
inline double Normal(void) {
  double x, y, s;
  do {
    x = 2 * NextDouble2() - 1.0;
    y = 2 * NextDouble2() - 1.0;
    s = x*x + y*y;
  } while (s >= 1.0 || s == 0.0);

  return x * sqrt(-2.0 * log(s) / s);
}
#else
// include declarations, to be implemented
void Seed(unsigned seed);
double Uniform(void);
double Normal(void);
#endif

/*! \brief return a real number uniform in [0,1) */
inline double NextDouble(void) {
  return Uniform();
}
/*! \brief return a random number in n */
inline uint32_t NextUInt32(uint32_t n) {
  return (uint32_t)std::floor(NextDouble() * n);
}
/*! \brief return  x~N(mu,sigma^2) */
inline double SampleNormal(double mu, double sigma) {
  return Normal() * sigma + mu;
}
/*! \brief  return 1 with probability p, coin flip */
inline int SampleBinary(double p) {
  return NextDouble() < p;
}

template<typename T>
inline void Shuffle(T *data, size_t sz) {
  if (sz == 0) return;
  for (uint32_t i = (uint32_t)sz - 1; i > 0; i--) {
    std::swap(data[i], data[NextUInt32(i + 1)]);
  }
}
// random shuffle the data inside, require PRNG
template<typename T>
inline void Shuffle(std::vector<T> &data) {
  Shuffle(&data[0], data.size());
}

/*! \brief random number generator with independent random number seed*/
struct Random{
  /*! \brief set random number seed */
  inline void Seed(unsigned sd) {
	 this->rseed = sd;
#if defined(_MSC_VER)||defined(_WIN32)
     ::xgboost::random::Seed(sd);
#endif
  }
  /*! \brief return a real number uniform in [0,1) */
  inline double RandDouble(void) {
	// use rand instead of rand_r in windows, for MSVC it is fine since rand is threadsafe
	// For cygwin and mingw, this can slows down parallelism, but rand_r is only used in objective-inl.hpp, won't affect speed in general
	// todo, replace with another PRNG
#if defined(_MSC_VER)||defined(_WIN32)||defined(XGBOOST_STRICT_CXX98_)
    return Uniform();
#else
    return static_cast<double>(rand_r(&rseed)) / (static_cast<double>(RAND_MAX) + 1.0);
#endif
  }
  // random number seed
  unsigned rseed;
};
}  // namespace random
}  // namespace xgboost
#endif  // XGBOOST_UTILS_RANDOM_H_
