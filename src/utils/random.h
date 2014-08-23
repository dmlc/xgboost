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

/*! \brief seed the PRNG */
inline void Seed(uint32_t seed) {
  srand(seed);
}
/*! \brief return a real number uniform in [0,1) */
inline double NextDouble(void) {
  return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX)+1.0);
}
/*! \brief return a real numer uniform in (0,1) */
inline double NextDouble2(void) {
  return (static_cast<double>(rand()) + 1.0) / (static_cast<double>(RAND_MAX)+2.0);
}

/*! \brief return a random number */
inline uint32_t NextUInt32(void) {
  return (uint32_t)rand();
}
/*! \brief return a random number in n */
inline uint32_t NextUInt32(uint32_t n) {
  return (uint32_t)floor(NextDouble() * n);
}
/*! \brief return  x~N(0,1) */
inline double SampleNormal() {
  double x, y, s;
  do {
    x = 2 * NextDouble2() - 1.0;
    y = 2 * NextDouble2() - 1.0;
    s = x*x + y*y;
  } while (s >= 1.0 || s == 0.0);

  return x * sqrt(-2.0 * log(s) / s);
}

/*! \brief return iid x,y ~N(0,1) */
inline void SampleNormal2D(double &xx, double &yy) {
  double x, y, s;
  do {
    x = 2 * NextDouble2() - 1.0;
    y = 2 * NextDouble2() - 1.0;
    s = x*x + y*y;
  } while (s >= 1.0 || s == 0.0);
  double t = sqrt(-2.0 * log(s) / s);
  xx = x * t;
  yy = y * t;
}
/*! \brief return  x~N(mu,sigma^2) */
inline double SampleNormal(double mu, double sigma) {
  return SampleNormal() * sigma + mu;
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
  }
  /*! \brief return a real number uniform in [0,1) */
  inline double RandDouble(void) {
    return static_cast<double>( rand_r( &rseed ) ) / (static_cast<double>( RAND_MAX )+1.0);
  }
  // random number seed
  unsigned rseed;
};
}  // namespace random
}  // namespace xgboost
#endif  // XGBOOST_UTILS_RANDOM_H_
