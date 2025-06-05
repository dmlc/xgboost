/**
 * Copyright 2015-2025, XGBoost Contributors
 * \file base.h
 * \brief Defines configuration macros and basic types for xgboost.
 */
#ifndef XGBOOST_BASE_H_
#define XGBOOST_BASE_H_

#include <dmlc/omp.h>  // for omp_uint, omp_ulong
// Put the windefs here to guard as many files as possible.
#include <xgboost/windefs.h>

#include <cstdint>  // for int32_t, uint64_t, int16_t
#include <ostream>  // for ostream
#include <string>   // for string
#include <utility>  // for pair
#include <vector>   // for vector

/*!
 * \brief string flag for R library, to leave hooks when needed.
 */
#ifndef XGBOOST_STRICT_R_MODE
#define XGBOOST_STRICT_R_MODE 0
#endif  // XGBOOST_STRICT_R_MODE

/*!
 * \brief Whether always log console message with time.
 *  It will display like, with timestamp appended to head of the message.
 *  "[21:47:50] 6513x126 matrix with 143286 entries loaded from
 * ../data/agaricus.txt.train"
 */
#ifndef XGBOOST_LOG_WITH_TIME
#define XGBOOST_LOG_WITH_TIME 1
#endif  // XGBOOST_LOG_WITH_TIME

/*!
 * \brief Whether to customize global PRNG.
 */
#ifndef XGBOOST_CUSTOMIZE_GLOBAL_PRNG
#define XGBOOST_CUSTOMIZE_GLOBAL_PRNG 0
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG

/*!
 * \brief Check if alignas(*) keyword is supported. (g++ 4.8 or higher)
 */
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ > 4)
#define XGBOOST_ALIGNAS(X) alignas(X)
#else
#define XGBOOST_ALIGNAS(X)
#endif  // defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ > 4)

#if defined(__GNUC__)
#define XGBOOST_EXPECT(cond, ret)  __builtin_expect((cond), (ret))
#else
#define XGBOOST_EXPECT(cond, ret) (cond)
#endif  // defined(__GNUC__)

/*!
 * \brief Tag function as usable by device
 */
#if defined (__CUDA__) || defined(__NVCC__)
#define XGBOOST_DEVICE __host__ __device__
#else
#define XGBOOST_DEVICE
#endif  // defined (__CUDA__) || defined(__NVCC__)

#if defined(__CUDA__) || defined(__CUDACC__)
#define XGBOOST_HOST_DEV_INLINE XGBOOST_DEVICE __forceinline__
#define XGBOOST_DEV_INLINE __device__ __forceinline__
#else
#define XGBOOST_HOST_DEV_INLINE
#define XGBOOST_DEV_INLINE
#endif  // defined(__CUDA__) || defined(__CUDACC__)


// restrict
#if defined(_MSC_VER)
#define XGBOOST_RESTRICT __restrict
#else
#define XGBOOST_RESTRICT __restrict__
#endif

// These check are for Makefile.
#if !defined(XGBOOST_MM_PREFETCH_PRESENT) && !defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
/* default logic for software pre-fetching */
#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_AMD64))) || defined(__INTEL_COMPILER)
// Enable _mm_prefetch for Intel compiler and MSVC+x86
  #define XGBOOST_MM_PREFETCH_PRESENT
  #define XGBOOST_BUILTIN_PREFETCH_PRESENT
#elif defined(__GNUC__)
// Enable __builtin_prefetch for GCC
#define XGBOOST_BUILTIN_PREFETCH_PRESENT
#endif  // GUARDS

#endif  // !defined(XGBOOST_MM_PREFETCH_PRESENT) && !defined()

namespace xgboost {
/*! \brief unsigned integer type used for feature index. */
using bst_uint = std::uint32_t;  // NOLINT
/*! \brief unsigned long integers */
using bst_ulong = std::uint64_t;  // NOLINT
/*! \brief float type, used for storing statistics */
using bst_float = float;  // NOLINT
/*! \brief Categorical value type. */
using bst_cat_t = std::int32_t;  // NOLINT
/*! \brief Type for data column (feature) index. */
using bst_feature_t = std::uint32_t;  // NOLINT
/**
 * @brief Type for histogram bin index.  We sometimes use -1 to indicate invalid bin.
 */
using bst_bin_t = std::int32_t;  // NOLINT
/**
 * @brief Type for data row index (sample).
 */
using bst_idx_t = std::uint64_t;  // NOLINT
/**
 * \brief Type for tree node index.
 */
using bst_node_t = std::int32_t;      // NOLINT
/**
 * @brief Type for ranking group index.
 */
using bst_group_t = std::uint32_t;  // NOLINT
/**
 * @brief Type for indexing into output targets.
 */
using bst_target_t = std::uint32_t;  // NOLINT
/**
 * @brief Type for indexing boosted layers.
 */
using bst_layer_t = std::int32_t;  // NOLINT
/**
 * @brief Type for indexing trees.
 */
using bst_tree_t = std::int32_t;  // NOLINT
/**
 * @brief Ordinal of a CUDA device.
 */
using bst_d_ordinal_t = std::int16_t;  // NOLINT

namespace detail {
/*! \brief Implementation of gradient statistics pair. Template specialisation
 * may be used to overload different gradients types e.g. low precision, high
 * precision, integer, floating point. */
template <typename T>
class GradientPairInternal {
  /*! \brief gradient statistics */
  T grad_{0};
  /*! \brief second order gradient statistics */
  T hess_{0};

  XGBOOST_DEVICE void SetGrad(T g) { grad_ = g; }
  XGBOOST_DEVICE void SetHess(T h) { hess_ = h; }

 public:
  using ValueT = T;

  inline void Add(const ValueT& grad, const ValueT& hess) {
    grad_ += grad;
    hess_ += hess;
  }

  inline static void Reduce(GradientPairInternal<T>& a, const GradientPairInternal<T>& b) { // NOLINT(*)
    a += b;
  }

  GradientPairInternal() = default;

  XGBOOST_DEVICE GradientPairInternal(T grad, T hess) {
    SetGrad(grad);
    SetHess(hess);
  }

  // Copy constructor if of same value type, marked as default to be trivially_copyable
  GradientPairInternal(GradientPairInternal const &g) = default;
  GradientPairInternal(GradientPairInternal &&g) = default;
  GradientPairInternal &operator=(GradientPairInternal const &that) = default;
  GradientPairInternal &operator=(GradientPairInternal &&that) = default;

  // Copy constructor if different value type - use getters and setters to
  // perform conversion
  template <typename T2>
  XGBOOST_DEVICE explicit GradientPairInternal(const GradientPairInternal<T2> &g) {
    SetGrad(g.GetGrad());
    SetHess(g.GetHess());
  }

  XGBOOST_DEVICE T GetGrad() const { return grad_; }
  XGBOOST_DEVICE T GetHess() const { return hess_; }

  XGBOOST_DEVICE GradientPairInternal<T> &operator+=(
      const GradientPairInternal<T> &rhs) {
    grad_ += rhs.grad_;
    hess_ += rhs.hess_;
    return *this;
  }

  XGBOOST_DEVICE GradientPairInternal<T> operator+(
      const GradientPairInternal<T> &rhs) const {
    GradientPairInternal<T> g;
    g.grad_ = grad_ + rhs.grad_;
    g.hess_ = hess_ + rhs.hess_;
    return g;
  }

  XGBOOST_DEVICE GradientPairInternal<T> &operator-=(
      const GradientPairInternal<T> &rhs) {
    grad_ -= rhs.grad_;
    hess_ -= rhs.hess_;
    return *this;
  }

  XGBOOST_DEVICE GradientPairInternal<T> operator-(
      const GradientPairInternal<T> &rhs) const {
    GradientPairInternal<T> g;
    g.grad_ = grad_ - rhs.grad_;
    g.hess_ = hess_ - rhs.hess_;
    return g;
  }

  XGBOOST_DEVICE GradientPairInternal<T> &operator*=(float multiplier) {
    grad_ *= multiplier;
    hess_ *= multiplier;
    return *this;
  }

  XGBOOST_DEVICE GradientPairInternal<T> operator*(float multiplier) const {
    GradientPairInternal<T> g;
    g.grad_ = grad_ * multiplier;
    g.hess_ = hess_ * multiplier;
    return g;
  }

  XGBOOST_DEVICE GradientPairInternal<T> &operator/=(float divisor) {
    grad_ /= divisor;
    hess_ /= divisor;
    return *this;
  }

  XGBOOST_DEVICE GradientPairInternal<T> operator/(float divisor) const {
    GradientPairInternal<T> g;
    g.grad_ = grad_ / divisor;
    g.hess_ = hess_ / divisor;
    return g;
  }

  XGBOOST_DEVICE bool operator==(const GradientPairInternal<T> &rhs) const {
    return grad_ == rhs.grad_ && hess_ == rhs.hess_;
  }

  XGBOOST_DEVICE explicit GradientPairInternal(int value) {
    *this = GradientPairInternal<T>(static_cast<float>(value),
                                    static_cast<float>(value));
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const GradientPairInternal<T> &g) {
    os << g.GetGrad() << "/" << g.GetHess();
    return os;
  }
};
}  // namespace detail

/*! \brief gradient statistics pair usually needed in gradient boosting */
using GradientPair = detail::GradientPairInternal<float>;
/*! \brief High precision gradient statistics pair */
using GradientPairPrecise = detail::GradientPairInternal<double>;

/*! \brief Fixed point representation for high precision gradient pair. Has a different interface so
 * we don't accidentally use it in gain calculations.*/
class GradientPairInt64 {
  using T = int64_t;
  T grad_ = 0;
  T hess_ = 0;

 public:
  using ValueT = T;

  XGBOOST_DEVICE GradientPairInt64(T grad, T hess) : grad_(grad), hess_(hess) {}
  GradientPairInt64() = default;

  // Copy constructor if of same value type, marked as default to be trivially_copyable
  GradientPairInt64(GradientPairInt64 const &g) = default;
  GradientPairInt64 &operator=(GradientPairInt64 const &g) = default;

  [[nodiscard]] XGBOOST_DEVICE T GetQuantisedGrad() const { return grad_; }
  [[nodiscard]] XGBOOST_DEVICE T GetQuantisedHess() const { return hess_; }

  XGBOOST_DEVICE GradientPairInt64 &operator+=(const GradientPairInt64 &rhs) {
    grad_ += rhs.grad_;
    hess_ += rhs.hess_;
    return *this;
  }

  XGBOOST_DEVICE GradientPairInt64 operator+(const GradientPairInt64 &rhs) const {
    GradientPairInt64 g;
    g.grad_ = grad_ + rhs.grad_;
    g.hess_ = hess_ + rhs.hess_;
    return g;
  }

  XGBOOST_DEVICE GradientPairInt64 &operator-=(const GradientPairInt64 &rhs) {
    grad_ -= rhs.grad_;
    hess_ -= rhs.hess_;
    return *this;
  }

  XGBOOST_DEVICE GradientPairInt64 operator-(const GradientPairInt64 &rhs) const {
    GradientPairInt64 g;
    g.grad_ = grad_ - rhs.grad_;
    g.hess_ = hess_ - rhs.hess_;
    return g;
  }

  XGBOOST_DEVICE bool operator==(const GradientPairInt64 &rhs) const {
    return grad_ == rhs.grad_ && hess_ == rhs.hess_;
  }
  friend std::ostream &operator<<(std::ostream &os, const GradientPairInt64 &g) {
    os << g.GetQuantisedGrad() << "/" << g.GetQuantisedHess();
    return os;
  }
};

using Args = std::vector<std::pair<std::string, std::string> >;

/*! \brief small eps gap for minimum split decision. */
constexpr bst_float kRtEps = 1e-6f;

/*! \brief define unsigned long for openmp loop */
using omp_ulong = dmlc::omp_ulong;  // NOLINT
/*! \brief define unsigned int for openmp loop */
using bst_omp_uint = dmlc::omp_uint;  // NOLINT
/*! \brief Type used for representing version number in binary form.*/
using XGBoostVersionT = std::int32_t;
}  // namespace xgboost

#endif  // XGBOOST_BASE_H_
