/*!
 * Copyright 2019 by Contributors
 * \file build_config.h
 */
#ifndef XGBOOST_BUILD_CONFIG_H_
#define XGBOOST_BUILD_CONFIG_H_

/* default logic for software pre-fetching */
#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_AMD64))) || defined(__INTEL_COMPILER)
// Enable _mm_prefetch for Intel compiler and MSVC+x86
  #define XGBOOST_MM_PREFETCH_PRESENT
  #define XGBOOST_BUILTIN_PREFETCH_PRESENT
#elif defined(__GNUC__)
// Enable __builtin_prefetch for GCC
#define XGBOOST_BUILTIN_PREFETCH_PRESENT
#endif

#endif  // XGBOOST_BUILD_CONFIG_H_