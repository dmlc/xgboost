/**
 * Copyright 2024, XGBoost Contributors
 *
 * @brief Macro for Windows.
 */
#pragma once

#if !defined(xgboost_IS_WIN)

#if defined(_MSC_VER) || defined(__MINGW32__)
#define xgboost_IS_WIN 1
#endif  // defined(_MSC_VER) || defined(__MINGW32__)

#endif  // !defined(xgboost_IS_WIN)

#if defined(xgboost_IS_WIN)

#if !defined(NOMINMAX)
#define NOMINMAX
#endif  // !defined(NOMINMAX)

#if !defined(xgboost_IS_MINGW)

#if defined(__MINGW32__)
#define xgboost_IS_MINGW 1
#endif  // defined(__MINGW32__)

#endif  // xgboost_IS_MINGW

#endif  // !defined(xgboost_IS_WIN)
