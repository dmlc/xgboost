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

// A macro used inside `windows.h` to avoid conflicts with `winsock2.h`
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif  // !defined(WIN32_LEAN_AND_MEAN)

// Stop windows.h from including winsock.h
// mingw uses _WINSOCKAPI_ to check whether windows.h is included, don't mess with it.
#if !defined(_WINSOCKAPI_) && !defined(__MINGW32__)
#define _WINSOCKAPI_
#endif  // !defined(_WINSOCKAPI_)

#if !defined(xgboost_IS_MINGW)

#if defined(__MINGW32__)
#define xgboost_IS_MINGW 1
#endif  // defined(__MINGW32__)

#endif  // xgboost_IS_MINGW

#endif  // !defined(xgboost_IS_WIN)
