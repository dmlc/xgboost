/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once

#if defined(XGBOOST_USE_NVTX)
#include <nvtx3/nvtx3.hpp>
#endif  // defined(XGBOOST_USE_NVTX)

#include "xgboost/string_view.h"  // for StringView

namespace xgboost::nvtx {
struct Domain {
  static constexpr char const* name{"libxgboost"};  // NOLINT
};

#if defined(XGBOOST_USE_NVTX)
using ScopedRange = ::nvtx3::scoped_range_in<Domain>;
using EventAttr = ::nvtx3::event_attributes;
using Rgb = ::nvtx3::rgb;

inline auto MakeScopedRange(StringView name, Rgb color) {
  ::nvtx3::v1::registered_string_in<Domain> const scope_name{name.c_str()};
  ::nvtx3::v1::event_attributes const scope_attr{scope_name, color};
  return ::nvtx3::v1::scoped_range_in<Domain>{scope_attr};
}

#else
class ScopedRange {
 public:
  template <typename... Args>
  explicit ScopedRange(Args&&...) {}
};
class EventAttr {
 public:
  template <typename... Args>
  explicit EventAttr(Args&&...) {}
};
class Rgb {
 public:
  template <typename... Args>
  explicit Rgb(Args&&...) {}
};

inline auto MakeScopedRange(StringView, Rgb) { return ScopedRange{}; }
#endif  // defined(XGBOOST_USE_NVTX)
}  // namespace xgboost::nvtx

#if defined(XGBOOST_USE_NVTX)

// Macro for making NVTX function range.
#define xgboost_NVTX_FN_RANGE() NVTX3_FUNC_RANGE_IN(::xgboost::nvtx::Domain)

// Macro for making colored NVTX function range.
#define xgboost_NVTX_FN_RANGE_C(r, g, b) \
  auto __nvtx_scoped__ = ::xgboost::nvtx::MakeScopedRange(__func__, (nvtx::Rgb((r), (g), (b))))

#else

#define xgboost_NVTX_FN_RANGE()

#define xgboost_NVTX_FN_RANGE_C(r, g, b)

#endif  // defined(XGBOOST_USE_NVTX)
