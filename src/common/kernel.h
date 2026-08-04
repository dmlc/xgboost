/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_KERNEL_H_
#define XGBOOST_COMMON_KERNEL_H_

#include <string_view>  // for string_view
#include <type_traits>  // for is_function_v, is_invocable_v
#include <utility>      // for forward
#include <vector>       // for vector

#include "xgboost/context.h"  // for Context, DeviceOrd
#include "xgboost/logging.h"  // for CHECK

namespace xgboost::common {
using DeviceMatcher = bool (*)(DeviceOrd);

inline bool MatchCPU(DeviceOrd device) { return device.IsCPU(); }
inline bool MatchCUDA(DeviceOrd device) { return device.IsCUDA(); }

template <typename Kernel>
class KernelRegistry {
 public:
  using Signature = typename Kernel::Signature;
  static_assert(std::is_function_v<Signature>, "A kernel signature must be a function type.");
  using Function = std::add_pointer_t<Signature>;

  struct Entry {
    char const* name;
    DeviceMatcher matcher;
    Function function;
  };

  static void Register(char const* name, DeviceMatcher matcher, Function function) {
    CHECK(name);
    CHECK(matcher);
    CHECK(function);
    auto& entries = Entries();
    for (auto const& entry : entries) {
      CHECK_NE(std::string_view{entry.name}, std::string_view{name})
          << "Duplicate kernel variant: " << name;
      CHECK_NE(entry.matcher, matcher) << "Duplicate device matcher for kernel variant: " << name;
    }
    entries.push_back({name, matcher, function});
  }

  static Function Find(DeviceOrd device) {
    Function result{nullptr};
    for (auto const& entry : Entries()) {
      if (entry.matcher(device)) {
        CHECK_EQ(result, nullptr) << "Multiple kernel variants support device " << device.Name();
        result = entry.function;
      }
    }
    return result;
  }

 private:
  static std::vector<Entry>& Entries() {
    static std::vector<Entry> entries;
    return entries;
  }
};

template <typename Kernel, auto Implementation>
class KernelRegistration {
 public:
  using Function = typename KernelRegistry<Kernel>::Function;
  static_assert(std::is_same_v<decltype(Implementation), Function>,
                "Kernel implementation does not match its declared signature.");

  KernelRegistration(char const* name, DeviceMatcher matcher) {
    KernelRegistry<Kernel>::Register(name, matcher, Implementation);
  }
};

template <typename Kernel, typename... Args>
decltype(auto) DispatchKernel(Context const* ctx, Args&&... args) {
  using Function = typename KernelRegistry<Kernel>::Function;
  static_assert(std::is_invocable_v<Function, Context const*, Args...>,
                "Kernel arguments do not match its declared signature.");

  auto function = KernelRegistry<Kernel>::Find(ctx->Device());
  if (function) {
    return function(ctx, std::forward<Args>(args)...);
  }

  auto cpu_ctx = ctx->MakeCPU();
  function = KernelRegistry<Kernel>::Find(cpu_ctx.Device());
  CHECK(function) << "No kernel variant supports device " << ctx->Device().Name();
  return function(&cpu_ctx, std::forward<Args>(args)...);
}
}  // namespace xgboost::common

#define XGBOOST_KERNEL_CONCAT_IMPL_(Left, Right) Left##Right
#define XGBOOST_KERNEL_CONCAT_(Left, Right) XGBOOST_KERNEL_CONCAT_IMPL_(Left, Right)
#define XGBOOST_REGISTER_KERNEL(Kernel, Name, Matcher, Implementation)                  \
  [[maybe_unused]] static ::xgboost::common::KernelRegistration<Kernel, Implementation> \
  XGBOOST_KERNEL_CONCAT_(__xgboost_kernel_registration_, __COUNTER__) {                 \
    Name, Matcher                                                                       \
  }

#endif  // XGBOOST_COMMON_KERNEL_H_
