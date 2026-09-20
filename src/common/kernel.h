/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_KERNEL_H_
#define XGBOOST_COMMON_KERNEL_H_

#include <type_traits>  // for is_function_v, is_invocable_v
#include <utility>      // for forward
#include <vector>       // for vector

#include "xgboost/context.h"  // for Context, DeviceOrd
#include "xgboost/logging.h"  // for CHECK

namespace xgboost::common {
template <typename Kernel>
class KernelRegistry {
 public:
  using Signature = typename Kernel::Signature;
  static_assert(std::is_function_v<Signature>, "A kernel signature must be a function type.");
  using Function = std::add_pointer_t<Signature>;

  void Register(DeviceOrd::Type device, Function function) {
    CHECK(function);
    for (auto const& entry : entries_) {
      CHECK_NE(entry.device, device)
          << "A kernel implementation is already registered for device type " << device;
    }
    entries_.push_back({device, function});
  }

  Function Find(DeviceOrd::Type device) const {
    for (auto const& entry : entries_) {
      if (entry.device == device) {
        return entry.function;
      }
    }
    return nullptr;
  }

 private:
  struct Entry {
    DeviceOrd::Type device;
    Function function;
  };

  std::vector<Entry> entries_;
};

template <typename Kernel>
KernelRegistry<Kernel>& GetKernelRegistry() {
  static KernelRegistry<Kernel> registry;
  return registry;
}

template <typename Kernel>
class KernelRegistration {
 public:
  using Function = typename KernelRegistry<Kernel>::Function;

  KernelRegistration(DeviceOrd::Type device, Function implementation) {
    GetKernelRegistry<Kernel>().Register(device, implementation);
  }
};

template <typename Kernel, typename... Args>
decltype(auto) DispatchKernel(Context const* ctx, Args&&... args) {
  CHECK(ctx);
  using Function = typename KernelRegistry<Kernel>::Function;
  static_assert(std::is_invocable_v<Function, Context const*, Args...>,
                "Kernel arguments do not match its declared signature.");

  auto function = GetKernelRegistry<Kernel>().Find(ctx->Device().device);
  if (function) {
    return function(ctx, std::forward<Args>(args)...);
  }

  auto cpu_ctx = ctx->MakeCPU();
  function = GetKernelRegistry<Kernel>().Find(cpu_ctx.Device().device);
  CHECK(function) << "No kernel variant supports device " << ctx->Device().Name();
  return function(&cpu_ctx, std::forward<Args>(args)...);
}
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_KERNEL_H_
