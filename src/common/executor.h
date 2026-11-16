/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <utility>  // for forward, move

namespace xgboost::common {
/**
 * @brief Execution metadata for one invocation of a parallel loop.
 *
 * IDs satisfy 0 <= Id() < requested threads. Serial calls supply ID 0,
 * including nested serial calls. IDs do not guarantee a work assignment
 * across separate loops.
 */
class Worker {
  std::int32_t id_;

 public:
  explicit Worker(std::int32_t id) : id_{id} {}

  [[nodiscard]] std::int32_t Id() const { return id_; }
};

/** @brief Opt in to receiving Worker as the final callback argument. */
template <typename Fn>
class WithWorker {
  Fn fn_;

 public:
  explicit WithWorker(Fn fn) : fn_{std::move(fn)} {}

  template <typename... Args>
  decltype(auto) operator()(Worker worker, Args&&... args) {
    return fn_(std::forward<Args>(args)..., worker);
  }
};

namespace detail {
// Bind an explicit worker only for callbacks that opt in with WithWorker.
template <typename Fn, typename GetWorker>
Fn& BindWorker(Fn& fn, GetWorker) {  // NOLINT
  return fn;
}

template <typename Fn, typename GetWorker>
auto BindWorker(WithWorker<Fn>& fn, GetWorker get_worker) {  // NOLINT
  return [&fn, get_worker](auto&&... args) -> decltype(auto) {
    return fn(get_worker(), std::forward<decltype(args)>(args)...);
  };
}

// Adapt loop arguments while preserving whether the callback needs a worker ID.
template <typename Fn, typename Adapter>
auto TransformCallback(Fn& fn, Adapter adapter) {  // NOLINT
  return [&fn, adapter](auto i) {
    return adapter(fn, i);
  };
}

template <typename Fn, typename Adapter>
auto TransformCallback(WithWorker<Fn>& fn, Adapter adapter) {  // NOLINT
  return WithWorker{[&fn, adapter](auto i, Worker worker) {
    auto callback = BindWorker(fn, [worker] { return worker; });
    return adapter(callback, i);
  }};
}
}  // namespace detail

/**
 * Parallel loop scheduling hint
 */
struct Sched {
  enum {
    kAuto,
    kDynamic,
    kStatic,
    kGuided,
  } sched;
  size_t chunk{0};

  Sched static Auto() { return Sched{kAuto}; }
  Sched static Dyn(size_t n = 0) { return Sched{kDynamic, n}; }
  Sched static Static(size_t n = 0) { return Sched{kStatic, n}; }
  Sched static Guided() { return Sched{kGuided}; }
};
}  // namespace xgboost::common
