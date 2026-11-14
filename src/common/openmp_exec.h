/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once
#include <dmlc/common.h>  // for OMPException
#include <dmlc/omp.h>     // for omp_get_thread_num, omp_get_max_threads, omp_in_parallel

#include <algorithm>  // for min
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t

#include "executor.h"         // for Sched, BindWorker
#include "xgboost/logging.h"  // for CHECK_GE

#if defined(_MSC_VER)
#include <limits>       // for numeric_limits
#include <type_traits>  // for conditional_t, is_signed

#include "xgboost/base.h"  // for omp_ulong
#endif

namespace xgboost::common {
// Without OpenMP, the pragmas are ignored and dmlc's OpenMP stubs run loops serially.
struct OpenMPExecutor {
  /**
   * @brief Whether a parallel region always uses the requested number of threads.
   *
   * Regions smaller than the thread count normally get a smaller team, which saves waking idle
   * threads. GCC's libgomp, however, retires the extra threads when a team shrinks and starts
   * new ones when it grows again, so team sizes that vary from region to region (as in a
   * lossguide tree, with many tiny regions per node) are many times slower than a fixed team.
   * With libgomp the team therefore stays at the requested size and threads without work skip
   * the loop. The check keys on libgomp's omp.h, a compile-time proxy for the runtime that is
   * loaded at run time; it is false without OpenMP.
   */
#if defined(_LIBGOMP_OMP_LOCK_DEFINED)
  static constexpr bool kFixedTeam = true;
#else
  static constexpr bool kFixedTeam = false;
#endif

  [[nodiscard]] static std::int32_t MaxNumWorkers() noexcept {
#if !defined(_OPENMP)
    return 1;
#else
    if (omp_in_parallel()) {
      return 1;
    }
#if defined(_MSC_VER)
    // MSVC doesn't implement omp_get_thread_limit.
    auto limit = std::numeric_limits<std::int32_t>::max();
#else
    auto limit = omp_get_thread_limit();
#endif
    CHECK_GE(limit, 1) << "Invalid thread limit for OpenMP.";
    return std::min({omp_get_num_procs(), omp_get_max_threads(), limit});
#endif
  }

  template <typename Index, typename Func>
  static void Execute(Index size, std::int32_t n_threads, Sched sched, Func&& fn) {
#if defined(_MSC_VER)
    // msvc doesn't support unsigned integer as openmp index.
    using OmpInd = std::conditional_t<std::is_signed<Index>::value, Index, omp_ulong>;
#else
    using OmpInd = Index;
#endif
    OmpInd length = static_cast<OmpInd>(size);
    CHECK_GE(n_threads, 1);
    if (size <= 0) {
      return;
    }
    if (!kFixedTeam) {
      n_threads = static_cast<std::int32_t>(std::min<std::size_t>(n_threads, size));
    }

    auto&& callback = detail::BindWorker(fn, [] { return Worker{omp_get_thread_num()}; });
    dmlc::OMPException exc;
    switch (sched.sched) {
      case Sched::kAuto: {
#pragma omp parallel for num_threads(n_threads)
        for (OmpInd i = 0; i < length; ++i) {
          exc.Run(callback, i);
        }
        break;
      }
      case Sched::kDynamic: {
        if (sched.chunk == 0) {
#pragma omp parallel for num_threads(n_threads) schedule(dynamic)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        } else {
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, sched.chunk)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        }
        break;
      }
      case Sched::kStatic: {
        if (sched.chunk == 0) {
#pragma omp parallel for num_threads(n_threads) schedule(static)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        } else {
#pragma omp parallel for num_threads(n_threads) schedule(static, sched.chunk)
          for (OmpInd i = 0; i < length; ++i) {
            exc.Run(callback, i);
          }
        }
        break;
      }
      case Sched::kGuided: {
#pragma omp parallel for num_threads(n_threads) schedule(guided)
        for (OmpInd i = 0; i < length; ++i) {
          exc.Run(callback, i);
        }
        break;
      }
    }
    exc.Rethrow();
  }

  // Wrapper to implement nested parallelism with simple omp parallel for
  template <typename Space, typename Func>
  static void Execute2d(Space const& space, std::int32_t n_threads, Func&& fn) {
    std::size_t n_blocks_in_space = space.Size();
    CHECK_GE(n_threads, 1);
#if !defined(_OPENMP)
    n_threads = 1;
#endif
    if (n_blocks_in_space == 0) {
      return;
    }
    if (n_blocks_in_space == 1) {
      // Run a single block on the calling thread without starting a parallel region.
      auto&& callback = detail::BindWorker(fn, [] { return Worker{0}; });
      callback(space.GetFirstDimension(0), space.GetRange(0));
      return;
    }
    if (!kFixedTeam) {
      n_threads = static_cast<std::int32_t>(std::min<std::size_t>(n_threads, n_blocks_in_space));
    }

    dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
    {
      exc.Run([&]() {
        std::size_t tid = omp_get_thread_num();
        auto&& callback =
            detail::BindWorker(fn, [tid] { return Worker{static_cast<std::int32_t>(tid)}; });
        std::size_t chunck_size = n_blocks_in_space / n_threads + !!(n_blocks_in_space % n_threads);

        std::size_t begin = chunck_size * tid;
        std::size_t end = std::min(begin + chunck_size, n_blocks_in_space);
        for (auto i = begin; i < end; i++) {
          callback(space.GetFirstDimension(i), space.GetRange(i));
        }
      });
    }
    exc.Rethrow();
  }
};
}  // namespace xgboost::common
