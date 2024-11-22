/**
 * Copyright 2021-2024, XGBoost Contributors
 * \file linalg_op.h
 */
#ifndef PLUGIN_SYCL_COMMON_LINALG_OP_H_
#define PLUGIN_SYCL_COMMON_LINALG_OP_H_

#include <vector>
#include <utility>

#include "../data.h"
#include "../device_manager.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace linalg {

template<typename T, std::int32_t D>
using TensorView = xgboost::linalg::TensorView<T, D>;

struct WorkGroupsParams {
  size_t n_workgroups;
  size_t workgroup_size;
};

template <typename Fn>
::sycl::event GroupWiseKernel(::sycl::queue* qu, int* flag_ptr,
                              const std::vector<::sycl::event>& events,
                              const WorkGroupsParams& wg, Fn &&fn) {
  ::sycl::buffer<int, 1> flag_buf(flag_ptr, 1);
  auto event = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto flag  = flag_buf.get_access<::sycl::access::mode::write>(cgh);
    cgh.parallel_for_work_group<>(::sycl::range<1>(wg.n_workgroups),
                                  ::sycl::range<1>(wg.workgroup_size),
                                  [=](::sycl::group<1> group) {
      group.parallel_for_work_item([&](::sycl::h_item<1> item) {
        const size_t idx = item.get_global_id()[0];
        fn(idx, flag);
      });
    });
  });
  return event;
}

template<typename Fn, typename TupleType, size_t ... I>
auto call(Fn&& fn, TupleType t, std::index_sequence<I ...>) {
     return fn(std::get<I>(t) ...);
}

template<typename Fn, typename TupleType>
auto call(Fn&& fn, TupleType t) {
    static constexpr auto size = std::tuple_size<TupleType>::value;
    return call(fn, t, std::make_index_sequence<size>{});
}

template <typename T, int32_t D, typename Fn>
void ElementWiseKernel(TensorView<T, D> t, Fn&& fn) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(t.Device());
  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(t.Size()),
                       [=](::sycl::id<1> pid) {
      const size_t idx = pid[0];
      call(const_cast<Fn&&>(fn), xgboost::linalg::UnravelIndex(idx, t.Shape()));
    });
  }).wait_and_throw();
}

}  // namespace linalg
}  // namespace sycl

namespace linalg {
template <typename T, int32_t D, typename Fn>
void ElementWiseKernel(Context const* ctx, TensorView<T, D> t, Fn&& fn) {
  if (ctx->IsSycl()) {
    sycl::linalg::ElementWiseKernel(t, fn);
  } else {
    ElementWiseKernelHost(t, ctx->Threads(), fn);
  }
}

}  // namespace linalg
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_LINALG_OP_H_
