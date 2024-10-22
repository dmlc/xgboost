/**
 * Copyright 2021-2024, XGBoost Contributors
 * \file linalg_op.h
 */
#ifndef PLUGIN_SYCL_COMMON_LINALG_OP_H_
#define PLUGIN_SYCL_COMMON_LINALG_OP_H_

#include <vector>
#include <utility>

#include "../data.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace linalg {

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
}  // namespace linalg
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_LINALG_OP_H_
