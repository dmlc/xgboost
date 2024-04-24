/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_updater.cc
 */

#include "hist_updater.h"

#include <oneapi/dpl/random>

namespace xgboost {
namespace sycl {
namespace tree {

template<typename GradientSumT>
void HistUpdater<GradientSumT>::InitSampling(
      const USMVector<GradientPair, MemoryType::on_device> &gpair,
      USMVector<size_t, MemoryType::on_device>* row_indices) {
  const size_t num_rows = row_indices->Size();
  auto* row_idx = row_indices->Data();
  const auto* gpair_ptr = gpair.DataConst();
  uint64_t num_samples = 0;
  const auto subsample = param_.subsample;
  ::sycl::event event;

  {
    ::sycl::buffer<uint64_t, 1> flag_buf(&num_samples, 1);
    uint64_t seed = seed_;
    seed_ += num_rows;
    event = qu_.submit([&](::sycl::handler& cgh) {
      auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(num_rows)),
                                          [=](::sycl::item<1> pid) {
        uint64_t i = pid.get_id(0);

        // Create minstd_rand engine
        oneapi::dpl::minstd_rand engine(seed, i);
        oneapi::dpl::bernoulli_distribution coin_flip(subsample);

        auto rnd = coin_flip(engine);
        if (gpair_ptr[i].GetHess() >= 0.0f && rnd) {
          AtomicRef<uint64_t> num_samples_ref(flag_buf_acc[0]);
          row_idx[num_samples_ref++] = i;
        }
      });
    });
    /* After calling a destructor for flag_buf,  content will be copyed to num_samples */
  }

  row_indices->Resize(&qu_, num_samples, 0, &event);
  qu_.wait();
}

template class HistUpdater<float>;
template class HistUpdater<double>;

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
