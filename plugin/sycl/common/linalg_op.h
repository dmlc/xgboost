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

struct Argument {
    template <typename T>
    operator T&&() const;
};

template <typename Fn, typename Is, typename = void>
struct ArgumentsPassedImpl
       : std::false_type {};

template <typename Fn, size_t ...Is>
struct ArgumentsPassedImpl<Fn, std::index_sequence<Is...>,
                       decltype(std::declval<Fn>()(((void)Is, Argument{})...), void())>
      : std::true_type {};

template <typename Fn, size_t N>
struct ArgumentsPassed : ArgumentsPassedImpl<Fn, std::make_index_sequence<N>> {};

template <typename OutputDType, typename InputDType,
          size_t BatchSize, size_t MaxNumInputs>
class BatchProcessingHelper {
 public:
  static constexpr size_t kBatchSize = BatchSize;
  using InputType = HostDeviceVector<InputDType>;
  using OutputType = HostDeviceVector<OutputDType>;

 private:
  template <size_t NumInput = 0>
  void Host2Buffers(InputDType* in_buffer_ptr, const InputType& input) {
    /* 
     * Some inputs may have less than 1 sample per output symbol.
     */
    const size_t sub_sample_rate = ndata_ * sample_rates_[NumInput+1] / input.Size();
    const size_t n_samples = batch_size_ * sample_rates_[NumInput+1] / sub_sample_rate;

    const InputDType* in_host_ptr = input.HostPointer() +
                                    batch_begin_ * sample_rates_[NumInput+1] / sub_sample_rate;

    events_[NumInput] =
      qu_->memcpy(in_buffer_ptr, in_host_ptr, n_samples * sizeof(InputDType),
                  events_[MaxNumInputs - 2]);
  }

  template <size_t NumInput = 0, class... InputTypes>
  void Host2Buffers(InputDType* in_buffer_ptr, const InputType& input,
                    const InputTypes&... other_inputs) {
    // Make copy for the first input in the list
    Host2Buffers<NumInput>(in_buffer_ptr, input);
    // Recurent call for next inputs
    InputDType* next_input = in_buffer_.Data() + in_buff_offsets_[NumInput + 1];
    Host2Buffers<NumInput+1>(next_input, other_inputs...);
  }

  void Buffers2Host(OutputType* output) {
    const size_t n_samples = batch_size_ * sample_rates_[0];
    OutputDType* out_host_ptr = output->HostPointer() + batch_begin_* sample_rates_[0];
    events_[MaxNumInputs - 1] =
      qu_->memcpy(out_host_ptr, out_buffer_.DataConst(), n_samples * sizeof(OutputDType),
                  events_[MaxNumInputs - 2]);
  }

  void Buffers2Host(InputType* output) {
    const size_t n_samples = batch_size_ * sample_rates_[1];
    InputDType* out_host_ptr = output->HostPointer() + batch_begin_* sample_rates_[1];
    events_[MaxNumInputs - 1] =
      qu_->memcpy(out_host_ptr, in_buffer_.DataConst(), n_samples * sizeof(InputDType),
                  events_[MaxNumInputs - 2]);
  }

  template <size_t NumInputs = 1, typename Fn, class... InputTypes>
  void Call(Fn &&fn, const InputDType* input, const InputTypes*... other_inputs) {
    static_assert(NumInputs <= MaxNumInputs,
                  "To many arguments in the passed function");
    /* Passed lambda may have less inputs than MaxNumInputs,
     * need to pass only requared number of arguments
     */
    // 1 for events, 1 for batch_size, 1 for output
    if constexpr (ArgumentsPassed<Fn, NumInputs + 1 + 1 + 1>::value) {
      events_[MaxNumInputs - 2] = fn(events_, batch_size_,
                                     out_buffer_.Data(), input, other_inputs...);
    } else {
      const InputDType* next_input = in_buffer_.DataConst() +
                                     in_buff_offsets_[MaxNumInputs - 1 - NumInputs];
      Call<NumInputs+1>(std::forward<Fn>(fn), next_input, input, other_inputs...);
    }
  }

  template <size_t NumInputs = 1, typename Fn, class... InputTypes>
  void Call(Fn &&fn, InputDType* io, const InputDType* input, const InputTypes*... other_inputs) {
    static_assert(NumInputs <= MaxNumInputs,
                  "To many arguments in the passed function");
    if constexpr (ArgumentsPassed<Fn, NumInputs + 1 + 1>::value) {
      events_[MaxNumInputs - 2] = fn(events_, batch_size_,
                                     io, input, other_inputs...);
    } else {
      const InputDType* next_input = in_buffer_.DataConst() +
                                     in_buff_offsets_[MaxNumInputs - NumInputs];
      Call<NumInputs+1>(std::forward<Fn>(fn), io, next_input, input, other_inputs...);
    }
  }

  template <size_t NumInputs = 1, typename Fn>
  void Call(Fn &&fn, InputDType* io) {
    static_assert(NumInputs <= MaxNumInputs,
                  "To many arguments in the passed function");
    if constexpr (ArgumentsPassed<Fn, NumInputs + 1 + 1>::value) {
      events_[MaxNumInputs - 2] = fn(events_, batch_size_, io);
    } else {
      const InputDType* next_input = in_buffer_.DataConst() +
                                     in_buff_offsets_[MaxNumInputs - 1];
      Call<NumInputs+1>(std::forward<Fn>(fn), io, next_input);
    }
  }

 public:
  BatchProcessingHelper() = default;

  // The first element of sample_rate always corresonds to output sample rate
  void InitBuffers(::sycl::queue* qu, const std::vector<int>& sample_rate) {
    assert(sample_rate.size() == MaxNumInputs + 1);
    sample_rates_ = sample_rate;
    qu_ = qu;
    events_.resize(MaxNumInputs + 2);
    out_buffer_.Resize(qu, kBatchSize * sample_rate.front());

    in_buff_offsets_[0] = 0;
    for (size_t i = 1; i < MaxNumInputs; ++i) {
      in_buff_offsets_[i] = in_buff_offsets_[i - 1] + kBatchSize * sample_rate[i];
    }
    const size_t in_buff_size = in_buff_offsets_.back() + kBatchSize * sample_rate.back();
    in_buffer_.Resize(qu, in_buff_size);
  }

  /*
   * Batch-wise proces on sycl device
   * output = fn(inputs)
   */
  template <typename Fn, class... InputTypes>
  void Calculate(Fn &&fn, OutputType* output, const InputTypes&... inputs) {
    ndata_ = output->Size() / sample_rates_.front();
    const size_t nBatch = ndata_ / kBatchSize + (ndata_ % kBatchSize > 0);
    for (size_t batch = 0; batch < nBatch; ++batch) {
      batch_begin_ = batch * kBatchSize;
      batch_end_ = (batch == nBatch - 1) ? ndata_ : batch_begin_ + kBatchSize;
      batch_size_ = batch_end_ - batch_begin_;

      // Iteratively copy all inputs to device buffers
      Host2Buffers(in_buffer_.Data(), inputs...);
      // Pack buffers and call function
      // We shift input pointer to keep the same order of inputs after packing
      Call(std::forward<Fn>(fn), in_buffer_.DataConst() + in_buff_offsets_.back());
      // Copy results to host
      Buffers2Host(output);
    }
  }

  /*
   * Batch-wise proces on sycl device
   * input = fn(input, other_inputs)
   */
  template <typename Fn, class... InputTypes>
  void Calculate(Fn &&fn, InputType* input, const InputTypes&... other_inputs) {
    ndata_ = input->Size();
    const size_t nBatch = ndata_ / kBatchSize + (ndata_ % kBatchSize > 0);
    for (size_t batch = 0; batch < nBatch; ++batch) {
      batch_begin_ = batch * kBatchSize;
      batch_end_ = (batch == nBatch - 1) ? ndata_ : batch_begin_ + kBatchSize;
      batch_size_ = batch_end_ - batch_begin_;

      // Iteratively copy all inputs to device buffers.
      // inputs are pased by const reference
      Host2Buffers(in_buffer_.Data(), *(input), other_inputs...);
      // Pack buffers and call function
      // We shift input pointer to keep the same order of inputs after packing
      Call(std::forward<Fn>(fn), in_buffer_.Data());
      // Copy results to host
      Buffers2Host(input);
    }
  }

 private:
  std::array<int, MaxNumInputs> in_buff_offsets_;
  std::vector<int> sample_rates_;
  size_t ndata_;
  size_t batch_begin_;
  size_t batch_end_;
  // is not equal to kBatchSize for the last batch
  size_t batch_size_;
  ::sycl::queue* qu_;
  std::vector<::sycl::event> events_;
  USMVector<InputDType, MemoryType::on_device> in_buffer_;
  USMVector<OutputDType, MemoryType::on_device> out_buffer_;
};

}  // namespace linalg
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_LINALG_OP_H_
