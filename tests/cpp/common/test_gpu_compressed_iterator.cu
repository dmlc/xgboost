/**
 * Copyright 2018-2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>  // for sequence

#include <algorithm>  // for generate
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint32_t
#include <vector>     // for vector

#include "../../../src/common/compressed_iterator.h"
#include "../../../src/common/cuda_context.cuh"    // for CUDAContext
#include "../../../src/common/device_helpers.cuh"  // for LaunchN
#include "../../../src/common/device_vector.cuh"   // for DeviceUVector
#include "../helpers.h"

namespace xgboost::common {
struct WriteSymbolFunction {
  CompressedBufferWriter cbw;
  unsigned char* buffer_data_d;
  int const* input_data_d;
  WriteSymbolFunction(CompressedBufferWriter cbw, unsigned char* buffer_data_d,
                      int const* input_data_d)
      : cbw(cbw), buffer_data_d(buffer_data_d), input_data_d(input_data_d) {}

  __device__ void operator()(size_t i) { cbw.AtomicWriteSymbol(buffer_data_d, input_data_d[i], i); }
};

struct ReadSymbolFunction {
  CompressedIterator<int> ci;
  int* output_data_d;
  ReadSymbolFunction(CompressedIterator<int> ci, int* output_data_d)
    : ci(ci), output_data_d(output_data_d) {}

  __device__ void operator()(size_t i) {
    output_data_d[i] = ci[i];
  }
};

TEST(CompressedIterator, TestGPU) {
  dh::safe_cuda(cudaSetDevice(0));
  std::vector<int> test_cases = {1, 3, 426, 21, 64, 256, 100000, INT32_MAX};
  int num_elements = 1000;
  int repetitions = 1000;
  srand(9);

  for (auto alphabet_size : test_cases) {
    for (int i = 0; i < repetitions; i++) {
      std::vector<int> input(num_elements);
      std::generate(input.begin(), input.end(),
        [=]() { return rand() % alphabet_size; });
      CompressedBufferWriter cbw(alphabet_size);
      thrust::device_vector<int> input_d(input);

      thrust::device_vector<unsigned char> buffer_d(
        CompressedBufferWriter::CalculateBufferSize(input.size(),
          alphabet_size));

      // write the data on device
      auto input_data_d = input_d.data().get();
      auto buffer_data_d = buffer_d.data().get();
      dh::LaunchN(input_d.size(),
                  WriteSymbolFunction(cbw, buffer_data_d, input_data_d));

      // read the data on device
      CompressedIterator<int> ci(buffer_d.data().get(), alphabet_size);
      thrust::device_vector<int> output_d(input.size());
      auto output_data_d = output_d.data().get();
      dh::LaunchN(output_d.size(), ReadSymbolFunction(ci, output_data_d));

      std::vector<int> output(output_d.size());
      thrust::copy(output_d.begin(), output_d.end(), output.begin());

      ASSERT_TRUE(input == output);
    }
  }
}

namespace {
class TestDoubleCompressedIter : public ::testing::TestWithParam<std::size_t> {
 public:
  constexpr std::size_t static CompressedBytes() { return 24; }

 private:
  dh::DeviceUVector<std::int32_t> input_;
  Context ctx_{MakeCUDACtx(0)};
  std::size_t n_symbols_{11};

  void SetUp() override {
    input_.resize(n_symbols_ * 3);
    auto policy = ctx_.CUDACtx()->CTP();
    for (std::size_t i = 0; i < 3; ++i) {
      auto beg = input_.begin() + n_symbols_ * i;
      auto end = beg + n_symbols_;
      thrust::sequence(policy, beg, end, 0);
    }
  }

 public:
  void Run(std::size_t n0_bytes) const {
    auto policy = ctx_.CUDACtx()->CTP();

    auto compressed_nbytes = CompressedBufferWriter::CalculateBufferSize(input_.size(), n_symbols_);
    ASSERT_EQ(compressed_nbytes, CompressedBytes());

    dh::device_vector<CompressedByteT> buf(compressed_nbytes, 0);
    CompressedBufferWriter cbw(n_symbols_);
    dh::LaunchN(input_.size(), ctx_.CUDACtx()->Stream(),
                WriteSymbolFunction{cbw, buf.data().get(), input_.data()});

    dh::device_vector<CompressedByteT> buf0(n0_bytes);
    dh::device_vector<CompressedByteT> buf1(compressed_nbytes - buf0.size());
    thrust::copy_n(policy, buf.begin(), buf0.size(), buf0.begin());
    thrust::copy_n(policy, buf.begin() + buf0.size(), buf1.size(), buf1.begin());

    HostDeviceVector<std::int32_t> output(input_.size(), 0, ctx_.Device());
    auto it = DoubleCompressedIter<std::uint32_t>{buf0.data().get(), buf0.size(), buf1.data().get(),
                                                  n_symbols_};
    auto d_out = output.DeviceSpan();
    dh::LaunchN(input_.size(), ctx_.CUDACtx()->Stream(),
                [=] __device__(std::size_t i) { d_out[i] = it[i]; });
    auto h_out = output.ConstHostVector();
    for (std::size_t i = 0; i < 3; ++i) {
      auto beg = h_out.begin() + n_symbols_ * i;
      auto end = beg + n_symbols_;
      std::size_t k = 0;
      for (auto it = beg; it != end; ++it) {
        ASSERT_EQ(*it, k);
        k++;
      }
    }
  }
};

inline auto kCnBytes = TestDoubleCompressedIter::CompressedBytes();
}  // namespace

TEST_P(TestDoubleCompressedIter, Basic) {
  auto n0_bytes = this->GetParam();
  this->Run(n0_bytes);
}

INSTANTIATE_TEST_SUITE_P(Gpu, TestDoubleCompressedIter,
                         ::testing::Values(0, kCnBytes, 1, kCnBytes - 1, kCnBytes / 2, kCnBytes / 3,
                                           kCnBytes / 4, kCnBytes / 6, kCnBytes / 8,
                                           kCnBytes / 12));
}  // namespace xgboost::common
