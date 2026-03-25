/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>  // for max
#include <cstdint>    // for int32_t
#include <vector>     // for vector

#include "../../../src/collective/allreduce.h"
#include "../../../src/collective/comm.cuh"
#include "../../../src/collective/topo.h"
#include "../../../src/common/cuda_context.cuh"
#include "../../../src/common/cuda_rt_utils.h"
#include "../../../src/common/device_helpers.cuh"
#include "test_worker.h"

#if defined(XGBOOST_USE_NCCL)
namespace {
template <typename T>
void CopyToDevice(std::vector<T> const& src, dh::device_vector<T>* dst, cudaStream_t stream) {
  dst->resize(src.size());
  if (!src.empty()) {
    dh::safe_cuda(cudaMemcpyAsync(dst->data().get(), src.data(), src.size() * sizeof(T),
                                  cudaMemcpyHostToDevice, stream));
  }
}

template <typename T>
std::vector<T> CopyToHost(dh::device_vector<T> const& src, cudaStream_t stream) {
  std::vector<T> out(src.size());
  if (!src.empty()) {
    dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data().get(), src.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost, stream));
  }
  dh::safe_cuda(cudaStreamSynchronize(stream));
  return out;
}
}  // namespace

#include "test_worker.cuh"
#endif  // defined(XGBOOST_USE_NCCL)

namespace xgboost::collective {
namespace {

template <typename T>
void SumToMaxLen(common::Span<T const> lhs, common::Span<T const> rhs, std::vector<T>* out) {
  auto n = std::max(lhs.size(), rhs.size());
  out->assign(n, T{0});
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    (*out)[i] += lhs[i];
  }
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    (*out)[i] += rhs[i];
  }
}

template <typename T, typename Fn>
std::vector<T> ReferenceAllreduceV(std::vector<std::vector<T>> const& inputs, Fn redop) {
  CHECK(!inputs.empty());
  auto out = inputs.front();
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    std::vector<T> next;
    redop(common::Span<T const>{out.data(), out.size()},
          common::Span<T const>{inputs[i].data(), inputs[i].size()}, &next);
    out = std::move(next);
  }
  return out;
}

TEST(AllreduceVSpec, RaggedSum) {
  std::vector<std::vector<std::int32_t>> inputs{{1}, {1, 1}, {1, 1, 1}, {1, 1, 1, 1}};
  auto out = ReferenceAllreduceV(inputs, SumToMaxLen<std::int32_t>);

  std::vector<std::int32_t> expected{4, 3, 2, 1};
  ASSERT_EQ(out, expected);
}

TEST(AllreduceVSpec, EmptyRanks) {
  std::vector<std::vector<std::int32_t>> inputs{{}, {2, 3, 5}, {}, {7}};
  auto out = ReferenceAllreduceV(inputs, SumToMaxLen<std::int32_t>);

  std::vector<std::int32_t> expected{9, 3, 5};
  ASSERT_EQ(out, expected);
}

TEST(AllreduceVSpec, RepeatedReference) {
  std::vector<std::vector<std::int32_t>> inputs{{1, 2}, {3}, {4, 5, 6}};
  auto first = ReferenceAllreduceV(inputs, SumToMaxLen<std::int32_t>);
  auto second = ReferenceAllreduceV(inputs, SumToMaxLen<std::int32_t>);

  ASSERT_EQ(first, second);
}

#if defined(XGBOOST_USE_NCCL)
class MGPUAllreduceVTest : public SocketTest {};

class AllreduceVWorker : public NCCLWorkerForTest {
 public:
  using NCCLWorkerForTest::NCCLWorkerForTest;

  template <typename T>
  struct TreeScratch {
    dh::device_vector<T> payload;
    dh::device_vector<T> next;
    dh::device_vector<std::int64_t> size;

    void Reserve(std::size_t n) {
      payload.reserve(n);
      next.reserve(n);
      size.resize(1);
    }
  };

  template <typename T>
  void DeviceSumToMaxLen(dh::device_vector<T> const& lhs, dh::device_vector<T> const& rhs,
                         dh::device_vector<T>* out, cudaStream_t stream) {
    auto n = std::max(lhs.size(), rhs.size());
    out->resize(n);
    if (n == 0) {
      return;
    }

    auto lhs_ptr = lhs.data().get();
    auto rhs_ptr = rhs.data().get();
    auto out_ptr = out->data().get();
    auto lhs_size = lhs.size();
    auto rhs_size = rhs.size();

    dh::LaunchN(n, stream, [=] __device__(std::size_t i) {
      T value{0};
      if (i < lhs_size) {
        value += lhs_ptr[i];
      }
      if (i < rhs_size) {
        value += rhs_ptr[i];
      }
      out_ptr[i] = value;
    });
  }

  template <typename T>
  void TreeAllreduceV(dh::device_vector<T>* data, TreeScratch<T>* scratch) {
    auto* nccl = dynamic_cast<NCCLComm const*>(nccl_comm_.get());
    ASSERT_NE(nccl, nullptr);
    auto stub = nccl->Stub();
    auto stream = cudaStream_t{nccl->Stream()};

    auto wait_nccl = [&] {
      SafeColl(collective::BusyWait(stub, nccl->Handle(), nccl->Timeout()));
    };

    auto send_size = [&](std::int32_t peer, std::int64_t n) {
      dh::safe_cuda(cudaMemcpyAsync(scratch->size.data().get(), &n, sizeof(n),
                                    cudaMemcpyHostToDevice, stream));
      auto rc = stub->Send(scratch->size.data().get(), 1, ncclInt64, peer, nccl->Handle(), stream);
      SafeColl(rc);
      wait_nccl();
    };

    auto recv_size = [&](std::int32_t peer) {
      std::int64_t n{0};
      auto rc = stub->Recv(scratch->size.data().get(), 1, ncclInt64, peer, nccl->Handle(), stream);
      SafeColl(rc);
      wait_nccl();
      dh::safe_cuda(cudaMemcpyAsync(&n, scratch->size.data().get(), sizeof(n),
                                    cudaMemcpyDeviceToHost, stream));
      dh::safe_cuda(cudaStreamSynchronize(stream));
      return n;
    };

    auto send_vec = [&](std::int32_t peer, dh::device_vector<T> const& payload) {
      send_size(peer, static_cast<std::int64_t>(payload.size()));
      if (payload.empty()) {
        return;
      }
      auto count = payload.size() * sizeof(T);
      auto rc = stub->Send(reinterpret_cast<std::int8_t const*>(payload.data().get()), count,
                           ncclInt8, peer, nccl->Handle(), stream);
      SafeColl(rc);
      wait_nccl();
    };

    auto recv_vec = [&](std::int32_t peer, dh::device_vector<T>* payload) {
      auto n = recv_size(peer);
      payload->resize(static_cast<std::size_t>(n));
      if (n == 0) {
        return;
      }
      auto count = static_cast<std::size_t>(n) * sizeof(T);
      auto rc = stub->Recv(reinterpret_cast<std::int8_t*>(payload->data().get()), count, ncclInt8,
                           peer, nccl->Handle(), stream);
      SafeColl(rc);
      wait_nccl();
    };

    auto rank = comm_.Rank();
    auto world = comm_.World();

    bool continue_reduce = true;
    for (std::int32_t level = 0; (std::int32_t{1} << level) < world; ++level) {
      if (!continue_reduce) {
        continue;
      }
      if (rank > 0 && binomial_tree::ParentLevel(rank) == level) {
        auto parent = binomial_tree::Parent(rank);
        send_vec(parent, *data);
        continue_reduce = false;
        continue;
      }
      if (binomial_tree::HasChild(rank, level, world)) {
        auto child = binomial_tree::Child(rank, level);
        recv_vec(child, &scratch->payload);
        DeviceSumToMaxLen(*data, scratch->payload, &scratch->next, stream);
        std::swap(*data, scratch->next);
      }
    }

    std::int64_t n = 0;
    if (rank == 0) {
      n = static_cast<std::int64_t>(data->size());
      dh::safe_cuda(cudaMemcpyAsync(scratch->size.data().get(), &n, sizeof(n),
                                    cudaMemcpyHostToDevice, stream));
    }
    auto rc = stub->Broadcast(scratch->size.data().get(), scratch->size.data().get(), 1, ncclInt64,
                              0, nccl->Handle(), stream);
    SafeColl(rc);
    wait_nccl();
    dh::safe_cuda(
        cudaMemcpyAsync(&n, scratch->size.data().get(), sizeof(n), cudaMemcpyDeviceToHost, stream));
    dh::safe_cuda(cudaStreamSynchronize(stream));

    data->resize(static_cast<std::size_t>(n));
    auto count = static_cast<std::size_t>(n) * sizeof(T);
    if (count != 0) {
      rc = stub->Broadcast(reinterpret_cast<std::int8_t const*>(data->data().get()),
                           reinterpret_cast<std::int8_t*>(data->data().get()), count, ncclInt8, 0,
                           nccl->Handle(), stream);
      SafeColl(rc);
      wait_nccl();
    }
  }

  void Prototype() {
    std::vector<std::int32_t> h_data(comm_.Rank() + 1, 1);
    dh::device_vector<std::int32_t> d_data;
    TreeScratch<std::int32_t> scratch;
    scratch.Reserve(static_cast<std::size_t>(comm_.World()));
    d_data.reserve(static_cast<std::size_t>(comm_.World()));
    CopyToDevice(h_data, &d_data, cudaStream_t{ctx_.CUDACtx()->Stream()});
    this->TreeAllreduceV(&d_data, &scratch);

    auto out = CopyToHost(d_data, cudaStream_t{ctx_.CUDACtx()->Stream()});
    ASSERT_EQ(out.size(), static_cast<std::size_t>(comm_.World()));
    for (std::size_t i = 0; i < out.size(); ++i) {
      ASSERT_EQ(out[i], comm_.World() - static_cast<std::int32_t>(i)) << i;
    }
  }

  void PrototypeRepeated() {
    auto world = comm_.World();
    auto rank = comm_.Rank();
    auto stream = cudaStream_t{ctx_.CUDACtx()->Stream()};
    dh::device_vector<std::int32_t> d_data;
    TreeScratch<std::int32_t> scratch;
    auto max_n = static_cast<std::size_t>(world);
    d_data.reserve(max_n);
    scratch.Reserve(max_n);

    for (std::int32_t iter = 0; iter < 5; ++iter) {
      std::vector<std::vector<std::int32_t>> inputs(world);
      for (std::int32_t r = 0; r < world; ++r) {
        auto n = (r + iter) % (world + 1);
        inputs[r].assign(n, r + iter + 1);
      }

      auto expected = ReferenceAllreduceV(inputs, SumToMaxLen<std::int32_t>);

      CopyToDevice(inputs.at(rank), &d_data, stream);
      this->TreeAllreduceV(&d_data, &scratch);

      auto out = CopyToHost(d_data, stream);
      ASSERT_EQ(out, expected) << "iter=" << iter;
    }
  }
};

TEST_F(MGPUAllreduceVTest, SimpleTree) {
  auto n_workers = curt::AllVisibleGPUs();
  if (n_workers < 2) {
    GTEST_SKIP_("Requires at least 2 GPUs.");
  }
  n_workers = std::min<std::int32_t>(n_workers, 4);
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    AllreduceVWorker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.Prototype();
  });
}

TEST_F(MGPUAllreduceVTest, RepeatedTree) {
  auto n_workers = curt::AllVisibleGPUs();
  if (n_workers < 2) {
    GTEST_SKIP_("Requires at least 2 GPUs.");
  }
  n_workers = std::min<std::int32_t>(n_workers, 4);
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    AllreduceVWorker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.PrototypeRepeated();
  });
}
#endif  // defined(XGBOOST_USE_NCCL)

}  // namespace
}  // namespace xgboost::collective
