/**
 * Copyright 2023, XGBoost Contributors
 */
#include "coll.h"

#include <algorithm>   // for min, max
#include <cstddef>     // for size_t
#include <cstdint>     // for int8_t, int64_t
#include <functional>  // for bit_and, bit_or, bit_xor, plus

#include "allgather.h"  // for RingAllgatherV, RingAllgather
#include "allreduce.h"  // for Allreduce
#include "broadcast.h"  // for Broadcast
#include "comm.h"       // for Comm

namespace xgboost::collective {
[[nodiscard]] Result Coll::Allreduce(Comm const& comm, common::Span<std::int8_t> data,
                                     ArrayInterfaceHandler::Type type, Op op) {
  namespace coll = ::xgboost::collective;

  auto redop_fn = [](auto lhs, auto out, auto elem_op) {
    auto p_lhs = lhs.data();
    auto p_out = out.data();
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      p_out[i] = elem_op(p_lhs[i], p_out[i]);
    }
  };

  auto fn = [&](auto elem_op, auto t) {
    using T = decltype(t);

    auto redop = [redop_fn, elem_op](auto lhs, auto rhs) { redop_fn(lhs, rhs, elem_op); };

    auto erased_fn = [redop](common::Span<std::int8_t const> lhs, common::Span<std::int8_t> out) {
      CHECK_EQ(lhs.size(), out.size()) << "Invalid input for reduction.";
      auto lhs_t = common::RestoreType<T const>(lhs);
      auto rhs_t = common::RestoreType<T>(out);
      redop(lhs_t, rhs_t);
    };

    return cpu_impl::RingAllreduce(comm, data, erased_fn, type);
  };

  auto rc = DispatchDType(type, [&] (auto t) {
    using T = decltype(t);
    switch (op) {
      case Op::kMax: {
        return fn([](auto l, auto r) { return std::max(l, r); }, t);
      }
      case Op::kMin: {
        return fn([](auto l, auto r) { return std::min(l, r); }, t);
      }
      case Op::kSum: {
        return fn(std::plus<>{}, t);
      }
      case Op::kBitwiseAND: {
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<__half, T>) {
          return Fail("Invalid type.");
        } else {
          return fn(std::bit_and<>{}, t);
        }
      }
      case Op::kBitwiseOR: {
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<__half, T>) {
          return Fail("Invalid type.");
        } else {
          return fn(std::bit_and<>{}, t);
        }
      }
      case Op::kBitwiseXOR: {
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<__half, T>) {
          return Fail("Invalid type.");
        } else {
          return fn(std::bit_xor<>{}, t);
        }
      }
    }
    return Fail("Invalid op.");
  });

  return std::move(rc) << [&] { return comm.Block(); };
}

[[nodiscard]] Result Coll::Broadcast(Comm const& comm, common::Span<std::int8_t> data,
                                     std::int32_t root) {
  return cpu_impl::Broadcast(comm, data, root);
}

[[nodiscard]] Result Coll::Allgather(Comm const& comm, common::Span<std::int8_t> data,
                                     std::int64_t size) {
  return RingAllgather(comm, data, size);
}

[[nodiscard]] Result Coll::AllgatherV(Comm const& comm, common::Span<std::int8_t const> data,
                                      common::Span<std::int64_t const> sizes,
                                      common::Span<std::int64_t> recv_segments,
                                      common::Span<std::int8_t> recv, AllgatherVAlgo algo) {
  // get worker offset
  detail::AllgatherVOffset(sizes, recv_segments);

  // copy data
  auto current = recv.subspan(recv_segments[comm.Rank()], data.size_bytes());
  if (current.data() != data.data()) {
    std::copy_n(data.data(), data.size(), current.data());
  }

  switch (algo) {
    case AllgatherVAlgo::kRing:
      return detail::RingAllgatherV(comm, sizes, recv_segments, recv);
    case AllgatherVAlgo::kBcast:
      return cpu_impl::BroadcastAllgatherV(comm, sizes, recv);
    default: {
      return Fail("Unknown algorithm for allgather-v");
    }
  }
}

#if !defined(XGBOOST_USE_NCCL)
Coll* Coll::MakeCUDAVar() {
  LOG(FATAL) << "NCCL is required for device communication.";
  return nullptr;
}
#endif

}  // namespace xgboost::collective
