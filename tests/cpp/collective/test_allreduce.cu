/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include <gtest/gtest.h>
#include <thrust/host_vector.h>  // for host_vector

#include <algorithm>
#include <memory>

#include "../../../src/collective/comm.cuh"        // for NCCLComm
#include "../../../src/collective/topo.h"          // for binomial_tree
#include "../../../src/common/cuda_rt_utils.h"     // for AllVisibleGPUs
#include "../../../src/common/device_helpers.cuh"  // for device_vector, ToSpan
#include "../../../src/common/type.h"              // for EraseType
#include "../../../src/common/utils.h"             // for MakeCleanup
#include "test_worker.cuh"                         // for NCCLWorkerForTest
#include "test_worker.h"                           // for WorkerForTest, TestDistributed

namespace xgboost::collective {
namespace {
class MGPUAllreduceTest : public SocketTest {};

class Worker : public NCCLWorkerForTest {
 public:
  using NCCLWorkerForTest::NCCLWorkerForTest;

  bool SkipIfOld() {
    auto nccl = dynamic_cast<NCCLComm const*>(nccl_comm_.get());
    std::int32_t major = 0, minor = 0, patch = 0;
    SafeColl(nccl->Stub()->GetVersion(&major, &minor, &patch));
    CHECK_GE(major, 2);
    bool too_old = minor < 23;
    if (too_old) {
      LOG(INFO) << "NCCL compile version:" << NCCL_VERSION_CODE << " runtime version:" << major
                << "." << minor << "." << patch;
    }
    return too_old;
  }

  void BitOr() {
    dh::device_vector<std::uint32_t> data(comm_.World(), 0);
    data[comm_.Rank()] = ~std::uint32_t{0};
    auto rc = nccl_coll_->Allreduce(*nccl_comm_, common::EraseType(dh::ToSpan(data)),
                                    ArrayInterfaceHandler::kU4, Op::kBitwiseOR);
    SafeColl(rc);
    thrust::host_vector<std::uint32_t> h_data(data.size());
    thrust::copy(data.cbegin(), data.cend(), h_data.begin());
    for (auto v : h_data) {
      ASSERT_EQ(v, ~std::uint32_t{0});
    }
  }

  void Acc() {
    dh::device_vector<double> data(314, 1.5);
    auto rc = nccl_coll_->Allreduce(*nccl_comm_, common::EraseType(dh::ToSpan(data)),
                                    ArrayInterfaceHandler::kF8, Op::kSum);
    SafeColl(rc);
    for (std::size_t i = 0; i < data.size(); ++i) {
      auto v = data[i];
      ASSERT_EQ(v, 1.5 * static_cast<double>(comm_.World())) << i;
    }
  }

  Result NoCheck() {
    dh::device_vector<double> data(314, 1.5);
    return nccl_coll_->Allreduce(*nccl_comm_, common::EraseType(dh::ToSpan(data)),
                                 ArrayInterfaceHandler::kF8, Op::kSum);
  }

  void CachedTreeScalarSplitP2P() {
    auto* nccl = dynamic_cast<NCCLComm const*>(nccl_comm_.get());
    ASSERT_NE(nccl, nullptr);

    struct GroupResource {
      std::vector<std::int32_t> active_ranks;
      ncclComm_t comm{nullptr};
      std::shared_ptr<curt::Stream> stream;
    };
    struct LevelComm {
      std::size_t group_idx{0};
      bool active{false};
      bool is_sender{false};
      bool is_receiver{false};
      int send_peer{-1};
      int recv_peer{-1};
    };

    dh::device_vector<std::int32_t> token(1);
    dh::device_vector<std::int32_t> incoming(1);

    auto subrank = [](std::vector<std::int32_t> const& active_ranks, std::int32_t rank) {
      auto it = std::find(active_ranks.cbegin(), active_ranks.cend(), rank);
      CHECK(it != active_ranks.cend());
      return static_cast<std::int32_t>(std::distance(active_ranks.cbegin(), it));
    };
    auto make_subcomm = [&](std::vector<std::int32_t> const& active_ranks) {
      ncclComm_t subcomm{nullptr};
      auto active = std::find(active_ranks.cbegin(), active_ranks.cend(), comm_.Rank()) !=
                    active_ranks.cend();
      auto color = active ? 1 : NCCL_SPLIT_NOCOLOR;
      ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
      config.blocking = 0;
      config.splitShare = 0;
      SafeColl(nccl->Stub()->CommSplit(nccl->Handle(), color, comm_.Rank(), &subcomm, &config));
      if (subcomm != nullptr) {
        SafeColl(BusyWait(nccl->Stub(), subcomm, nccl->Timeout()));
      }
      return subcomm;
    };
    auto destroy_subcomm = [&](ncclComm_t subcomm) {
      if (subcomm == nullptr) {
        return;
      }
      auto rc = Success() << [&] {
        return nccl->Stub()->CommFinalize(subcomm);
      } << [&] {
        return BusyWait(nccl->Stub(), subcomm, nccl->Timeout());
      } << [&] {
        return nccl->Stub()->CommDestroy(subcomm);
      };
      SafeColl(rc);
    };
    auto load_token = [&](curt::StreamRef s, std::int32_t value) {
      dh::safe_cuda(
          cudaMemcpyAsync(token.data().get(), &value, sizeof(value), cudaMemcpyHostToDevice, s));
      SafeColl(GetCUDAResult(s.Sync(false)));
    };
    auto read_vec = [&](curt::StreamRef s, dh::device_vector<std::int32_t> const& vec) {
      std::int32_t value{0};
      dh::safe_cuda(
          cudaMemcpyAsync(&value, vec.data().get(), sizeof(value), cudaMemcpyDeviceToHost, s));
      SafeColl(GetCUDAResult(s.Sync(false)));
      return value;
    };

    std::vector<GroupResource> groups;
    std::vector<LevelComm> reduce_levels;
    std::vector<LevelComm> bcast_levels;

    auto sync = [&](LevelComm const& lc) {
      if (!lc.active) {
        return;
      }
      auto const& group = groups[lc.group_idx];
      auto rc = Success() << [&] {
        return GetCUDAResult(group.stream->View().Sync(false));
      } << [&] {
        return BusyWait(nccl->Stub(), group.comm, nccl->Timeout());
      };
      SafeColl(rc);
    };
    auto get_group = [&](std::vector<std::int32_t> const& active_ranks) -> std::size_t {
      auto it = std::find_if(groups.begin(), groups.end(),
                             [&](auto const& g) { return g.active_ranks == active_ranks; });
      if (it != groups.end()) {
        return static_cast<std::size_t>(std::distance(groups.begin(), it));
      }

      GroupResource g;
      g.active_ranks = active_ranks;
      g.comm = make_subcomm(active_ranks);
      g.stream = std::make_shared<curt::Stream>();
      groups.push_back(std::move(g));
      return groups.size() - 1;
    };

    std::vector<bool> active(comm_.World(), true);
    for (std::int32_t level = 0; (std::int32_t{1} << level) < comm_.World(); ++level) {
      LevelComm lc;
      std::vector<std::int32_t> active_ranks;
      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        if (active[r]) {
          active_ranks.push_back(r);
        }
      }
      lc.active = std::find(active_ranks.cbegin(), active_ranks.cend(), comm_.Rank()) !=
                  active_ranks.cend();
      lc.is_sender =
          lc.active && comm_.Rank() > 0 && binomial_tree::ParentLevel(comm_.Rank()) == level;
      lc.is_receiver = lc.active && binomial_tree::HasChild(comm_.Rank(), level, comm_.World());
      lc.group_idx = get_group(active_ranks);
      if (lc.is_sender) {
        lc.send_peer = subrank(active_ranks, binomial_tree::Parent(comm_.Rank()));
      }
      if (lc.is_receiver) {
        lc.recv_peer = subrank(active_ranks, binomial_tree::Child(comm_.Rank(), level));
      }
      reduce_levels.push_back(std::move(lc));
      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        if (active[r] && r > 0 && binomial_tree::ParentLevel(r) == level) {
          active[r] = false;
        }
      }
    }

    for (std::int32_t level = binomial_tree::Depth(comm_.World()); level >= 0; --level) {
      LevelComm lc;
      std::vector<std::int32_t> active_ranks;
      for (std::int32_t r = 0; r < comm_.World(); ++r) {
        if (r == 0 || binomial_tree::ParentLevel(r) >= level) {
          active_ranks.push_back(r);
        }
      }
      lc.active = std::find(active_ranks.cbegin(), active_ranks.cend(), comm_.Rank()) !=
                  active_ranks.cend();
      lc.is_sender = binomial_tree::HasChild(comm_.Rank(), level, comm_.World());
      lc.is_receiver = comm_.Rank() != 0 && binomial_tree::ParentLevel(comm_.Rank()) == level;
      lc.group_idx = get_group(active_ranks);
      if (lc.is_sender) {
        lc.send_peer = subrank(active_ranks, binomial_tree::Child(comm_.Rank(), level));
      }
      if (lc.is_receiver) {
        lc.recv_peer = subrank(active_ranks, binomial_tree::Parent(comm_.Rank()));
      }
      bcast_levels.push_back(std::move(lc));
    }

    auto cleanup = common::MakeCleanup([&] {
      for (auto& g : groups) {
        destroy_subcomm(g.comm);
      }
    });

    for (std::int32_t iter = 0; iter < 8; ++iter) {
      load_token(nccl->Stream(), comm_.Rank() + 1);

      for (auto const& lc : reduce_levels) {
        if (!lc.active) {
          continue;
        }
        auto const& group = groups[lc.group_idx];
        auto rc = Success() << [&] {
          return nccl->Stub()->GroupStart();
        } << [&] {
          if (lc.is_sender) {
            auto rc = nccl->Stub()->Send(token.data().get(), sizeof(std::int32_t), ncclInt8,
                                         lc.send_peer, group.comm, group.stream->View());
            if (!rc.OK()) {
              return rc;
            }
          }
          if (lc.is_receiver) {
            return nccl->Stub()->Recv(incoming.data().get(), sizeof(std::int32_t), ncclInt8,
                                      lc.recv_peer, group.comm, group.stream->View());
          }
          return Success();
        } << [&] {
          return nccl->Stub()->GroupEnd();
        };
        SafeColl(rc);
        sync(lc);

        if (lc.is_receiver) {
          auto acc =
              read_vec(group.stream->View(), token) + read_vec(group.stream->View(), incoming);
          load_token(group.stream->View(), acc);
        }
      }

      for (auto const& lc : bcast_levels) {
        if (!lc.active) {
          continue;
        }
        auto const& group = groups[lc.group_idx];
        auto rc = Success() << [&] {
          return nccl->Stub()->GroupStart();
        } << [&] {
          if (lc.is_sender) {
            auto rc = nccl->Stub()->Send(token.data().get(), sizeof(std::int32_t), ncclInt8,
                                         lc.send_peer, group.comm, group.stream->View());
            if (!rc.OK()) {
              return rc;
            }
          }
          if (lc.is_receiver) {
            return nccl->Stub()->Recv(incoming.data().get(), sizeof(std::int32_t), ncclInt8,
                                      lc.recv_peer, group.comm, group.stream->View());
          }
          return Success();
        } << [&] {
          return nccl->Stub()->GroupEnd();
        };
        SafeColl(rc);
        sync(lc);

        if (lc.is_receiver) {
          auto value = read_vec(group.stream->View(), incoming);
          load_token(group.stream->View(), value);
        }
      }

      ASSERT_EQ(read_vec(nccl->Stream(), token), comm_.World() * (comm_.World() + 1) / 2)
          << "iter=" << iter;
    }
  }

  ~Worker() noexcept(false) override = default;
};
}  // namespace

TEST_F(MGPUAllreduceTest, BitOr) {
  auto n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.BitOr();
  });
}

TEST_F(MGPUAllreduceTest, Sum) {
  auto n_workers = curt::AllVisibleGPUs();
  TestDistributed(n_workers, [=](std::string host, std::int32_t port, std::chrono::seconds timeout,
                                 std::int32_t r) {
    Worker w{host, port, timeout, n_workers, r};
    w.Setup();
    w.Acc();
  });
}

TEST_F(MGPUAllreduceTest, CachedTreeScalarSplitP2P) {
  auto n_workers = curt::AllVisibleGPUs();
  using std::chrono_literals::operator""s;
  TestDistributed(
      n_workers,
      [=](std::string host, std::int32_t port, std::chrono::seconds timeout, std::int32_t r) {
        Worker w{host, port, timeout, n_workers, r};
        w.Setup();
        w.CachedTreeScalarSplitP2P();
      },
      8s);
}

TEST_F(MGPUAllreduceTest, Timeout) {
  auto n_workers = curt::AllVisibleGPUs();
  if (n_workers <= 1) {
    GTEST_SKIP_("Requires more than one GPU to run.");
  }
  using std::chrono_literals::operator""s;

  TestDistributed(
      n_workers,
      [=](std::string host, std::int32_t port, std::chrono::seconds, std::int32_t r) {
        auto w = std::make_unique<Worker>(host, port, 1s, n_workers, r);
        w->Setup();
        if (w->SkipIfOld()) {
          GTEST_SKIP_("nccl is too old.");
          return;
        }
        if (r == 0) {
          std::this_thread::sleep_for(2s);
        }
        auto rc = w->NoCheck();
        if (r == 1) {
          auto rep = rc.Report();
          ASSERT_NE(rep.find("NCCL timeout:"), std::string::npos) << rep;
        }
      },
      8s);

  TestDistributed(
      n_workers,
      [=](std::string host, std::int32_t port, std::chrono::seconds, std::int32_t r) {
        auto w = std::make_unique<Worker>(host, port, 1s, n_workers, r);
        w->Setup();
        if (w->SkipIfOld()) {
          GTEST_SKIP_("nccl is too old.");
          return;
        }
        if (r == 0) {
          auto rc = w->NoCheck();
          ASSERT_NE(rc.Report().find("NCCL timeout:"), std::string::npos) << rc.Report();
        }
      },
      8s);
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
