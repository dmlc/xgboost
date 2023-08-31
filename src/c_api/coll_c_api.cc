/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <cstddef>  // for size_t
#include <cstdint>  // for int8_t
#include <memory>   // for unique_ptr
#include <utility>  // for pair

#include "../collective/allgather.h"         // for Allgather
#include "../collective/allreduce.h"         // for Allreduce
#include "../collective/broadcast.h"         // for Broadcast
#include "../collective/comm.h"              // for GlobalCommGroup
#include "../collective/communicator-inl.h"  // for GetRank
#include "../collective/tracker.h"           // for RabitTracker
#include "../data/array_interface.h"         // for DispatchDType, Type
#include "c_api_error.h"                     // for API_BEGIN
#include "xgboost/collective/result.h"       // for Result
#include "xgboost/context.h"                 // for Context
#include "xgboost/linalg.h"                  // for MakeTensorView
#include "xgboost/span.h"                    // for Span

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_tracker.h"  // for FederatedTracker
#endif

namespace xgboost::collective {
void Allreduce(void *send_receive_buffer, std::size_t count, std::int32_t data_type, int op) {
  Context ctx;
  DispatchDType(static_cast<ArrayInterfaceHandler::Type>(data_type), [&](auto t) {
    using T = decltype(t);
    auto data = linalg::MakeTensorView(
        &ctx, common::Span{static_cast<T *>(send_receive_buffer), count}, count);
    auto rc = Allreduce(&ctx, *GlobalCommGroup(), data, static_cast<Op>(op));
    SafeColl(rc);
  });
}

void Broadcast(void *send_receive_buffer, std::size_t size, int root) {
  Context ctx;
  auto rc = Broadcast(&ctx, *GlobalCommGroup(),
                      linalg::MakeVec(static_cast<std::int8_t *>(send_receive_buffer), size), root);
  SafeColl(rc);
}

void Allgather(void *send_receive_buffer, std::size_t size) {
  Context ctx;
  auto const &comm = GlobalCommGroup();
  auto rc = Allgather(&ctx, *comm,
                      linalg::MakeVec(reinterpret_cast<std::int8_t *>(send_receive_buffer), size));
  SafeColl(rc);
}
}  // namespace xgboost::collective

using namespace xgboost;  // NOLINT

namespace {
using TrackerHandleT =
    std::pair<std::unique_ptr<collective::Tracker>, std::shared_future<collective::Result>>;

TrackerHandleT *GetTrackerHandle(TrackerHandle handle) {
  xgboost_CHECK_C_ARG_PTR(handle);
  auto *ptr = static_cast<TrackerHandleT *>(handle);
  CHECK(ptr);
  return ptr;
}

struct CollAPIEntry {
  std::string ret_str;
};
using CollAPIThreadLocalStore = dmlc::ThreadLocalStore<CollAPIEntry>;

void WaitImpl(TrackerHandleT *ptr, std::chrono::seconds timeout) {
  constexpr std::int64_t kDft{60};
  std::chrono::seconds wait_for{timeout.count() != 0 ? std::min(kDft, timeout.count()) : kDft};

  common::Timer timer;
  timer.Start();

  auto fut = ptr->second;
  while (fut.valid()) {
    auto res = fut.wait_for(wait_for);
    CHECK(res != std::future_status::deferred);

    if (res == std::future_status::ready) {
      auto const &rc = ptr->second.get();
      collective::SafeColl(rc);
      break;
    }

    if (timer.Duration() > timeout && timeout.count() != 0) {
      collective::SafeColl(collective::Fail("Timeout waiting for the tracker."));
    }
  }
}
}  // namespace

XGB_DLL int XGTrackerCreate(char const *config, TrackerHandle *handle) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(config);

  Json jconfig = Json::Load(config);

  auto type = RequiredArg<String>(jconfig, "dmlc_communicator", __func__);
  std::unique_ptr<collective::Tracker> tptr;
  if (type == "federated") {
#if defined(XGBOOST_USE_FEDERATED)
    tptr = std::make_unique<collective::FederatedTracker>(jconfig);
#else
    LOG(FATAL) << error::NoFederated();
#endif  // defined(XGBOOST_USE_FEDERATED)
  } else if (type == "rabit") {
    tptr = std::make_unique<collective::RabitTracker>(jconfig);
  } else {
    LOG(FATAL) << "Unknown communicator:" << type;
  }

  auto ptr = new TrackerHandleT{std::move(tptr), std::future<collective::Result>{}};
  static_assert(std::is_same_v<std::remove_pointer_t<decltype(ptr)>, TrackerHandleT>);

  xgboost_CHECK_C_ARG_PTR(handle);
  *handle = ptr;
  API_END();
}

XGB_DLL int XGTrackerWorkerArgs(TrackerHandle handle, char const **args) {
  API_BEGIN();
  auto *ptr = GetTrackerHandle(handle);
  auto &local = *CollAPIThreadLocalStore::Get();
  local.ret_str = Json::Dump(ptr->first->WorkerArgs());
  xgboost_CHECK_C_ARG_PTR(args);
  *args = local.ret_str.c_str();
  API_END();
}

XGB_DLL int XGTrackerRun(TrackerHandle handle) {
  API_BEGIN();
  auto *ptr = GetTrackerHandle(handle);
  CHECK(!ptr->second.valid()) << "Tracker is already running.";
  ptr->second = ptr->first->Run();
  API_END();
}

XGB_DLL int XGTrackerWait(TrackerHandle handle, char const *config) {
  API_BEGIN();
  auto *ptr = GetTrackerHandle(handle);
  xgboost_CHECK_C_ARG_PTR(config);
  auto jconfig = Json::Load(StringView{config});
  // Internally, 0 indicates no timeout, which is the default since we don't want to
  // interrupt the model training.
  auto timeout = OptionalArg<Integer>(jconfig, "timeout", std::int64_t{0});
  WaitImpl(ptr, std::chrono::seconds{timeout});
  API_END();
}

XGB_DLL int XGTrackerFree(TrackerHandle handle) {
  API_BEGIN();
  auto *ptr = GetTrackerHandle(handle);
  WaitImpl(ptr, ptr->first->Timeout());
  delete ptr;
  API_END();
}

XGB_DLL int XGCommunicatorInit(char const *json_config) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(json_config);
  Json config{Json::Load(StringView{json_config})};
  collective::GlobalCommGroupInit(config);
  API_END();
}

XGB_DLL int XGCommunicatorFinalize() {
  API_BEGIN();
  collective::GlobalCommGroupFinalize();
  API_END();
}

XGB_DLL int XGCommunicatorGetRank(void) {
  API_BEGIN();
  return collective::GetRank();
  API_END();
}

XGB_DLL int XGCommunicatorGetWorldSize(void) { return collective::GetWorldSize(); }

XGB_DLL int XGCommunicatorIsDistributed(void) { return collective::IsDistributed(); }

XGB_DLL int XGCommunicatorPrint(char const *message) {
  API_BEGIN();
  collective::Print(message);
  API_END();
}

XGB_DLL int XGCommunicatorGetProcessorName(char const **name_str) {
  API_BEGIN();
  auto &local = *CollAPIThreadLocalStore::Get();
  local.ret_str = collective::GetProcessorName();
  xgboost_CHECK_C_ARG_PTR(name_str);
  *name_str = local.ret_str.c_str();
  API_END();
}

XGB_DLL int XGCommunicatorBroadcast(void *send_receive_buffer, size_t size, int root) {
  API_BEGIN();
  collective::Broadcast(send_receive_buffer, size, root);
  API_END();
}

XGB_DLL int XGCommunicatorAllreduce(void *send_receive_buffer, size_t count, int enum_dtype,
                                    int enum_op) {
  API_BEGIN();
  collective::Allreduce(send_receive_buffer, count, enum_dtype, enum_op);
  API_END();
}

XGB_DLL int XGCommunicatorAllgather(void *send_receive_buffer, size_t count) {
  API_BEGIN();
  collective::Allgather(send_receive_buffer, count);
  API_END();
}

XGB_DLL int XGCommunicatorSignalError() {
  API_BEGIN();
  auto msg = XGBGetLastError();
  SafeColl(xgboost::collective::GlobalCommGroup()->SignalError(xgboost::collective::Fail(msg)));
  API_END()
}
