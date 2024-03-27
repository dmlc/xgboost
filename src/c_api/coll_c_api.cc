/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include <chrono>       // for seconds
#include <future>       // for future
#include <memory>       // for unique_ptr
#include <string>       // for string
#include <type_traits>  // for is_same_v, remove_pointer_t
#include <utility>      // for pair

#include "../collective/tracker.h"  // for RabitTracker
#include "../common/timer.h"        // for Timer
#include "c_api_error.h"            // for API_BEGIN
#include "xgboost/c_api.h"
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/json.h"               // for Json
#include "xgboost/string_view.h"        // for StringView

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_tracker.h"  // for FederatedTracker
#else
#include "../common/error_msg.h"  // for NoFederated
#endif

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
