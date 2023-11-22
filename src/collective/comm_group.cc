/**
 * Copyright 2023, XGBoost Contributors
 */
#include "comm_group.h"

#include <algorithm>  // for transform
#include <chrono>     // for seconds
#include <cstdint>    // for int32_t
#include <memory>     // for shared_ptr, unique_ptr
#include <string>     // for string
#include <vector>     // for vector

#include "../common/json_utils.h"       // for OptionalArg
#include "coll.h"                       // for Coll
#include "comm.h"                       // for Comm
#include "tracker.h"                    // for GetHostAddress
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/context.h"            // for DeviceOrd
#include "xgboost/json.h"               // for Json

#if defined(XGBOOST_USE_FEDERATED)
#include "../../plugin/federated/federated_coll.h"
#include "../../plugin/federated/federated_comm.h"
#endif

namespace xgboost::collective {
[[nodiscard]] std::shared_ptr<Coll> CommGroup::Backend(DeviceOrd device) const {
  if (device.IsCUDA()) {
    if (!gpu_coll_) {
      gpu_coll_.reset(backend_->MakeCUDAVar());
    }
    return gpu_coll_;
  }
  return backend_;
}

[[nodiscard]] Comm const& CommGroup::Ctx(Context const* ctx, DeviceOrd device) const {
  if (device.IsCUDA()) {
    CHECK(ctx->IsCUDA());
    if (!gpu_comm_ || gpu_comm_->World() != comm_->World()) {
      gpu_comm_.reset(comm_->MakeCUDAVar(ctx, backend_));
    }
    return *gpu_comm_;
  }
  return *comm_;
}

CommGroup::CommGroup()
    : comm_{std::shared_ptr<RabitComm>(new RabitComm{})},  // NOLINT
      backend_{std::shared_ptr<Coll>(new Coll{})} {}       // NOLINT

[[nodiscard]] CommGroup* CommGroup::Create(Json config) {
  if (IsA<Null>(config)) {
    return new CommGroup;
  }

  std::string type = OptionalArg<String>(config, "dmlc_communicator", std::string{"rabit"});
  // Try both lower and upper case for compatibility
  auto get_param = [&](std::string name, auto dft, auto t) {
    std::string upper;
    std::transform(name.cbegin(), name.cend(), std::back_inserter(upper),
                   [](char c) { return std::toupper(c); });
    std::transform(name.cbegin(), name.cend(), name.begin(),
                   [](char c) { return std::tolower(c); });

    auto const& obj = get<Object const>(config);
    auto it = obj.find(upper);
    if (it != obj.cend()) {
      return OptionalArg<decltype(t)>(config, upper, dft);
    } else {
      return OptionalArg<decltype(t)>(config, name, dft);
    }
  };
  // Common args
  auto retry = get_param("dmlc_retry", static_cast<Integer::Int>(DefaultRetry()), Integer{});
  auto timeout =
      get_param("dmlc_timeout_sec", static_cast<Integer::Int>(DefaultTimeoutSec()), Integer{});
  auto task_id = get_param("dmlc_task_id", std::string{}, String{});

  if (type == "rabit") {
    auto host = get_param("dmlc_tracker_uri", std::string{}, String{});
    auto port = get_param("dmlc_tracker_port", static_cast<std::int64_t>(0), Integer{});
    auto nccl = get_param("dmlc_nccl_path", std::string{DefaultNcclName()}, String{});
    auto ptr =
        new CommGroup{std::shared_ptr<RabitComm>{new RabitComm{  // NOLINT
                          host, static_cast<std::int32_t>(port), std::chrono::seconds{timeout},
                          static_cast<std::int32_t>(retry), task_id, nccl}},
                      std::shared_ptr<Coll>(new Coll{})};  // NOLINT
    return ptr;
  } else if (type == "federated") {
#if defined(XGBOOST_USE_FEDERATED)
    auto ptr = new CommGroup{
        std::make_shared<FederatedComm>(retry, std::chrono::seconds{timeout}, task_id, config),
        std::make_shared<FederatedColl>()};
    return ptr;
#endif  // defined(XGBOOST_USE_FEDERATED)
  } else {
    LOG(FATAL) << "Invalid communicator type";
  }

  return nullptr;
}

std::unique_ptr<collective::CommGroup>& GlobalCommGroup() {
  static thread_local std::unique_ptr<collective::CommGroup> sptr;
  if (!sptr) {
    Json config{Null{}};
    sptr.reset(CommGroup::Create(config));
  }
  return sptr;
}

void GlobalCommGroupInit(Json config) {
  auto& sptr = GlobalCommGroup();
  sptr.reset(CommGroup::Create(std::move(config)));
}

void GlobalCommGroupFinalize() {
  auto& sptr = GlobalCommGroup();
  sptr.reset();
}
}  // namespace xgboost::collective
