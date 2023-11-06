/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <memory>   // for shared_ptr, unique_ptr
#include <string>   // for string
#include <utility>  // for move

#include "coll.h"                       // for Comm
#include "comm.h"                       // for Coll
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for GetHostName

namespace xgboost::collective {
/**
 * @brief Communicator group used for double dispatching between communicators and
 *        collective implementations.
 */
class CommGroup {
  std::shared_ptr<Comm> comm_;
  mutable std::shared_ptr<Comm> gpu_comm_;

  std::shared_ptr<Coll> backend_;
  mutable std::shared_ptr<Coll> gpu_coll_;  // lazy initialization

  CommGroup(std::shared_ptr<Comm> comm, std::shared_ptr<Coll> coll)
      : comm_{std::move(comm)}, backend_{std::move(coll)} {}

 public:
  CommGroup();

  [[nodiscard]] auto World() const { return comm_->World(); }
  [[nodiscard]] auto Rank() const { return comm_->Rank(); }
  [[nodiscard]] bool IsDistributed() const { return comm_->IsDistributed(); }

  [[nodiscard]] static CommGroup* Create(Json config);

  [[nodiscard]] std::shared_ptr<Coll> Backend(DeviceOrd device) const;
  [[nodiscard]] Comm const& Ctx(Context const* ctx, DeviceOrd device) const;
  [[nodiscard]] Result SignalError(Result const& res) { return comm_->SignalError(res); }

  [[nodiscard]] Result ProcessorName(std::string* out) const {
    auto rc = GetHostName(out);
    return rc;
  }
};

std::unique_ptr<collective::CommGroup>& GlobalCommGroup();

void GlobalCommGroupInit(Json config);

void GlobalCommGroupFinalize();
}  // namespace xgboost::collective
