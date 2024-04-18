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

namespace xgboost::collective {
/**
 * @brief Communicator group used for double dispatching between communicators and
 *        collective implementations.
 */
class CommGroup {
  std::shared_ptr<HostComm> comm_;
  mutable std::shared_ptr<Comm> gpu_comm_;

  std::shared_ptr<Coll> backend_;
  mutable std::shared_ptr<Coll> gpu_coll_;  // lazy initialization

  CommGroup(std::shared_ptr<Comm> comm, std::shared_ptr<Coll> coll)
      : comm_{std::dynamic_pointer_cast<HostComm>(comm)}, backend_{std::move(coll)} {
    CHECK(comm_);
  }

 public:
  CommGroup();

  [[nodiscard]] auto World() const noexcept { return comm_->World(); }
  [[nodiscard]] auto Rank() const noexcept { return comm_->Rank(); }
  [[nodiscard]] bool IsDistributed() const noexcept { return comm_->IsDistributed(); }

  [[nodiscard]] Result Finalize() const {
    return Success() << [this] {
      if (gpu_comm_) {
        return gpu_comm_->Shutdown();
      }
      return Success();
    } << [&] {
      return comm_->Shutdown();
    };
  }

  [[nodiscard]] static CommGroup* Create(Json config);

  [[nodiscard]] std::shared_ptr<Coll> Backend(DeviceOrd device) const;
  /**
   * @brief Decide the context to use for communication.
   *
   * @param ctx Global context, provides the CUDA stream and ordinal.
   * @param device The device used by the data to be communicated.
   */
  [[nodiscard]] Comm const& Ctx(Context const* ctx, DeviceOrd device) const;
  [[nodiscard]] Result SignalError(Result const& res) { return comm_->SignalError(res); }

  [[nodiscard]] Result ProcessorName(std::string* out) const {
    return this->comm_->ProcessorName(out);
  }
};

std::unique_ptr<collective::CommGroup>& GlobalCommGroup();

void GlobalCommGroupInit(Json config);

void GlobalCommGroupFinalize();
}  // namespace xgboost::collective
