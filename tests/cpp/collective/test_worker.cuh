/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <memory>  // for shared_ptr

#include "../../../src/collective/coll.h"  // for Coll
#include "../../../src/collective/comm.h"  // for Comm
#include "test_worker.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::collective {
class NCCLWorkerForTest : public WorkerForTest {
 protected:
  std::shared_ptr<Coll> coll_;
  std::shared_ptr<xgboost::collective::Comm> nccl_comm_;
  std::shared_ptr<Coll> nccl_coll_;
  Context ctx_;

 public:
  using WorkerForTest::WorkerForTest;

  void Setup() {
    ctx_ = MakeCUDACtx(comm_.Rank());
    coll_.reset(new Coll{});
    nccl_comm_.reset(this->comm_.MakeCUDAVar(&ctx_, coll_));
    nccl_coll_.reset(coll_->MakeCUDAVar());
    ASSERT_EQ(comm_.World(), nccl_comm_->World());
    ASSERT_EQ(comm_.Rank(), nccl_comm_->Rank());
  }
};
}  // namespace xgboost::collective
