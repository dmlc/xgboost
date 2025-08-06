/**
 * Copyright 2019-2025, XGBoost Contributors
 */

#include <cstdint>  // for int32_t, int8_t
#include <memory>   // for make_shared

#include "../common/cuda_rt_utils.h"  // for CurrentDevice, SetDevice
#include "cat_container.h"            // for CatContainer
#include "device_adapter.cuh"
#include "proxy_dmatrix.cuh"  // for DispatchAny
#include "simple_dmatrix.cuh"
#include "simple_dmatrix.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"

namespace xgboost::data {
// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
template <typename AdapterT>
SimpleDMatrix::SimpleDMatrix(AdapterT* adapter, float missing, std::int32_t nthread,
                             DataSplitMode data_split_mode) {
  CHECK(data_split_mode != DataSplitMode::kCol)
      << "Column-wise data split is currently not supported by the GPU.";
  auto device = (!adapter->Device().IsCUDA() || adapter->NumRows() == 0)
                    ? DeviceOrd::CUDA(curt::CurrentDevice())
                    : adapter->Device();
  CHECK(device.IsCUDA());
  curt::SetDevice(device.ordinal);

  Context ctx;
  ctx.Init(Args{{"nthread", std::to_string(nthread)}, {"device", device.Name()}});

  CHECK(adapter->NumRows() != kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();

  // Enforce single batch
  CHECK(!adapter->Next());

  cuda_impl::DispatchAny<true, std::add_pointer_t>(&ctx, adapter, [&](auto const& batch) {
    info_.num_nonzero_ = CopyToSparsePage(&ctx, batch, device, missing, sparse_page_.get());
  });
  info_.num_col_ = adapter->NumColumns();
  info_.num_row_ = adapter->NumRows();

  if constexpr (std::is_same_v<AdapterT, CudfAdapter>) {
    if (adapter->HasRefCategorical()) {
      info_.Cats(std::make_shared<CatContainer>(&ctx, adapter->RefCats(), true));
    } else if (adapter->HasCategorical()) {
      info_.Cats(std::make_shared<CatContainer>(&ctx, adapter->Cats(), false));
    }
  }
  this->info_.SynchronizeNumberOfColumns(&ctx, data_split_mode);

  this->fmat_ctx_ = ctx;
}

template SimpleDMatrix::SimpleDMatrix(CudfAdapter* adapter, float missing, std::int32_t nthread,
                                      DataSplitMode data_split_mode);
template SimpleDMatrix::SimpleDMatrix(CupyAdapter* adapter, float missing, std::int32_t nthread,
                                      DataSplitMode data_split_mode);
}  // namespace xgboost::data
