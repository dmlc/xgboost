/**
 * Copyright 2024, XGBoost Contributors
 */
#include <memory>   // for shared_ptr
#include <variant>  // for visit

#include "extmem_quantile_dmatrix.h"

namespace xgboost::data {
void ExtMemQuantileDMatrix::InitFromCUDA(
    Context const *, std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>>,
    DMatrixHandle, BatchParam const &, float, std::shared_ptr<DMatrix>) {
  LOG(FATAL) << "Not implemented.";
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackBatches(Context const *,
                                                               const BatchParam &) {
  LOG(FATAL) << "Not implemented.";
  auto batch_set =
      std::visit([this](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}
}  // namespace xgboost::data
