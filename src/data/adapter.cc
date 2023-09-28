/**
 *  Copyright 2019-2023, XGBoost Contributors
 */
#include "adapter.h"

#include "../c_api/c_api_error.h"  // for API_BEGIN, API_END
#include "xgboost/c_api.h"

namespace xgboost::data {
template <typename DataIterHandle, typename XGBCallbackDataIterNext, typename XGBoostBatchCSR>
bool IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>::Next() {
  if ((*next_callback_)(
          data_handle_,
          [](void *handle, XGBoostBatchCSR batch) -> int {
            API_BEGIN();
            static_cast<IteratorAdapter *>(handle)->SetData(batch);
            API_END();
          },
          this) != 0) {
    at_first_ = false;
    return true;
  } else {
    return false;
  }
}

template class IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>;
}  // namespace xgboost::data
