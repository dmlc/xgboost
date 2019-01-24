/*!
 * Copyright 2018 XGBoost contributors
 */
#include "common.h"

namespace xgboost {

int AllVisibleImpl::AllVisible() {
  int n_visgpus = 0;
  try {
    // When compiled with CUDA but running on CPU only device,
    // cudaGetDeviceCount will fail.
    //dh::safe_cuda(cudaGetDeviceCount(&n_visgpus));
    cudaGetDeviceCount(&n_visgpus);
  } catch(const dmlc::Error &except) {
    return 0;
  } catch(const std::exception& e) {
    return 0;
  } catch(const std::string& e) {
    return 0;
  } catch(...) {
    return 0;
  }
  return n_visgpus;
}

}  // namespace xgboost
