/**
 * Copyright 2023 by XGBoost contributors
 *
 * Higher level functions built on top the Communicator API, taking care of behavioral differences
 * between row-split vs column-split distributed training, and horizontal vs vertical federated
 * learning.
 */
#pragma once
#include <xgboost/data.h>

#include <string>
#include <utility>
#include <vector>

#include "communicator-inl.h"

namespace xgboost {
namespace collective {

/**
 * @brief Apply the given function where the labels are.
 *
 * Normally all the workers have access to the labels, so the function is just applied locally. In
 * vertical federated learning, we assume labels are only available on worker 0, so the function is
 * applied there, with the results broadcast to other workers.
 *
 * @tparam Function The function used to calculate the results.
 * @tparam Args Arguments to the function.
 * @param info MetaInfo about the DMatrix.
 * @param buffer The buffer storing the results.
 * @param size The size of the buffer.
 * @param function The function used to calculate the results.
 * @param args Arguments to the function.
 */
template <typename Function, typename... Args>
void ApplyWithLabels(MetaInfo const& info, void* buffer, size_t size, Function&& function,
                     Args&&... args) {
  if (info.IsVerticalFederated()) {
    // We assume labels are only available on worker 0, so the calculation is done there and result
    // broadcast to other workers.
    std::vector<char> message(1024);
    if (collective::GetRank() == 0) {
      try {
        std::forward<Function>(function)(std::forward<Args>(args)...);
      } catch (dmlc::Error& e) {
        strncpy(&message[0], e.what(), message.size());
        message.back() = '\0';
      }
    }
    collective::Broadcast(&message[0], message.size(), 0);
    if (strlen(&message[0]) == 0) {
      collective::Broadcast(buffer, size, 0);
    } else {
      LOG(FATAL) << &message[0];
    }
  } else {
    std::forward<Function>(function)(std::forward<Args>(args)...);
  }
}

}  // namespace collective
}  // namespace xgboost
