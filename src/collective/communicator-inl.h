/**
 * Copyright 2022-2024, XGBoost contributors
 */
#pragma once
#include <string>
#include <vector>
#include "xgboost/json.h"  // for Json

namespace xgboost::collective {
/**
 * \brief Initialize the collective communicator.
 *
 *  Currently the communicator API is experimental, function signatures may change in the future
 *  without notice.
 *
 *  Call this once before using anything.
 *
 *  The additional configuration is not required. Usually the communicator will detect settings
 *  from environment variables.
 */
void Init(Json const& config);

/**
 * @brief Finalize the collective communicator.
 *
 * Call this function after you finished all jobs.
 */
void Finalize();

/**
 * @brief Get rank of current process.
 *
 * @return Rank of the worker.
 */
[[nodiscard]] std::int32_t GetRank();

/**
 * @brief Get total number of processes.
 *
 * @return Total world size.
 */
[[nodiscard]] std::int32_t GetWorldSize();

/**
 * @brief Get if the communicator is distributed.
 *
 * @return True if the communicator is distributed.
 */
[[nodiscard]] bool IsDistributed();

/*!
 * \brief Get if the communicator is federated.
 *
 * \return True if the communicator is federated.
 */
[[nodiscard]] bool IsFederated();

/**
 * @brief Print the message to the communicator.
 *
 * This function can be used to communicate the information of the progress to the user who monitors
 * the communicator.
 *
 * @param message The message to be printed.
 */
void Print(std::string const& message);
/**
 * @brief Get the name of the processor.
 *
 * @return Name of the processor.
 */
std::string GetProcessorName();
}  // namespace xgboost::collective
