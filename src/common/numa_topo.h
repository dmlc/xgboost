/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once
#include <cstdint>     // for int32_t
#include <filesystem>  // for path
#include <vector>      // for vector

namespace xgboost::common {
/**
 * @brief Read a file with the `cpulist` format.
 *
 *   Linux-Only.
 *
 */
void ReadCpuList(std::filesystem::path const &path, std::vector<std::int32_t> *p_cpus);

/**
 * @brief Get the list of CPU cores grouped under the NUMA node.
 *
 *   Linux-Only.
 *
 */
void GetNumaNodeCpus(std::int32_t node_id, std::vector<std::int32_t> *p_cpus);

/**
 * @brief Find the maximum number of NUMA nodes.
 *
 *   Linux-Only.
 *
 * @return -1 if fail to get the number of nodes. Otherwise, the maximum number of nodes
 *         for allocating node mask.
 */
[[nodiscard]] std::int32_t GetNumaMaxNumNodes();

/**
 * @brief Check whether the memory policy is set to bind.
 *
 *   Linux-Only.
 *
 */
[[nodiscard]] bool GetNumaMemBind();

/**
 * @brief Get the number of configured NUMA nodes. This does not represent the highest
 *        node ID as NUMA node ID doesn't have to be contiguous.
 *
 *   Linux-Only.
 *
 * @return -1 if there's no NUMA node. Otherwise, returns the number of NUMA nodes.
 */
[[nodiscard]] std::int32_t GetNumaNumNodes();
}  // namespace xgboost::common
