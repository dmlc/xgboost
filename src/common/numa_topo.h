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

/**
 * @brief Read the `has_normal_memory` system file.
 */
void GetNumaHasNormalMemoryNodes(std::vector<std::int32_t> *p_nodes);

/**
 * @brief Read the `has_cpu` system file.
 */
void GetNumaHasCpuNodes(std::vector<std::int32_t> *p_nodes);

/**
 * @brief Get numa node on Linux. Other platforms are not supported. Returns false if the
 *        call fails.
 */
[[nodiscard]] bool GetCpuNuma(unsigned int* cpu, unsigned int* numa);

/**
 * @brief Is it physically possible to access the wrong memory?
 */
[[nodiscard]] inline bool NumaMemCanCross() {
  std::vector<std::int32_t> nodes;
  GetNumaHasCpuNodes(&nodes);
  bool result = nodes.size() > 1;
  GetNumaHasNormalMemoryNodes(&nodes);
  result &= nodes.size() > 1;
  return result;
}
}  // namespace xgboost::common
