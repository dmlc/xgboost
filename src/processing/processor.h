/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#pragma once

#include <any>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace processing {

const char kLibraryPath[] = "LIBRARY_PATH";
const char kMockProcessor[] = "mock";
const char kLoadFunc[] = "LoadProcessor";

/*! \brief An processor interface to handle tasks that require external library through plugins */
class Processor {
 public:
    /*!
     * \brief Virtual destructor
     *
     */
    virtual ~Processor() = default;

    /*!
     * \brief Initialize the processor
     *
     * \param active If true, this is the active node
     * \param params Optional parameters
     */
    virtual void Initialize(bool active, std::map<std::string, std::string> params) = 0;

    /*!
     * \brief Shutdown the processor and free all the resources
     *
     */
    virtual void Shutdown() = 0;

    /*!
     * \brief Free buffer
     *
     * \param buffer Any buffer returned by the calls from the plugin
     */
    virtual void FreeBuffer(void* buffer) = 0;

    /*!
     * \brief Preparing g & h pairs to be sent to other clients by active client
     *
     * \param size The size of the buffer
     * \param pairs g&h pairs in a vector (g1, h1, g2, h2 ...) for every sample
     *
     * \return The encoded buffer to be sent
     */
    virtual void* ProcessGHPairs(size_t *size, const std::vector<double>& pairs) = 0;

    /*!
     * \brief Handle buffers with encoded pairs received from broadcast
     *
     * \param size Output buffer size
     * \param The encoded buffer
     * \param The encoded buffer size
     *
     * \return The encoded buffer
     */
    virtual void* HandleGHPairs(size_t *size, void *buffer, size_t buf_size) = 0;

    /*!
     * \brief Initialize aggregation context by providing global GHistIndexMatrix
     *
     * \param cuts The cut point for each feature
     * \param slots The slot assignment in a flattened matrix for each feature/row.
     * The size is num_feature*num_row
     */
    virtual void InitAggregationContext(const std::vector<uint32_t> &cuts,
                                        const std::vector<int> &slots) = 0;

    /*!
     * \brief Prepare row set for aggregation
     *
     * \param size The output buffer size
     * \param nodes Map of node and the rows belong to this node
     *
     * \return The encoded buffer to be sent via AllGatherV
     */
    virtual void *ProcessAggregation(size_t *size, std::map<int, std::vector<int>> nodes) = 0;

    /*!
     * \brief Handle all gather result
     *
     * \param buffer Buffer from all gather, only buffer from active site is needed
     * \param buf_size The size of the buffer
     *
     * \return A flattened vector of histograms for each site, each node in the form of
     *     site1_node1, site1_node2 site1_node3, site2_node1, site2_node2, site2_node3
     */
    virtual std::vector<double> HandleAggregation(void *buffer, size_t buf_size) = 0;
};

class ProcessorLoader {
 private:
    std::map<std::string, std::string> params;
    void *handle_ = NULL;

 public:
    ProcessorLoader(): params{} {}

    explicit ProcessorLoader(const std::map<std::string, std::string>& params): params(params) {}

    Processor* load(const std::string& plugin_name);

    void unload();
};

}  // namespace processing

extern processing::Processor *processor_instance;
