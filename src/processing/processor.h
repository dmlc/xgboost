/**
 * Copyright 2014-2024 by XGBoost Contributors
 */
#pragma once

#include <xgboost/span.h>
#include <map>
#include <any>
#include <string>
#include <vector>
#include "../data/gradient_index.h"

namespace xgboost::processing {

const char kLibraryPath[] = "LIBRARY_PATH";
const char kDummyProcessor[] = "dummy";
const char kLoadFunc[] = "LoadProcessor";

//  Data type definition
const int kDataTypeGHPairs = 1;
const int kDataTypeHisto = 2;

/*! \brief An processor interface to handle tasks that require external library through plugins */
class Processor {
 public:
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
    virtual void FreeBuffer(common::Span<std::int8_t> buffer) = 0;

    /*!
     * \brief Preparing g & h pairs to be sent to other clients by active client
     *
     * \param pairs g&h pairs in a vector (g1, h1, g2, h2 ...) for every sample
     *
     * \return The encoded buffer to be sent
     */
    virtual common::Span<std::int8_t> ProcessGHPairs(std::vector<double>& pairs) = 0;

    /*!
     * \brief Handle buffers with encoded pairs received from broadcast
     *
     * \param The encoded buffer
     *
     * \return The encoded buffer
     */
    virtual common::Span<std::int8_t> HandleGHPairs(common::Span<std::int8_t> buffer) = 0;

    /*!
     * \brief Initialize aggregation context by providing global GHistIndexMatrix
     *
     * \param gidx The matrix for every sample with its feature and slot assignment
     */
    virtual void InitAggregationContext(GHistIndexMatrix const &gidx) = 0;

    /*!
     * \brief Prepare row set for aggregation
     *
     * \param row_set Information for node IDs and its sample IDs
     *
     * \return The encoded buffer to be sent via AllGather
     */
    virtual common::Span<std::int8_t> ProcessAggregation(std::vector<bst_node_t> const &nodes_to_build,
                                                         common::RowSetCollection const &row_set) = 0;

    /*!
     * \brief Handle all gather result
     *
     * \param buffers Buffer from all gather, only buffer from active site is needed
     *
     * \return A flattened vector of histograms for each site, each node in the form of
     *     site1_node1, site1_node2 site1_node3, site2_node1, site2_node2, site2_node3
     */
    virtual std::vector<double> HandleAggregation(common::Span<std::int8_t> buffer) = 0;
};

class ProcessorLoader {
 private:
    std::map<std::string, std::string> params;
    void *handle = NULL;


 public:
    ProcessorLoader(): params{} {}

    ProcessorLoader(std::map<std::string, std::string>& params): params(params) {}

    Processor* load(const std::string& plugin_name);

    void unload();
};

extern Processor *processor_instance;

}  // namespace xgboost::processing
