#pragma once

#include <map>
#include <any>
#include <xgboost/span.h>

namespace xgboost::processing {

    const std::string kLibraryPath = "LIBRARY_PATH";
    const std::string kDummyProcessor = "dummy";

    /*! \brief An processing interface to handle tasks that require external library through plugins */
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
         * \brief Preparing g & h pairs to be sent to other clients by active client
         *
         * \param pairs g&h pairs in a vector (g1, h1, g2, h2 ...)
         *
         * \return The encoded buffer to be sent
         */
        virtual common::Span<std::int8_t> ProcessGHPairs(std::vector<double>& pairs) = 0;

        /*!
         * \brief Handle encoded pairs received from other clients
         *
         * \param The encoded buffer
         *
         * \return The encoded buffer
         */
        virtual common::Span<std::int8_t> HandleEncodedPairs(common::Span<std::int8_t> buffer) = 0;

        /*!
         * \brief Initialize aggregation context by providing sample feature data and histogram bin sizes
         *
         * \param sample_feature_bin_ids The flatten array with sample, feature, bins
         * \param feature_bin_sizes Vector with each feature's bin size
         * \param available_features Available features for this node
         */
        virtual void InitAggregationContext(
                std::vector<int>& sample_feature_bin_ids,
                std::vector<int>& feature_bin_sizes,
                std::vector<int>& available_features) = 0;

        /*!
         * \brief Perform histogram add (maybe in encrypted space) for samples of multiple nodes
         *
         * \param sample_ids The flatten array with sample_ids for every node
         * \param node_sizes Samples in each node
         *
         * \return The encoded result
         */
        virtual common::Span<std::int8_t> ProcessAggregationContext(
                std::vector<int>& sample_ids,
                std::vector<int>& node_sizes) = 0;

        /*!
         * \brief Handle encoded result
         *
         * \param buffers A list of encoded buffer, one for each node
         *
         * \return A flattened vector of g&h pairs for each node
         */
        virtual std::vector<double> HandleAggregation(std::vector<common::Span<std::int8_t>> buffers) = 0;
    };

    class ProcessorLoader {

    private:
        std::map<std::string, std::string> params;
        void *handle = NULL;


    public:
        ProcessorLoader(): params{} {}

        ProcessorLoader(std::map<std::string, std::string>& params): params(params) {}

        Processor* load(std::string plugin_name);

        void unload();

    };

}  // namespace xgboost::processing
