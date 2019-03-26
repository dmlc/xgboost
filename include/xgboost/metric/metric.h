/*!
 * Copyright 2019 by Contributors
 * \file metric.h
 * \brief interface of evaluation metric function supported in xgboost.
 */
#ifndef XGBOOST_METRIC_H_
#define XGBOOST_METRIC_H_

#include <dmlc/registry.h>
#include <vector>
#include <string>
#include <functional>
#include <utility>

#include "../data.h"
#include "../base.h"
#include "../../../src/common/host_device_vector.h"

namespace xgboost {
/*!
 * \brief interface of evaluation metric used to evaluate model performance.
 *  This has nothing to do with training, but merely act as evaluation purpose.
 */
class Metric {
 public:
  /*!
   * \brief Configure the Metric with the specified parameters.
   * \param args arguments to the objective function.
   */
  virtual void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) {}
  /*!
   * \brief set configuration from pair iterators.
   * \param begin The beginning iterator.
   * \param end The end iterator.
   * \tparam PairIter iterator<std::pair<std::string, std::string> >
   */
  template<typename PairIter>
  inline void Configure(PairIter begin, PairIter end) {
    std::vector<std::pair<std::string, std::string> > vec(begin, end);
    this->Configure(vec);
  }
  /*!
   * \brief evaluate a specific metric
   * \param preds prediction
   * \param info information, including label etc.
   * \param distributed whether a call to Allreduce is needed to gather
   *        the average statistics across all the node,
   *        this is only supported by some metrics
   */
  virtual bst_float Eval(const HostDeviceVector<bst_float>& preds,
                         const MetaInfo& info,
                         bool distributed) = 0;
  /*! \return name of metric */
  virtual const char* Name() const = 0;
  /*! \brief virtual destructor */
  virtual ~Metric() = default;
  /*!
   * \brief create a metric according to name.
   * \param name name of the metric.
   *  name can be in form metric[@]param
   *  and the name will be matched in the registry.
   * \return the created metric.
   */
  static Metric* Create(const std::string& name);
};

/*!
 * \brief Registry entry for Metric factory functions.
 *  The additional parameter const char* param gives the value after @, can be null.
 *  For example, metric map@3, then: param == "3".
 */
struct MetricReg
    : public dmlc::FunctionRegEntryBase<MetricReg,
                                        std::function<Metric* (const char*)> > {
};

/*!
 * \brief Macro to register metric.
 *
 * \code
 * // example of registering a objective ndcg@k
 * XGBOOST_REGISTER_METRIC(RMSE, "ndcg")
 * .describe("Rooted mean square error.")
 * .set_body([](const char* param) {
 *     int at_k = atoi(param);
 *     return new NDCG(at_k);
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_METRIC(UniqueId, Name)                         \
  ::xgboost::MetricReg&  __make_ ## MetricReg ## _ ## UniqueId ## __ =  \
      ::dmlc::Registry< ::xgboost::MetricReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
#endif  // XGBOOST_METRIC_H_
