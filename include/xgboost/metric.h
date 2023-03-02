/**
 * Copyright 2014-2023 by XGBoost Contributors
 * \file metric.h
 * \brief interface of evaluation metric function supported in xgboost.
 * \author Tianqi Chen, Kailong Chen
 */
#ifndef XGBOOST_METRIC_H_
#define XGBOOST_METRIC_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/model.h>

#include <functional>
#include <memory>  // shared_ptr
#include <string>
#include <utility>
#include <vector>

namespace xgboost {
struct Context;

/*!
 * \brief interface of evaluation metric used to evaluate model performance.
 *  This has nothing to do with training, but merely act as evaluation purpose.
 */
class Metric : public Configurable {
 protected:
  Context const* ctx_{nullptr};

 public:
  /*!
   * \brief Configure the Metric with the specified parameters.
   * \param args arguments to the objective function.
   */
  virtual void Configure(
      const std::vector<std::pair<std::string, std::string> >&) {}
  /*!
   * \brief Load configuration from JSON object
   * By default, metric has no internal configuration;
   * override this function to maintain internal configuration
   * \param in JSON object containing the configuration
   */
  void LoadConfig(Json const&) override {}
  /*!
   * \brief Save configuration to JSON object
   * By default, metric has no internal configuration;
   * override this function to maintain internal configuration
   * \param out pointer to output JSON object
   */
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
  }

  /**
   * \brief Evaluate a metric with DMatrix as input.
   *
   * \param preds Prediction
   * \param p_fmat DMatrix that contains related information like labels.
   */
  virtual double Evaluate(HostDeviceVector<bst_float> const& preds,
                          std::shared_ptr<DMatrix> p_fmat) = 0;

  /*! \return name of metric */
  virtual const char* Name() const = 0;
  /*! \brief virtual destructor */
  ~Metric() override = default;
  /*!
   * \brief create a metric according to name.
   * \param name name of the metric.
   *        name can be in form metric[@]param and the name will be matched in the
   *        registry.
   * \param ctx A global context
   * \return the created metric.
   */
  static Metric* Create(const std::string& name, Context const* ctx);
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
