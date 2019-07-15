/*!
 * Copyright 2015-2019 by Contributors
 * \file metric_registry.cc
 * \brief Registry of objective functions.
 */
#include <dmlc/registry.h>
#include <xgboost/metric.h>
#include <xgboost/generic_parameters.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::MetricReg);
}

namespace xgboost {
Metric* Metric::Create(const std::string& name, GenericParameter const* tparam) {
  std::string buf = name;
  std::string prefix = name;
  const char* param;
  auto pos = buf.find('@');
  if (pos == std::string::npos) {
    if (!buf.empty() && buf.back() == '-') {
      // Metrics of form "metric-"
      prefix = buf.substr(0, buf.length() - 1);  // Chop off '-'
      param = "-";
    } else {
      prefix = buf;
      param = nullptr;
    }
    auto *e = ::dmlc::Registry< ::xgboost::MetricReg>::Get()->Find(prefix.c_str());
    if (e == nullptr) {
      LOG(FATAL) << "Unknown metric function " << name;
    }
    auto p_metric = (e->body)(param);
    p_metric->tparam_ = tparam;
    return p_metric;
  } else {
    std::string prefix = buf.substr(0, pos);
    auto *e = ::dmlc::Registry< ::xgboost::MetricReg>::Get()->Find(prefix.c_str());
    if (e == nullptr) {
      LOG(FATAL) << "Unknown metric function " << name;
    }
    auto p_metric = (e->body)(buf.substr(pos + 1, buf.length()).c_str());
    p_metric->tparam_ = tparam;
    return p_metric;
  }
}
}  // namespace xgboost

namespace xgboost {
namespace metric {

// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(elementwise_metric);
DMLC_REGISTRY_LINK_TAG(multiclass_metric);
DMLC_REGISTRY_LINK_TAG(rank_metric);
}  // namespace metric
}  // namespace xgboost
