/*!
 * Copyright 2015-2019 by Contributors
 * \file metric_registry.cc
 * \brief Registry of objective functions.
 */
#include <dmlc/registry.h>
#include <xgboost/metric.h>
#include <xgboost/generic_parameters.h>

#include "metric_common.h"

namespace xgboost {
template <typename MetricRegistry>
Metric* CreateMetricImpl(const std::string& name, GenericParameter const* tparam) {
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
    auto *e = ::dmlc::Registry<MetricRegistry>::Get()->Find(prefix.c_str());
    if (e == nullptr) {
      return nullptr;
    }
    auto p_metric = (e->body)(param);
    return p_metric;
  } else {
    std::string prefix = buf.substr(0, pos);
    auto *e = ::dmlc::Registry<MetricRegistry>::Get()->Find(prefix.c_str());
    if (e == nullptr) {
      return nullptr;
    }
    auto p_metric = (e->body)(buf.substr(pos + 1, buf.length()).c_str());
    return p_metric;
  }
}

Metric *
Metric::Create(const std::string& name, GenericParameter const* tparam) {
  auto metric = CreateMetricImpl<MetricReg>(name, tparam);
  if (metric == nullptr) {
    LOG(FATAL) << "Unknown metric function " << name;
  }

  metric->tparam_ = tparam;
  return metric;
}

Metric *
GPUMetric::CreateGPUMetric(const std::string& name, GenericParameter const* tparam) {
  auto metric = CreateMetricImpl<MetricGPUReg>(name, tparam);
  if (metric == nullptr) {
    LOG(WARNING) << "Cannot find a GPU metric builder for metric " << name
                 << ". Resorting to the CPU builder";
    return metric;
  }

  // Narrowing reference only for the compiler to allow assignment to a base class member.
  // As such, using this narrowed reference to refer to derived members will be an illegal op.
  // This is moot, as this type is stateless.
  static_cast<GPUMetric *>(metric)->tparam_ = tparam;
  return metric;
}
}  // namespace xgboost

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::MetricReg);
DMLC_REGISTRY_ENABLE(::xgboost::MetricGPUReg);
}

namespace xgboost {
namespace metric {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(elementwise_metric);
DMLC_REGISTRY_LINK_TAG(multiclass_metric);
DMLC_REGISTRY_LINK_TAG(rank_metric);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(rank_metric_gpu);
#endif
}  // namespace metric
}  // namespace xgboost
