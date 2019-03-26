/*
  Copyright (c) 2019 by Contributors
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#ifndef XGBOOST_METRIC_RANKING_METRIC_H_
#define XGBOOST_METRIC_RANKING_METRIC_H_

#include <xgboost/metric/metric.h>
#include <xgboost/metric/metric_param.h>

#include <limits>
#include <string>

namespace xgboost {
namespace metric {
/*! \brief Evaluate rank list */
struct EvalRankList : public Metric {
 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                const MetaInfo &info,
                bool distributed) override {
    CHECK_EQ(preds.Size(), info.labels_.Size())
      << "label size predict size not match";
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK_NE(gptr.size(), 0U) << "must specify group when constructing rank file";
    CHECK_EQ(gptr.back(), preds.Size())
     << "EvalRanklist: group structure must match number of prediction";
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;
    const auto &labels = info.labels_.HostVector();

    const std::vector<bst_float> &h_preds = preds.HostVector();
    #pragma omp parallel reduction(+:sum_metric)
    {
     // each thread takes a local rec
     std::vector<std::pair<bst_float, unsigned> > rec;
     #pragma omp for schedule(static)
     for (bst_omp_uint k = 0; k < ngroup; ++k) {
       rec.clear();
       for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
         rec.emplace_back(h_preds[j], static_cast<int>(labels[j]));
       }
       sum_metric += this->EvalMetric(rec);
     }
    }
    if (distributed) {
      bst_float dat[2];
      dat[0] = static_cast<bst_float>(sum_metric);
      dat[1] = static_cast<bst_float>(ngroup);
      // approximately estimate the metric using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<bst_float>(sum_metric) / ngroup;
    }
  }

  const char *Name() const override {
    return name_.c_str();
  }

 protected:
  explicit EvalRankList(const char *name, const char *param) {
    using namespace std;  // NOLINT(*)
    minus_ = false;
    if (param != nullptr) {
      std::ostringstream os;
      os << name << '@' << param;
      name_ = os.str();
      if (sscanf(param, "%u[-]?", &topn_) != 1) {
        topn_ = std::numeric_limits<unsigned>::max();
      }
      if (param[strlen(param) - 1] == '-') {
        minus_ = true;
      }
    } else {
      name_ = name;
      topn_ = std::numeric_limits<unsigned>::max();
    }
  }

  /*! \return evaluation metric, given the pair_sort record, (pred,label) */
  virtual bst_float
  EvalMetric(std::vector<std::pair<bst_float, unsigned> > &pair_sort) const = 0; // NOLINT(*)

 protected:
  unsigned topn_;
  std::string name_;
  bool minus_;
};

}  // namespace metric
}  // namespace xgboost

#endif  // XGBOOST_METRIC_RANKING_METRIC_H_
