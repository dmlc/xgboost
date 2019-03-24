/*!
 * Copyright 2019 by Contributors
 * \file multiclass_metric.cc
 * \brief evaluation metrics for multiclass classification.
 */

#ifndef XGBOOST_MULTICLASS_METRIC_H
#define XGBOOST_MULTICLASS_METRIC_H

#include <xgboost/metric.h>
namespace xgboost {
  namespace metric {
/*!
 * \brief base class of multi-class evaluation
 * \tparam Derived the name of subclass
 */
    template<typename Derived>
    struct EvalMClassBase : public Metric {
      bst_float Eval(const HostDeviceVector<bst_float> &preds,
                     const MetaInfo &info,
                     bool distributed) override {
        CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
        CHECK(preds.Size() % info.labels_.Size() == 0)
          << "label and prediction size not match";
        const size_t nclass = preds.Size() / info.labels_.Size();
        CHECK_GE(nclass, 1U)
          << "mlogloss and merror are only used for multi-class classification,"
          << " use logloss for binary classification";
        const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());
        double sum = 0.0, wsum = 0.0;
        int label_error = 0;

        const auto &labels = info.labels_.HostVector();
        const auto &weights = info.weights_.HostVector();
        const std::vector<bst_float> &h_preds = preds.HostVector();

#pragma omp parallel for reduction(+: sum, wsum) schedule(static)
        for (bst_omp_uint i = 0; i < ndata; ++i) {
          const bst_float wt = weights.size() > 0 ? weights[i] : 1.0f;
          auto label = static_cast<int>(labels[i]);
          if (label >= 0 && label < static_cast<int>(nclass)) {
            sum += Derived::EvalRow(label,
                                    h_preds.data() + i * nclass,
                                    nclass) * wt;
            wsum += wt;
          } else {
            label_error = label;
          }
        }
        CHECK(label_error >= 0 && label_error < static_cast<int>(nclass))
          << "MultiClassEvaluation: label must be in [0, num_class),"
          << " num_class=" << nclass << " but found " << label_error << " in label";

        double dat[2];
        dat[0] = sum, dat[1] = wsum;
        if (distributed) {
          rabit::Allreduce<rabit::op::Sum>(dat, 2);
        }
        return Derived::GetFinal(dat[0], dat[1]);
      }

      /*!
       * \brief to be implemented by subclass,
       *   get evaluation result from one row
       * \param label label of current instance
       * \param pred prediction value of current instance
       * \param nclass number of class in the prediction
       */
      inline static bst_float EvalRow(int label,
                                      const bst_float *pred,
                                      size_t nclass);

      /*!
       * \brief to be overridden by subclass, final transformation
       * \param esum the sum statistics returned by EvalRow
       * \param wsum sum of weight
       */
      inline static bst_float GetFinal(bst_float esum, bst_float wsum) {
        return esum / wsum;
      }

    private:
      // used to store error message
      const char *error_msg_;
    };
  };
}

#endif //XGBOOST_MULTICLASS_METRIC_H
