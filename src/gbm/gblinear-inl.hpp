#ifndef XGBOOST_GBM_GBLINEAR_INL_HPP_
#define XGBOOST_GBM_GBLINEAR_INL_HPP_
/*!
 * \file gblinear-inl.hpp
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is parallel coordinate descent (shotgun)
 * \author Tianqi Chen
 */
#include <vector>
#include <string>
#include <algorithm>
#include "./gbm.h"
#include "../tree/updater.h"

namespace xgboost {
namespace gbm {
/*!
 * \brief gradient boosted linear model
 * \tparam FMatrix the data type updater taking
 */
template<typename FMatrix>
class GBLinear : public IGradBooster<FMatrix> {
 public:
  virtual ~GBLinear(void) {
  }
  // set model parameters
  virtual void SetParam(const char *name, const char *val) {
    if (!strncmp(name, "bst:", 4)) {
      param.SetParam(name + 4, val);
    }
    if (model.weight.size() == 0) {
      model.param.SetParam(name, val);
    }
  }
  virtual void LoadModel(utils::IStream &fi) {
    model.LoadModel(fi);
  }
  virtual void SaveModel(utils::IStream &fo) const {
    model.SaveModel(fo);
  }
  virtual void InitModel(void) {
    model.InitModel();
  }
  virtual void DoBoost(const FMatrix &fmat,
                       const BoosterInfo &info,
                       std::vector<bst_gpair> *in_gpair) {
    this->InitFeatIndex(fmat);
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model.param.num_output_group;
    const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
    // for all the output group
    for (int gid = 0; gid < ngroup; ++gid) {
      double sum_grad = 0.0, sum_hess = 0.0;
      const unsigned ndata = static_cast<unsigned>(rowset.size());
      #pragma omp parallel for schedule(static) reduction(+: sum_grad, sum_hess)
      for (unsigned i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.hess >= 0.0f) {
          sum_grad += p.grad; sum_hess += p.hess;
        }
      }
      // remove bias effect
      bst_float dw = static_cast<bst_float>(
          param.learning_rate * param.CalcDeltaBias(sum_grad, sum_hess, model.bias()[gid]));
      model.bias()[gid] += dw;
      // update grad value
      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.hess >= 0.0f) {
          p.grad += p.hess * dw;
        }
      }
    }
    // number of features
    const unsigned nfeat = static_cast<unsigned>(feat_index.size());
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < nfeat; ++i) {
      const bst_uint fid = feat_index[i];
      for (int gid = 0; gid < ngroup; ++gid) {
        double sum_grad = 0.0, sum_hess = 0.0;
        for (typename FMatrix::ColIter it = fmat.GetSortedCol(fid); it.Next();) {
          const float v = it.fvalue();
          bst_gpair &p = gpair[it.rindex() * ngroup + gid];
          if (p.hess < 0.0f) continue;
          sum_grad += p.grad * v;
          sum_hess += p.hess * v * v;
        }
        float &w = model[fid][gid];
        bst_float dw = static_cast<bst_float>(param.learning_rate * param.CalcDelta(sum_grad, sum_hess, w));
        w += dw;
        // update grad value
        for (typename FMatrix::ColIter it = fmat.GetSortedCol(fid); it.Next();) {
          bst_gpair &p = gpair[it.rindex() * ngroup + gid];
          if (p.hess < 0.0f) continue;
          p.grad += p.hess * it.fvalue() * dw;
        }
      }
    }
  }

  virtual void Predict(const FMatrix &fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<float> *out_preds) {
    std::vector<float> &preds = *out_preds;
    preds.resize(0);
    // start collecting the prediction
    utils::IIterator<SparseBatch> *iter = fmat.RowIterator();
    iter->BeforeFirst();
    const int ngroup = model.param.num_output_group;
    while (iter->Next()) {
      const SparseBatch &batch = iter->Value();
      utils::Assert(batch.base_rowid * ngroup == preds.size(),
                    "base_rowid is not set correctly");
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      preds.resize(preds.size() + batch.size * ngroup);
      // parallel over local batch
      const unsigned nsize = static_cast<unsigned>(batch.size);
      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          this->Pred(batch[i], &preds[ridx * ngroup]);
        }
      }
    }
  }
  virtual std::vector<std::string> DumpModel(const utils::FeatMap& fmap, int option) {
    utils::Error("gblinear does not support dump model");
    return std::vector<std::string>();
  }

 protected:
  inline void InitFeatIndex(const FMatrix &fmat) {
    if (feat_index.size() != 0) return;
    // initialize feature index
    unsigned ncol = static_cast<unsigned>(fmat.NumCol());
    feat_index.reserve(ncol);
    for (unsigned i = 0; i < ncol; ++i) {
      if (fmat.GetColSize(i) != 0) {
        feat_index.push_back(i);
      }
    }
    random::Shuffle(feat_index);
  }
  inline void Pred(const SparseBatch::Inst &inst, float *preds) {
    for (int gid = 0; gid < model.param.num_output_group; ++gid) {
      float psum = model.bias()[gid];
      for (bst_uint i = 0; i < inst.length; ++i) {
        psum += inst[i].fvalue * model[inst[i].findex][gid];
      }
      preds[gid] = psum;
    }
  }
  // training parameter
  struct ParamTrain {
    /*! \brief learning_rate */
    float learning_rate;
    /*! \brief regularization weight for L2 norm */
    float reg_lambda;
    /*! \brief regularization weight for L1 norm */
    float reg_alpha;
    /*! \brief regularization weight for L2 norm in bias */
    float reg_lambda_bias;
    // parameter
    ParamTrain(void) {
      reg_alpha = 0.0f;
      reg_lambda = 0.0f;
      reg_lambda_bias = 0.0f;
      learning_rate = 1.0f;
    }
    inline void SetParam(const char *name, const char *val) {
      // sync-names
      if (!strcmp("eta", name)) learning_rate = static_cast<float>(atof(val));
      if (!strcmp("lambda", name)) reg_lambda = static_cast<float>(atof(val));
      if (!strcmp( "alpha", name)) reg_alpha = static_cast<float>(atof(val));
      if (!strcmp( "lambda_bias", name)) reg_lambda_bias = static_cast<float>(atof(val));
      // real names
      if (!strcmp( "learning_rate", name)) learning_rate = static_cast<float>(atof(val));
      if (!strcmp( "reg_lambda", name)) reg_lambda = static_cast<float>(atof(val));
      if (!strcmp( "reg_alpha", name)) reg_alpha = static_cast<float>(atof(val));
      if (!strcmp( "reg_lambda_bias", name)) reg_lambda_bias = static_cast<float>(atof(val));
    }
    // given original weight calculate delta
    inline double CalcDelta(double sum_grad, double sum_hess, double w) {
      if (sum_hess < 1e-5f) return 0.0f;
      double tmp = w - (sum_grad + reg_lambda * w) / (sum_hess + reg_lambda);
      if (tmp >=0) {
        return std::max(-(sum_grad + reg_lambda * w + reg_alpha) / (sum_hess + reg_lambda), -w);
      } else {
        return std::min(-(sum_grad + reg_lambda * w - reg_alpha) / (sum_hess + reg_lambda), -w);
      }
    }
    // given original weight calculate delta bias
    inline double CalcDeltaBias(double sum_grad, double sum_hess, double w) {
      return - (sum_grad + reg_lambda_bias * w) / (sum_hess + reg_lambda_bias);
    }
  };
  // model for linear booster
  class Model {
   public:
    // model parameter
    struct Param {
      // number of feature dimension
      int num_feature;
      // number of output group
      int num_output_group;
      // reserved field
      int reserved[32];
      // constructor
      Param(void) {
        num_feature = 0;
        num_output_group = 1;
        memset(reserved, 0, sizeof(reserved));
      }
      inline void SetParam(const char *name, const char *val) {
        if (!strcmp(name, "bst:num_feature")) num_feature = atoi(val);
        if (!strcmp(name, "num_output_group")) num_output_group = atoi(val);
      }
    };
    // parameter
    Param param;
    // weight for each of feature, bias is the last one
    std::vector<float> weight;
    // initialize the model parameter
    inline void InitModel(void) {
      // bias is the last weight
      weight.resize((param.num_feature + 1) * param.num_output_group);
      std::fill(weight.begin(), weight.end(), 0.0f);
    }
    // save the model to file
    inline void SaveModel(utils::IStream &fo) const {
      fo.Write(&param, sizeof(Param));
      fo.Write(weight);
    }
    // load model from file
    inline void LoadModel(utils::IStream &fi) {
      utils::Assert(fi.Read(&param, sizeof(Param)) != 0, "Load LinearBooster");
      fi.Read(&weight);
    }
    // model bias
    inline float* bias(void) {
      return &weight[param.num_feature * param.num_output_group];
    }
    // get i-th weight
    inline float* operator[](size_t i) {
      return &weight[i * param.num_output_group];
    }
  };
  // model field
  Model model;
  // training parameter
  ParamTrain param;
  // Per feature: shuffle index of each feature index
  std::vector<bst_uint> feat_index;
};

}  // namespace gbm
}  // namespace xgboost
#endif  // XGBOOST_GBM_GBLINEAR_INL_HPP_
