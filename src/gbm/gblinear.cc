/*!
 * Copyright 2014 by Contributors
 * \file gblinear.cc
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is parallel coordinate descent (shotgun)
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/gbm.h>
#include <xgboost/logging.h>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gblinear);

// model parameter
struct GBLinearModelParam :public dmlc::Parameter<GBLinearModelParam> {
  // number of feature dimension
  unsigned num_feature;
  // number of output group
  int num_output_group;
  // reserved field
  int reserved[32];
  // constructor
  GBLinearModelParam() {
    std::memset(this, 0, sizeof(GBLinearModelParam));
  }
  DMLC_DECLARE_PARAMETER(GBLinearModelParam) {
    DMLC_DECLARE_FIELD(num_feature).set_lower_bound(0)
        .describe("Number of features used in classification.");
    DMLC_DECLARE_FIELD(num_output_group).set_lower_bound(1).set_default(1)
        .describe("Number of output groups in the setting.");
  }
};

// training parameter
struct GBLinearTrainParam : public dmlc::Parameter<GBLinearTrainParam> {
  /*! \brief learning_rate */
  float learning_rate;
  /*! \brief regularization weight for L2 norm */
  float reg_lambda;
  /*! \brief regularization weight for L1 norm */
  float reg_alpha;
  /*! \brief regularization weight for L2 norm in bias */
  float reg_lambda_bias;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GBLinearTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate).set_lower_bound(0.0f).set_default(1.0f)
        .describe("Learning rate of each update.");
    DMLC_DECLARE_FIELD(reg_lambda).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L2 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_alpha).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L1 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_lambda_bias).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L2 regularization on bias.");
    // alias of parameters
    DMLC_DECLARE_ALIAS(learning_rate, eta);
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
    DMLC_DECLARE_ALIAS(reg_lambda_bias, lambda_bias);
  }
  // given original weight calculate delta
  inline double CalcDelta(double sum_grad, double sum_hess, double w) const {
    if (sum_hess < 1e-5f) return 0.0f;
    double tmp = w - (sum_grad + reg_lambda * w) / (sum_hess + reg_lambda);
    if (tmp >=0) {
      return std::max(-(sum_grad + reg_lambda * w + reg_alpha) / (sum_hess + reg_lambda), -w);
    } else {
      return std::min(-(sum_grad + reg_lambda * w - reg_alpha) / (sum_hess + reg_lambda), -w);
    }
  }
  // given original weight calculate delta bias
  inline double CalcDeltaBias(double sum_grad, double sum_hess, double w) const {
    return - (sum_grad + reg_lambda_bias * w) / (sum_hess + reg_lambda_bias);
  }
};

/*!
 * \brief gradient boosted linear model
 */
class GBLinear : public GradientBooster {
 public:
  explicit GBLinear(bst_float base_margin)
      : base_margin_(base_margin) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    if (model.weight.size() == 0) {
      model.param.InitAllowUnknown(cfg);
    }
    param.InitAllowUnknown(cfg);
  }
  void Load(dmlc::Stream* fi) override {
    model.Load(fi);
  }
  void Save(dmlc::Stream* fo) const override {
    model.Save(fo);
  }
  void DoBoost(DMatrix *p_fmat,
               std::vector<bst_gpair> *in_gpair,
               ObjFunction* obj) override {
    // lazily initialize the model when not ready.
    if (model.weight.size() == 0) {
      model.InitModel();
    }

    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model.param.num_output_group;
    const RowSet &rowset = p_fmat->buffered_rowset();
    // for all the output group
    for (int gid = 0; gid < ngroup; ++gid) {
      double sum_grad = 0.0, sum_hess = 0.0;
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static) reduction(+: sum_grad, sum_hess)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.GetHess() >= 0.0f) {
          sum_grad += p.GetGrad();
          sum_hess += p.GetHess();
        }
      }
      // remove bias effect
      bst_float dw = static_cast<bst_float>(
        param.learning_rate * param.CalcDeltaBias(sum_grad, sum_hess, model.bias()[gid]));
      model.bias()[gid] += dw;
      // update grad value
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.GetHess() >= 0.0f) {
          p += bst_gpair(p.GetHess() * dw, 0);
        }
      }
    }
    dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
    while (iter->Next()) {
      // number of features
      const ColBatch &batch = iter->Value();
      const bst_omp_uint nfeat = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        const bst_uint fid = batch.col_index[i];
        ColBatch::Inst col = batch[i];
        for (int gid = 0; gid < ngroup; ++gid) {
          double sum_grad = 0.0, sum_hess = 0.0;
          for (bst_uint j = 0; j < col.length; ++j) {
            const bst_float v = col[j].fvalue;
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            sum_grad += p.GetGrad() * v;
            sum_hess += p.GetHess() * v * v;
          }
          bst_float &w = model[fid][gid];
          bst_float dw = static_cast<bst_float>(param.learning_rate *
                                                param.CalcDelta(sum_grad, sum_hess, w));
          w += dw;
          // update grad value
          for (bst_uint j = 0; j < col.length; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.GetHess() < 0.0f) continue;
            p += bst_gpair(p.GetHess() * col[j].fvalue * dw, 0);
          }
        }
      }
    }
  }

  void PredictBatch(DMatrix *p_fmat,
               std::vector<bst_float> *out_preds,
               unsigned ntree_limit) override {
    if (model.weight.size() == 0) {
      model.InitModel();
    }
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::Predict ntrees is only valid for gbtree predictor";
    std::vector<bst_float> &preds = *out_preds;
    const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
    preds.resize(0);
    // start collecting the prediction
    dmlc::DataIter<RowBatch> *iter = p_fmat->RowIterator();
    const int ngroup = model.param.num_output_group;
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      CHECK_EQ(batch.base_rowid * ngroup, preds.size());
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      preds.resize(preds.size() + batch.size * ngroup);
      // parallel over local batch
      const omp_ulong nsize = static_cast<omp_ulong>(batch.size);
      #pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float margin =  (base_margin.size() != 0) ?
              base_margin[ridx * ngroup + gid] : base_margin_;
          this->Pred(batch[i], &preds[ridx * ngroup], gid, margin);
        }
      }
    }
  }
  // add base margin
  void PredictInstance(const SparseBatch::Inst &inst,
               std::vector<bst_float> *out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    const int ngroup = model.param.num_output_group;
    for (int gid = 0; gid < ngroup; ++gid) {
      this->Pred(inst, dmlc::BeginPtr(*out_preds), gid, base_margin_);
    }
  }

  void PredictLeaf(DMatrix *p_fmat,
                   std::vector<bst_float> *out_preds,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "gblinear does not support prediction of leaf index";
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate) override {
    if (model.weight.size() == 0) {
      model.InitModel();
    }
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::PredictContribution: ntrees is only valid for gbtree predictor";
    const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
    const int ngroup = model.param.num_output_group;
    const size_t ncolumns = model.param.num_feature + 1;
    // allocate space for (#features + bias) times #groups times #rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(p_fmat->info().num_row * ncolumns * ngroup);
    // make sure contributions is zeroed, we could be reusing a previously allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // start collecting the contributions
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const RowBatch::Inst &inst = batch[i];
        size_t row_idx = static_cast<size_t>(batch.base_rowid + i);
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float *p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          // calculate linear terms' contributions
          for (bst_uint c = 0; c < inst.length; ++c) {
            if (inst[c].index >= model.param.num_feature) continue;
            p_contribs[inst[c].index] = inst[c].fvalue * model[inst[c].index][gid];
          }
          // add base margin to BIAS
          p_contribs[ncolumns - 1] = model.bias()[gid] +
            ((base_margin.size() != 0) ? base_margin[row_idx * ngroup + gid] : base_margin_);
        }
      }
    }
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    const int ngroup = model.param.num_output_group;
    const unsigned nfeature = model.param.num_feature;

    std::stringstream fo("");
    if (format == "json") {
      fo << "  { \"bias\": [" << std::endl;
      for (int gid = 0; gid < ngroup; ++gid) {
        if (gid != 0) fo << "," << std::endl;
        fo << "      " << model.bias()[gid];
      }
      fo << std::endl << "    ]," << std::endl
         << "    \"weight\": [" << std::endl;
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          if (i != 0 || gid != 0) fo << "," << std::endl;
          fo << "      " << model[i][gid];
        }
      }
      fo << std::endl << "    ]" << std::endl << "  }";
    } else {
      fo << "bias:\n";
      for (int gid = 0; gid < ngroup; ++gid) {
        fo << model.bias()[gid] << std::endl;
      }
      fo << "weight:\n";
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          fo << model[i][gid] << std::endl;
        }
      }
    }
    std::vector<std::string> v;
    v.push_back(fo.str());
    return v;
  }

 protected:
  inline void Pred(const RowBatch::Inst &inst, bst_float *preds, int gid, bst_float base) {
    bst_float psum = model.bias()[gid] + base;
    for (bst_uint i = 0; i < inst.length; ++i) {
      if (inst[i].index >= model.param.num_feature) continue;
      psum += inst[i].fvalue * model[inst[i].index][gid];
    }
    preds[gid] = psum;
  }
  // model for linear booster
  class Model {
   public:
    // parameter
    GBLinearModelParam param;
    // weight for each of feature, bias is the last one
    std::vector<bst_float> weight;
    // initialize the model parameter
    inline void InitModel(void) {
      // bias is the last weight
      weight.resize((param.num_feature + 1) * param.num_output_group);
      std::fill(weight.begin(), weight.end(), 0.0f);
    }
    // save the model to file
    inline void Save(dmlc::Stream* fo) const {
      fo->Write(&param, sizeof(param));
      fo->Write(weight);
    }
    // load model from file
    inline void Load(dmlc::Stream* fi) {
      CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param));
      fi->Read(&weight);
    }
    // model bias
    inline bst_float* bias() {
      return &weight[param.num_feature * param.num_output_group];
    }
    inline const bst_float* bias() const {
      return &weight[param.num_feature * param.num_output_group];
    }
    // get i-th weight
    inline bst_float* operator[](size_t i) {
      return &weight[i * param.num_output_group];
    }
    inline const bst_float* operator[](size_t i) const {
      return &weight[i * param.num_output_group];
    }
  };
  // biase margin score
  bst_float base_margin_;
  // model field
  Model model;
  // training parameter
  GBLinearTrainParam param;
  // Per feature: shuffle index of each feature index
  std::vector<bst_uint> feat_index;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBLinearModelParam);
DMLC_REGISTER_PARAMETER(GBLinearTrainParam);

XGBOOST_REGISTER_GBM(GBLinear, "gblinear")
.describe("Linear booster, implement generalized linear model.")
.set_body([](const std::vector<std::shared_ptr<DMatrix> >&cache, bst_float base_margin) {
    return new GBLinear(base_margin);
  });
}  // namespace gbm
}  // namespace xgboost
