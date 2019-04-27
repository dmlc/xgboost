#include <xgboost/data.h>
#include <xgboost/gbm.h>
#include <vector>

#include "gibbs_updater.h"

namespace xgboost {
namespace bart {

struct BartTrainParam : public dmlc::Parameter<BartTrainParam> {
  bst_uint num_trees;
  DMLC_DECLARE_PARAMETER(BartTrainParam) {
    DMLC_DECLARE_FIELD(num_trees)
        .set_default(100);
  }
};

DMLC_REGISTER_PARAMETER(BartTrainParam);

class Bart : public GradientBooster {
  GibbsUpdater _updater;
  std::vector<std::shared_ptr<RegTree>> _p_trees;
  BartTrainParam _param;

 public:
  explicit Bart(const std::vector<std::shared_ptr<DMatrix> > &cache,
                bst_float base_margin) {}
  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    _param.InitAllowUnknown(cfg);
    _p_trees.resize(_param.num_trees);
    for (size_t i = 0; i < _param.num_trees; ++i)  {
      _p_trees[i].reset(new RegTree);
    }
    _updater.configure(cfg);
  }

  void Load(dmlc::Stream* fi) override {}
  void Save(dmlc::Stream* fo) const override {}
  void DoBoost(DMatrix *p_fmat,
               HostDeviceVector<GradientPair> *in_gpair,
               ObjFunction* obj) override {
    _updater.update(p_fmat, _p_trees);
  }
  void PredictBatch(DMatrix *p_fmat,
                    HostDeviceVector<bst_float> *out_preds,
                    unsigned ntree_limit) override {
    // _updater.predict(p_fmat, _p_trees, out_preds);
  }
  void PredictLeaf(DMatrix *p_fmat,
                   std::vector<bst_float> *out_preds,
                   unsigned ntree_limit) override {}
  void PredictInstance(const SparsePage::Inst &inst,
                       std::vector<bst_float> *out_preds,
                       unsigned ntree_limit,
                       unsigned root_index) override {}
  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate, int condition = 0,
                           unsigned condition_feature = 0) override {}
  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       unsigned ntree_limit, bool approximate) override {}
  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    return {};
  }
};

XGBOOST_REGISTER_GBM(Bart, "bart")
    .describe("Bayesian Additive Tree.")
    .set_body([](const std::vector<std::shared_ptr<DMatrix> > &cache,
                 bst_float base_margin) {
      return new Bart(cache, base_margin);
    });
}

}  // namespace xgboost
