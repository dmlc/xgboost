#ifndef XGBOOST_TREE_UPDATER_QUANTILE_INL_HPP_
#define XGBOOST_TREE_UPDATER_QUANTILE_INL_HPP_
/*!
 * \file updater_quantile_colmaker-inl.hpp
 * \brief use columnwise update to construct a quantile regression tree
 */
#include <vector>
#include <algorithm>
#include "./param.h"
#include "./updater.h"
#include "./updater_colmaker-inl.hpp"
#include "./updater_prune-inl.hpp"


namespace xgboost {
namespace tree {

  void SetQuantileGPair(const std::vector<bst_gpair> & gpair,float quantile,std::vector<bst_gpair> & quantile_gpair) {

    for (unsigned i = 0; i < gpair.size(); i++) {
      //BUGBUG make sure the math is right
      if (gpair[i].grad > 0) {
	quantile_gpair.push_back(bst_gpair(quantile,1.0));
      } else {
	quantile_gpair.push_back(bst_gpair(1.0-quantile,1.0));
      }
    }
}
/*! \brief colunwise update to construct a tree */

template<typename TStats>
class QuantileColMaker: public ColMaker<TStats> {
 public:
  virtual ~QuantileColMaker(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    if (!strcmp(name, "quantile")) quantile = static_cast<float>(atof(val));
    else {
      ColMaker<TStats>::SetParam(name,val);
    }
  }

  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    //create new set of gradient pairs for quantile regression
    std::vector<bst_gpair> quantile_gpairs (gpair.size());
    SetQuantileGPair(gpair,quantile,quantile_gpairs);
    ColMaker<TStats>::Update(quantile_gpairs,p_fmat,info,trees);
  }

 protected:
  // quantile
  float quantile;
};


/*! \brief pruner that prunes a tree after growing finishs */
class QuantileTreePruner: public TreePruner {
 public:
  virtual ~QuantileTreePruner(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    // sync-names                                                                                                                                                                                  
    if (!strcmp(name, "quantile")) quantile = static_cast<float>(atof(val));
    else {
      TreePruner::SetParam(name,val);
    }
  }
  // update the tree, do pruning
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    // rescale learning rate according to size of trees
    std::vector<bst_gpair> quantile_gpairs (gpair.size());
    SetQuantileGPair(gpair,quantile,quantile_gpairs);
    TreePruner::Update(quantile_gpairs,p_fmat,info,trees);
  }
protected:
  float quantile;

};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_QUANTILE_INL_HPP_
