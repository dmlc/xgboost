/*!
 * Copyright 2014 by Contributors
 * \file tree_updater.h
 * \brief General primitive for tree learning,
 *   Updating a collection of trees given the information.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_H_
#define XGBOOST_TREE_UPDATER_H_

#include <dmlc/registry.h>
#include <functional>
#include <vector>
#include <utility>
#include <string>
#include "./base.h"
#include "./data.h"
#include "./tree_model.h"
#include "../../src/common/host_device_vector.h"

namespace xgboost {
/*!
 * \brief interface of tree update module, that performs update of a tree.
 */
class TreeUpdater {
 public:
  /*! \brief virtual destructor */
  virtual ~TreeUpdater() = default;
  /*!
   * \brief Initialize the updater with given arguments.
   * \param args arguments to the objective function.
   */
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& args) = 0;
  /*!
   * \brief perform update to the tree models
   * \param gpair the gradient pair statistics of the data
   * \param data The data matrix passed to the updater.
   * \param trees references the trees to be updated, updater will change the content of trees
   *   note: all the trees in the vector are updated, with the same statistics,
   *         but maybe different random seeds, usually one tree is passed in at a time,
   *         there can be multiple trees when we train random forest style model
   */
  virtual void Update(HostDeviceVector<GradientPair>* gpair,
                      DMatrix* data,
                      const std::vector<RegTree*>& trees) = 0;

  /*!
   * \brief determines whether updater has enough knowledge about a given dataset
   *        to quickly update prediction cache its training data and performs the
   *        update if possible.
   * \param data: data matrix
   * \param out_preds: prediction cache to be updated
   * \return boolean indicating whether updater has capability to update
   *         the prediction cache. If true, the prediction cache will have been
   *         updated by the time this function returns.
   */
  virtual bool UpdatePredictionCache(const DMatrix* data,
                                     HostDeviceVector<bst_float>* out_preds) {
    return false;
  }

  /*!
   * \brief Create a tree updater given name
   * \param name Name of the tree updater.
   */
  static TreeUpdater* Create(const std::string& name);
};

/*!
 * \brief Registry entry for tree updater.
 */
struct TreeUpdaterReg
    : public dmlc::FunctionRegEntryBase<TreeUpdaterReg,
                                        std::function<TreeUpdater* ()> > {
};

/*!
 * \brief Macro to register tree updater.
 *
 * \code
 * // example of registering a objective ndcg@k
 * XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "colmaker")
 * .describe("Column based tree maker.")
 * .set_body([]() {
 *     return new ColMaker<TStats>();
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_TREE_UPDATER(UniqueId, Name)                   \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::TreeUpdaterReg&               \
  __make_ ## TreeUpdaterReg ## _ ## UniqueId ## __ =                    \
      ::dmlc::Registry< ::xgboost::TreeUpdaterReg>::Get()->__REGISTER__(Name)

}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_H_
