/*!
 * Copyright 2014-2022 by XGBoost Contributors
 * \file tree_updater.h
 * \brief General primitive for tree learning,
 *   Updating a collection of trees given the information.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_H_
#define XGBOOST_TREE_UPDATER_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/linalg.h>
#include <xgboost/model.h>
#include <xgboost/task.h>
#include <xgboost/tree_model.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {

class Json;

/*!
 * \brief interface of tree update module, that performs update of a tree.
 */
class TreeUpdater : public Configurable {
 protected:
  GenericParameter const* ctx_ = nullptr;

 public:
  explicit TreeUpdater(const GenericParameter* ctx) : ctx_(ctx) {}
  /*! \brief virtual destructor */
  ~TreeUpdater() override = default;
  /*!
   * \brief Initialize the updater with given arguments.
   * \param args arguments to the objective function.
   */
  virtual void Configure(const Args& args) = 0;
  /*! \brief Whether this updater can be used for updating existing trees.
   *
   *  Some updaters are used for building new trees (like `hist`), while some others are
   *  used for modifying existing trees (like `prune`).  Return true if it can modify
   *  existing trees.
   */
  virtual bool CanModifyTree() const { return false; }
  /*!
   * \brief Wether the out_position in `Update` is valid. This determines whether adaptive
   *        tree can be used.
   */
  virtual bool HasNodePosition() const { return false; }
  /*!
   * \brief perform update to the tree models
   * \param gpair the gradient pair statistics of the data
   * \param data The data matrix passed to the updater.
   * \param out_position The leaf index for each row.  The index is negated if that row is
   *                     removed during sampling. So the 3th node is ~3.
   * \param out_trees references the trees to be updated, updater will change the content of trees
   *   note: all the trees in the vector are updated, with the same statistics,
   *         but maybe different random seeds, usually one tree is passed in at a time,
   *         there can be multiple trees when we train random forest style model
   */
  virtual void Update(HostDeviceVector<GradientPair>* gpair, DMatrix* data,
                      common::Span<HostDeviceVector<bst_node_t>> out_position,
                      const std::vector<RegTree*>& out_trees) = 0;

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
  virtual bool UpdatePredictionCache(const DMatrix * /*data*/,
                                     linalg::VectorView<float> /*out_preds*/) {
    return false;
  }

  virtual char const* Name() const = 0;

  /*!
   * \brief Create a tree updater given name
   * \param name Name of the tree updater.
   * \param tparam A global runtime parameter
   */
  static TreeUpdater* Create(const std::string& name, GenericParameter const* tparam, ObjInfo task);
};

/*!
 * \brief Registry entry for tree updater.
 */
struct TreeUpdaterReg
    : public dmlc::FunctionRegEntryBase<
          TreeUpdaterReg,
          std::function<TreeUpdater*(GenericParameter const* tparam, ObjInfo task)> > {};

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
