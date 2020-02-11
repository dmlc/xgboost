/*!
 * Copyright 2014-2020 by Contributors
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
#include <xgboost/tree_model.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/model.h>

#include <functional>
#include <vector>
#include <utility>
#include <string>

namespace xgboost {

class Json;

/*!
 * \brief interface of tree update module, that performs update of a tree.
 */
class TreeUpdater : public Configurable {
 protected:
  GenericParameter const* tparam_;

 public:
  /*! \brief virtual destructor */
  ~TreeUpdater() override = default;
  /*!
   * \brief Initialize the updater with given arguments.
   * \param args arguments to the objective function.
   */
  virtual void Configure(const Args& args) = 0;
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

  virtual char const* Name() const = 0;

  /*!
   * \brief Create a tree updater given name
   * \param name Name of the tree updater.
   */
  static TreeUpdater* Create(const std::string& name, GenericParameter const* tparam);


  /** \brief Used to demarcate a contiguous set of row indices associated with
   * some tree node. */
  struct Segment {
    size_t begin {0};
    size_t end {0};

    Segment(size_t _begin, size_t _end) : begin(_begin), end(_end) {
      CHECK_GE(end, begin);
    }
    Segment() = default;
    size_t Size() const { return end - begin; }
  };

  /*!
   * \brief A cache storing node id -> row indices, used by CPU/GPU histogram algorithm.
   */
  class LeafIndexContainer {
    /*! \brief In here if you want to find the rows belong to a node nid, first you need to
     * get the indices segment from ridx_segments[nid], then get the row index that
     * represents position of row in input data X.
     *
     * mapping for node id -> rows.
     * This looks like:
     * node id      |    1    |    2   |
     * rid_segments | {0, 3}  | {3, 5} |
     * row_index    | 3, 5, 1 | 13, 31 |
     */
    /*! \brief Range of row index for each node, pointers into ridx below. */
    std::vector<Segment> ridx_segments_;
    HostDeviceVector<size_t> row_index_;

   public:
    using const_iterator = typename std::vector<Segment>::const_iterator; // NOLINT
    using iterator = typename std::vector<Segment>::iterator;  // NOLINT
    using value_type = Segment;                                // NOLINT
    using pointer = value_type*;                               // NOLINT
    using reference = value_type&;                             // NOLINT
    using index_type = int32_t;  // NOLINT Node ID type

    LeafIndexContainer() = default;

    std::vector<Segment>::const_iterator cbegin() const {  // NOLINT
      return ridx_segments_.cbegin();
    }
    std::vector<Segment>::const_iterator cend() const {  // NOLINT
      return ridx_segments_.cend();
    }

    /*! \brief return corresponding element set given the node_id.  Span points to host
     *  memory. */
    common::Span<size_t const> HostRows(int32_t node_id) const;
    common::Span<size_t> HostRows(int32_t node_id);

    /*! \brief return corresponding element set given the node_id.  Span points to device
     *  memory. */
    common::Span<size_t const> DeviceRows(int32_t node_id) const;
    common::Span<size_t> DeviceRows(int32_t node_id);

    /*! \brief Returns the segment belonging to node id. */
    Segment NodeSegment(int32_t node_id) const {
      return ridx_segments_.at(node_id);
    }

    /*! \brief Initialize the storage.  Due to CPU/GPU hist have their own way of
     *  initialization.  This only creates the storage and expand a root node which
     *  specifies total number of rows. */
    void Init(size_t num_rows, int device);
    /*! \brief Returns a Span pointing to all row indices in host memory. */
    common::Span<size_t> GetRows() {
      auto* ptr = row_index_.HostVector().data();
      auto size = row_index_.HostVector().size();
      return common::Span<size_t>{ ptr, size };
    }

    /*! \brief Returns the row indices as HostDeviceVector. */
    HostDeviceVector<size_t>& RowIndices() {
      return this->row_index_;
    }
    /*! \brief Returns the row indices segments. */
    auto RowSegments() -> decltype(this->ridx_segments_)& {
      return this->ridx_segments_;
    }

    /*!
     * \brief  Add a new node split to thie cache.
     *
     * \param node_id The node being split.
     * \param left_count Number of rows assigned to left node.
     * \param left_node_id left child ID.
     * \param right_node_id right child ID.
     */
    void AddSplit(int32_t node_id, size_t left_count,
                  int32_t left_node_id, int32_t right_node_id);

    void Clear() {
      this->row_index_.Resize(0);
      this->ridx_segments_.clear();
    }
  };
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
