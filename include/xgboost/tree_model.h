/*!
 * Copyright 2014-2019 by Contributors
 * \file tree_model.h
 * \brief model structure for tree
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_MODEL_H_
#define XGBOOST_TREE_MODEL_H_

#include <dmlc/io.h>
#include <dmlc/parameter.h>

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <xgboost/feature_map.h>
#include <xgboost/model.h>

#include <limits>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <tuple>
#include <stack>

namespace xgboost {

struct PathElement;  // forward declaration

class Json;
// FIXME(trivialfis): Once binary IO is gone, make this parameter internal as it should
// not be configured by users.
/*! \brief meta parameters of the tree */
struct TreeParam : public dmlc::Parameter<TreeParam> {
  /*! \brief (Deprecated) number of start root */
  int deprecated_num_roots;
  /*! \brief total number of nodes */
  int num_nodes;
  /*!\brief number of deleted nodes */
  int num_deleted;
  /*! \brief maximum depth, this is a statistics of the tree */
  int deprecated_max_depth;
  /*! \brief number of features used for tree construction */
  int num_feature;
  /*!
   * \brief leaf vector size, used for vector tree
   * used to store more than one dimensional information in tree
   */
  int size_leaf_vector;
  /*! \brief reserved part, make sure alignment works for 64bit */
  int reserved[31];
  /*! \brief constructor */
  TreeParam() {
    // assert compact alignment
    static_assert(sizeof(TreeParam) == (31 + 6) * sizeof(int),
                  "TreeParam: 64 bit align");
    std::memset(this, 0, sizeof(TreeParam));
    num_nodes = 1;
    deprecated_num_roots = 1;
  }
  // declare the parameters
  DMLC_DECLARE_PARAMETER(TreeParam) {
    // only declare the parameters that can be set by the user.
    // other arguments are set by the algorithm.
    DMLC_DECLARE_FIELD(num_nodes).set_lower_bound(1).set_default(1);
    DMLC_DECLARE_FIELD(num_feature)
        .describe("Number of features used in tree construction.");
    DMLC_DECLARE_FIELD(num_deleted);
    DMLC_DECLARE_FIELD(size_leaf_vector).set_lower_bound(0).set_default(0)
        .describe("Size of leaf vector, reserved for vector tree");
  }

  bool operator==(const TreeParam& b) const {
    return num_nodes == b.num_nodes &&
           num_deleted == b.num_deleted &&
           num_feature == b.num_feature &&
           size_leaf_vector == b.size_leaf_vector;
  }
};

/*! \brief node statistics used in regression tree */
struct RTreeNodeStat {
  /*! \brief loss change caused by current split */
  bst_float loss_chg;
  /*! \brief sum of hessian values, used to measure coverage of data */
  bst_float sum_hess;
  /*! \brief weight of current node */
  bst_float base_weight;
  /*! \brief number of child that is leaf node known up to now */
  int leaf_child_cnt {0};

  RTreeNodeStat() = default;
  RTreeNodeStat(float loss_chg, float sum_hess, float weight) :
      loss_chg{loss_chg}, sum_hess{sum_hess}, base_weight{weight} {}
  bool operator==(const RTreeNodeStat& b) const {
    return loss_chg == b.loss_chg && sum_hess == b.sum_hess &&
           base_weight == b.base_weight && leaf_child_cnt == b.leaf_child_cnt;
  }
};

class MultiTargetTreeNodeStat {
  std::vector<float> loss_chg_;
  std::vector<double> sum_hess_;
  std::vector<float> base_weight_;
  size_t targets_;

 public:
  explicit MultiTargetTreeNodeStat(size_t targets) : targets_{targets} {}
  void Set(bst_node_t nidx, float loss_chg, common::Span<float const> weight,
           common::Span<double const> hess) {
    if (loss_chg_.size() < static_cast<size_t>(nidx + 1)) {
      loss_chg_.resize(nidx + 1);
    }
    loss_chg_[nidx] = loss_chg;
    size_t beg = nidx * targets_;
    size_t end = beg + targets_;
    if (sum_hess_.size() < end) {
      sum_hess_.resize(end);
      base_weight_.resize(end);
    }
    for (size_t i = beg; i < end; ++i) {
      sum_hess_[i] = hess[i - beg];
      base_weight_[i] = weight[i - beg];
    }
  }
  void Prune(bst_node_t nidx) {
    loss_chg_[nidx] = std::numeric_limits<float>::quiet_NaN();
  }
  bool IsDeleted(bst_node_t nidx) const {
    return std::isnan(loss_chg_[nidx]);
  }
};

/*!
 * \brief define regression tree to be the most common tree model.
 *  This is the data structure used in xgboost's major tree models.
 */
class RegTree : public Model {
 private:
  OutputType kind_ {OutputType::kSingle};

 public:
  using SplitCondT = bst_float;
  static constexpr bst_node_t kInvalidNodeId {-1};
  static constexpr uint32_t kDeletedNodeMarker = std::numeric_limits<uint32_t>::max();
  static constexpr bst_node_t kRoot { 0 };

  /*! \brief tree node */
  class Node {
   public:
    XGBOOST_DEVICE Node()  {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    Node(int32_t cleft, int32_t cright, int32_t parent,
         uint32_t split_ind, float split_cond, bool default_left) :
        parent_{parent}, cleft_{cleft}, cright_{cright} {
      this->SetParent(parent_);
      this->SetSplit(split_ind, split_cond, default_left);
    }

    /*! \brief index of left child */
    XGBOOST_DEVICE int LeftChild() const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    XGBOOST_DEVICE int RightChild() const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    XGBOOST_DEVICE int DefaultChild() const {
      return this->DefaultLeft() ? this->LeftChild() : this->RightChild();
    }
    /*! \brief feature index of split condition */
    XGBOOST_DEVICE unsigned SplitIndex() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    XGBOOST_DEVICE bool DefaultLeft() const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    XGBOOST_DEVICE bool IsLeaf() const {
      return cleft_ == kInvalidNodeId;
    }
    /*! \return get leaf value of leaf node */
    XGBOOST_DEVICE bst_float SingleLeafValue() const {
      return (this->info_).leaf_value;
    }
    /*! \return get split condition of the node */
    XGBOOST_DEVICE SplitCondT SplitCond() const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    XGBOOST_DEVICE int Parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    XGBOOST_DEVICE bool IsLeftChild() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether this node is deleted */
    XGBOOST_DEVICE bool IsDeleted() const {
      return sindex_ == kDeletedNodeMarker;
    }
    /*! \brief whether current node is root */
    XGBOOST_DEVICE bool IsRoot() const { return parent_ == kInvalidNodeId; }
    /*!
     * \brief set the left child
     * \param nid node id to right child
     */
    XGBOOST_DEVICE void SetLeftChild(int nid) {
      this->cleft_ = nid;
    }
    /*!
     * \brief set the right child
     * \param nid node id to right child
     */
    XGBOOST_DEVICE void SetRightChild(int nid) {
      this->cright_ = nid;
    }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    XGBOOST_DEVICE void SetSplit(unsigned split_index, SplitCondT split_cond,
                          bool default_left = false) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).split_cond = split_cond;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     * \param right right index, could be used to store
     *        additional information
     */
    XGBOOST_DEVICE void SetLeaf(bst_float value, int right = kInvalidNodeId) {
      (this->info_).leaf_value = value;
      this->cleft_ = kInvalidNodeId;
      this->cright_ = right;
    }
    /*! \brief mark that this node is deleted */
    XGBOOST_DEVICE void MarkDelete() {
      this->sindex_ = kDeletedNodeMarker;
    }
    /*! \brief Reuse this deleted node. */
    XGBOOST_DEVICE void Reuse() {
      this->sindex_ = 0;
    }
    // set parent
    XGBOOST_DEVICE void SetParent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
    bool operator==(const Node& b) const {
      return parent_ == b.parent_ && cleft_ == b.cleft_ &&
             cright_ == b.cright_ && sindex_ == b.sindex_ &&
             info_.leaf_value == b.info_.leaf_value;
    }

   private:
    /*!
     * \brief in leaf node, we have weights, in non-leaf nodes,
     *        we have split condition
     */
    union Info{
      bst_float leaf_value;
      SplitCondT split_cond;
    };
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int32_t parent_{kInvalidNodeId};
    // pointer to left, right
    int32_t cleft_{kInvalidNodeId}, cright_{kInvalidNodeId};
    // split feature index, left split or right split depends on the highest bit
    uint32_t sindex_{0};
    // extra info
    Info info_;
  };

  explicit RegTree(bst_feature_t leaf_size = 1,
                   OutputType kind = OutputType::kSingle)
      : kind_{kind}, leaf_size_{leaf_size}, multi_target_stats_{leaf_size} {
    param.num_nodes = 1;
    param.num_deleted = 0;
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    for (int i = 0; i < param.num_nodes; i ++) {
      nodes_[i].SetLeaf(0.0f);
      nodes_[i].SetParent(kInvalidNodeId);
    }

    if (leaf_size_ != 1) {
      leaf_values_.resize(leaf_size_);
      CHECK_EQ(static_cast<int32_t>(kind_), static_cast<int32_t>(OutputType::kMulti));
    }
  }
  /*!
   * \brief Return tree kind, kSingle or kMulti.
   */
  OutputType Kind() const { return kind_; }
  /*!
   * \brief Return the size of leaf.
   */
  bst_feature_t LeafSize() const { return leaf_size_; }
  /*!
   * \brief change a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  void ChangeToLeaf(int rid, bst_float value) {
    CHECK(nodes_[nodes_[rid].LeftChild() ].IsLeaf());
    CHECK(nodes_[nodes_[rid].RightChild()].IsLeaf());
    this->DeleteNode(nodes_[rid].LeftChild());
    this->DeleteNode(nodes_[rid].RightChild());
    nodes_[rid].SetLeaf(value);
  }
  /*!
   * \brief collapse a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  void CollapseToLeaf(int rid, bst_float value) {
    if (nodes_[rid].IsLeaf()) return;
    if (!nodes_[nodes_[rid].LeftChild() ].IsLeaf()) {
      CollapseToLeaf(nodes_[rid].LeftChild(), 0.0f);
    }
    if (!nodes_[nodes_[rid].RightChild() ].IsLeaf()) {
      CollapseToLeaf(nodes_[rid].RightChild(), 0.0f);
    }
    this->ChangeToLeaf(rid, value);
  }

  /*! \brief get node given nid */
  Node& operator[](int nid) {
    return nodes_[nid];
  }
  /*! \brief get node given nid */
  const Node& operator[](int nid) const {
    return nodes_[nid];
  }

  /*! \brief get const reference to nodes */
  const std::vector<Node>& GetNodes() const { return nodes_; }

  /*! \brief get node statistics given nid */
  RTreeNodeStat& Stat(int nid) {
    return stats_[nid];
  }
  /*! \brief get node statistics given nid */
  const RTreeNodeStat& Stat(int nid) const {
    return stats_[nid];
  }

  /*!
   * \brief load model from stream
   * \param fi input stream
   */
  void Load(dmlc::Stream* fi);
  /*!
   * \brief save model to stream
   * \param fo output stream
   */
  void Save(dmlc::Stream* fo) const;

  void LoadModel(Json const& in) override;
  void SaveModel(Json* out) const override;

  bool operator==(const RegTree& b) const {
    return nodes_ == b.nodes_ && stats_ == b.stats_ &&
           deleted_nodes_ == b.deleted_nodes_ && param == b.param;
  }
  /* \brief Iterate through all nodes in this tree.
   *
   * \param Function that accepts a node index, and returns false when iteration should
   *        stop, otherwise returns true.
   */
  template <typename Func> void WalkTree(Func func) const {
    std::stack<bst_node_t> nodes;
    nodes.push(kRoot);
    auto &self = *this;
    while (!nodes.empty()) {
      auto nidx = nodes.top();
      nodes.pop();
      if (!func(nidx)) {
        return;
      }
      auto left = self[nidx].LeftChild();
      auto right = self[nidx].RightChild();
      if (left != RegTree::kInvalidNodeId) {
        nodes.push(left);
      }
      if (right != RegTree::kInvalidNodeId) {
        nodes.push(right);
      }
    }
  }
  /*!
   * \brief Compares whether 2 trees are equal from a user's perspective.  The equality
   *        compares only non-deleted nodes.
   *
   * \parm b The other tree.
   */
  bool Equal(const RegTree& b) const;

  /**
   * \brief Expands a leaf node into two additional leaf nodes.
   *
   * \param nid               The node index to expand.
   * \param split_index       Feature index of the split.
   * \param split_value       The split condition.
   * \param default_left      True to default left.
   * \param base_weight       The base weight, before learning rate.
   * \param left_leaf_weight  The left leaf weight for prediction, modified by learning rate.
   * \param right_leaf_weight The right leaf weight for prediction, modified by learning rate.
   * \param loss_change       The loss change.
   * \param sum_hess          The sum hess.
   * \param left_sum          The sum hess of left leaf.
   * \param right_sum         The sum hess of right leaf.
   * \param leaf_right_child  The right child index of leaf, by default kInvalidNodeId,
   *                          some updaters use the right child index of leaf as a marker
   */
  void ExpandNode(int nid, unsigned split_index, bst_float split_value,
                  bool default_left, bst_float base_weight,
                  bst_float left_leaf_weight, bst_float right_leaf_weight,
                  bst_float loss_change, float sum_hess, float left_sum,
                  float right_sum,
                  bst_node_t leaf_right_child = kInvalidNodeId) {
    int pleft = this->AllocNode();
    int pright = this->AllocNode();
    auto &node = nodes_[nid];
    CHECK(node.IsLeaf());
    node.SetLeftChild(pleft);
    node.SetRightChild(pright);
    nodes_[node.LeftChild()].SetParent(nid, true);
    nodes_[node.RightChild()].SetParent(nid, false);
    node.SetSplit(split_index, split_value,
                  default_left);

    nodes_[pleft].SetLeaf(left_leaf_weight, leaf_right_child);
    nodes_[pright].SetLeaf(right_leaf_weight, leaf_right_child);

    this->Stat(nid)    = {loss_change, sum_hess, base_weight};
    this->Stat(pleft)  = {0.0f, left_sum, left_leaf_weight};
    this->Stat(pright) = {0.0f, right_sum, right_leaf_weight};
  }

  void ExpandNode(int nid, unsigned split_index,
                  bst_float split_value,
                  bool default_left, std::vector<float> const& base_weight,
                  std::vector<float> const& left_leaf_weight,
                  std::vector<float> const& right_leaf_weight,
                  bst_float loss_change,
                  std::vector<double> const& sum_hess,
                  std::vector<double> const& left_sum, std::vector<double> const& right_sum) {
    int pleft = this->AllocNode();
    int pright = this->AllocNode();
    auto &node = nodes_[nid];
    CHECK(node.IsLeaf());
    node.SetLeftChild(pleft);
    node.SetRightChild(pright);
    nodes_[node.LeftChild()].SetParent(nid, true);
    nodes_[node.RightChild()].SetParent(nid, false);
    node.SetSplit(split_index, split_value,
                  default_left);
    this->SetLeaf(left_leaf_weight, pleft);
    this->SetLeaf(right_leaf_weight, pright);
    this->multi_target_stats_.Set(nid, loss_change,
                                  common::Span<float const>{base_weight},
                                  common::Span<double const>{sum_hess});
    this->multi_target_stats_.Set(pleft, 0, {left_leaf_weight}, {left_sum});
    this->multi_target_stats_.Set(pright, 0, {right_leaf_weight}, {right_sum});
  }

  /*!
   * \brief get current depth
   * \param nid node id
   */
  int GetDepth(int nid) const {
    int depth = 0;
    while (!nodes_[nid].IsRoot()) {
      ++depth;
      nid = nodes_[nid].Parent();
    }
    return depth;
  }
  /*!
   * \brief get maximum depth
   * \param nid node id
   */
  int MaxDepth(int nid) const {
    if (nodes_[nid].IsLeaf()) return 0;
    return std::max(MaxDepth(nodes_[nid].LeftChild())+1,
                     MaxDepth(nodes_[nid].RightChild())+1);
  }

  /*!
   * \brief get maximum depth
   */
  int MaxDepth() {
    return MaxDepth(0);
  }

  /*! \brief number of extra nodes besides the root */
  int NumExtraNodes() const {
    return param.num_nodes - 1 - param.num_deleted;
  }

  /* \brief Count number of leaves in tree. */
  bst_node_t GetNumLeaves() const;
  bst_node_t GetNumSplitNodes() const;

  /*!
   * \brief dense feature vector that can be taken by RegTree
   * and can be construct from sparse feature vector.
   */
  struct FVec {
    /*!
     * \brief initialize the vector with size vector
     * \param size The size of the feature vector.
     */
    void Init(size_t size);
    /*!
     * \brief fill the vector with sparse vector
     * \param inst The sparse instance to fill.
     */
    void Fill(const SparsePage::Inst& inst);
    /*!
     * \brief drop the trace after fill, must be called after fill.
     * \param inst The sparse instance to drop.
     */
    void Drop(const SparsePage::Inst& inst);
    /*!
     * \brief returns the size of the feature vector
     * \return the size of the feature vector
     */
    size_t Size() const;
    /*!
     * \brief get ith value
     * \param i feature index.
     * \return the i-th feature value
     */
    bst_float GetFvalue(size_t i) const;
    /*!
     * \brief check whether i-th entry is missing
     * \param i feature index.
     * \return whether i-th value is missing.
     */
    bool IsMissing(size_t i) const;

   private:
    /*!
     * \brief a union value of value and flag
     *  when flag == -1, this indicate the value is missing
     */
    union Entry {
      bst_float fvalue;
      int flag;
    };
    std::vector<Entry> data_;
  };

  common::Span<float const> VectorLeafValue(bst_node_t nidx) const {
    CHECK_EQ(static_cast<int32_t>(kind_), static_cast<int32_t>(OutputType::kMulti));
    auto s = common::Span<float const> {leaf_values_}.subspan(nidx * leaf_size_, leaf_size_);
    return s;
  }
  float LeafValue(bst_node_t nidx) const {
    CHECK_EQ(kind_, OutputType::kSingle);
    return (*this)[nidx].SingleLeafValue();
  }

  void SetLeaf(std::vector<float> const& leaf, bst_node_t nid,
               std::vector<double> const& sum_hess = {}) {
    (*this)[nid].SetLeaf(0);
    auto offset = LeafSize() * nid;
    if (leaf_values_.size() < offset + LeafSize()) {
      leaf_values_.resize(offset + LeafSize());
    }
    CHECK_EQ(leaf.size(), LeafSize());
    for (size_t i = 0; i < LeafSize(); ++i) {
      leaf_values_[i + offset] = leaf[i];
    }
    (*this)[nid].SetLeftChild(kInvalidNodeId);
    (*this)[nid].SetRightChild(kInvalidNodeId);

    if (sum_hess.size() != 0) {
      this->multi_target_stats_.Set(nid, 0, leaf, sum_hess);
    }
  }
  void SetLeaf(float const& leaf, bst_node_t nid,
               double sum_hess = 0) {
    (*this)[nid].SetLeaf(leaf);
    this->Stat(nid).loss_chg = 0;
    this->Stat(nid).base_weight = leaf;
    this->Stat(nid).sum_hess = sum_hess;
  }
  /*!
   * \brief get the leaf index
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \return the leaf index of the given feature
   */
  int GetLeafIndex(const FVec& feat) const;
  /*!
   * \brief calculate the feature contributions (https://arxiv.org/abs/1706.06060) for the tree
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param out_contribs output vector to hold the contributions
   * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
   * \param condition_feature the index of the feature to fix
   */
  void CalculateContributions(const RegTree::FVec& feat,
                              bst_float* out_contribs, int condition = 0,
                              unsigned condition_feature = 0) const;
  /*!
   * \brief Recursive function that computes the feature attributions for a single tree.
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param phi dense output vector of feature attributions
   * \param node_index the index of the current node in the tree
   * \param unique_depth how many unique features are above the current node in the tree
   * \param parent_unique_path a vector of statistics about our current path through the tree
   * \param parent_zero_fraction what fraction of the parent path weight is coming as 0 (integrated)
   * \param parent_one_fraction what fraction of the parent path weight is coming as 1 (fixed)
   * \param parent_feature_index what feature the parent node used to split
   * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
   * \param condition_feature the index of the feature to fix
   * \param condition_fraction what fraction of the current weight matches our conditioning feature
   */
  void TreeShap(const RegTree::FVec& feat, bst_float* phi, unsigned node_index,
                unsigned unique_depth, PathElement* parent_unique_path,
                bst_float parent_zero_fraction, bst_float parent_one_fraction,
                int parent_feature_index, int condition,
                unsigned condition_feature, bst_float condition_fraction) const;

  /*!
   * \brief calculate the approximate feature contributions for the given root
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param out_contribs output vector to hold the contributions
   */
  void CalculateContributionsApprox(const RegTree::FVec& feat,
                                    bst_float* out_contribs) const;
  /*!
   * \brief get next position of the tree given current pid
   * \param pid Current node id.
   * \param fvalue feature value if not missing.
   * \param is_unknown Whether current required feature is missing.
   */
  inline int GetNext(int pid, bst_float fvalue, bool is_unknown) const;
  /*!
   * \brief dump the model in the requested format as a text string
   * \param fmap feature map that may help give interpretations of feature
   * \param with_stats whether dump out statistics as well
   * \param format the format to dump the model in
   * \return the string of dumped model
   */
  std::string DumpModel(const FeatureMap& fmap,
                        bool with_stats,
                        std::string format) const;
  /*!
   * \brief calculate the mean value for each node, required for feature contributions
   */
  void FillNodeMeanValues();

  /*! \brief model parameter */
  TreeParam param;

 private:
  // vector of nodes
  std::vector<Node> nodes_;
  // free node space, used during training process
  std::vector<int>  deleted_nodes_;
  // stats of nodes
  std::vector<RTreeNodeStat> stats_;
  std::vector<bst_float> node_mean_values_;

  bst_feature_t leaf_size_ {0};
  std::vector<float> leaf_values_;
  MultiTargetTreeNodeStat multi_target_stats_;

  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  int AllocNode() {
    if (param.num_deleted != 0) {
      int nid = deleted_nodes_.back();
      deleted_nodes_.pop_back();
      nodes_[nid].Reuse();
      --param.num_deleted;
      return nid;
    }
    int nd = param.num_nodes++;
    CHECK_LT(param.num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    return nd;
  }
  // delete a tree node, keep the parent field to allow trace back
  void DeleteNode(int nid) {
    CHECK_GE(nid, 1);
    auto pid = (*this)[nid].Parent();
    if (nid == (*this)[pid].LeftChild()) {
      (*this)[pid].SetLeftChild(kInvalidNodeId);
    } else {
      (*this)[pid].SetRightChild(kInvalidNodeId);
    }

    deleted_nodes_.push_back(nid);
    nodes_[nid].MarkDelete();
    ++param.num_deleted;
  }
  bst_float FillNodeMeanValue(int nid);
};

inline void RegTree::FVec::Init(size_t size) {
  Entry e; e.flag = -1;
  data_.resize(size);
  std::fill(data_.begin(), data_.end(), e);
}

inline void RegTree::FVec::Fill(const SparsePage::Inst& inst) {
  for (auto const& entry : inst) {
    if (entry.index >= data_.size()) {
      continue;
    }
    data_[entry.index].fvalue = entry.fvalue;
  }
}

inline void RegTree::FVec::Drop(const SparsePage::Inst& inst) {
  for (auto const& entry : inst) {
    if (entry.index >= data_.size()) {
      continue;
    }
    data_[entry.index].flag = -1;
  }
}

inline size_t RegTree::FVec::Size() const {
  return data_.size();
}

inline bst_float RegTree::FVec::GetFvalue(size_t i) const {
  return data_[i].fvalue;
}

inline bool RegTree::FVec::IsMissing(size_t i) const {
  return data_[i].flag == -1;
}

inline int RegTree::GetLeafIndex(const RegTree::FVec& feat) const {
  bst_node_t nid = 0;
  while (!(*this)[nid].IsLeaf()) {
    unsigned split_index = (*this)[nid].SplitIndex();
    nid = this->GetNext(nid, feat.GetFvalue(split_index), feat.IsMissing(split_index));
  }
  return nid;
}

/*! \brief get next position of the tree given current pid */
inline int RegTree::GetNext(int pid, bst_float fvalue, bool is_unknown) const {
  bst_float split_value = (*this)[pid].SplitCond();
  if (is_unknown) {
    return (*this)[pid].DefaultChild();
  } else {
    if (fvalue < split_value) {
      return (*this)[pid].LeftChild();
    } else {
      return (*this)[pid].RightChild();
    }
  }
}
}  // namespace xgboost
#endif  // XGBOOST_TREE_MODEL_H_
