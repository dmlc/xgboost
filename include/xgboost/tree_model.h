/*!
 * Copyright 2014 by Contributors
 * \file tree_model.h
 * \brief model structure for tree
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_MODEL_H_
#define XGBOOST_TREE_MODEL_H_

#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <limits>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <tuple>
#include "./base.h"
#include "./data.h"
#include "./logging.h"
#include "./feature_map.h"

namespace xgboost {

struct PathElement;  // forward declaration

/*! \brief meta parameters of the tree */
struct TreeParam : public dmlc::Parameter<TreeParam> {
  /*! \brief number of start root */
  int num_roots;
  /*! \brief total number of nodes */
  int num_nodes;
  /*!\brief number of deleted nodes */
  int num_deleted;
  /*! \brief maximum depth, this is a statistics of the tree */
  int max_depth;
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
    num_nodes = num_roots = 1;
  }
  // declare the parameters
  DMLC_DECLARE_PARAMETER(TreeParam) {
    // only declare the parameters that can be set by the user.
    // other arguments are set by the algorithm.
    DMLC_DECLARE_FIELD(num_roots).set_lower_bound(1).set_default(1)
        .describe("Number of start root of trees.");
    DMLC_DECLARE_FIELD(num_feature)
        .describe("Number of features used in tree construction.");
    DMLC_DECLARE_FIELD(size_leaf_vector).set_lower_bound(0).set_default(0)
        .describe("Size of leaf vector, reserved for vector tree");
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
  int leaf_child_cnt;
};

/*!
 * \brief define regression tree to be the most common tree model.
 *  This is the data structure used in xgboost's major tree models.
 */
class RegTree {
 public:
  /*! \brief auxiliary statistics of node to help tree building */
  using SplitCondT = bst_float;
  /*! \brief tree node */
  class Node {
   public:
    Node()  {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    /*! \brief index of left child */
    int LeftChild() const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    int RightChild() const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    int DefaultChild() const {
      return this->DefaultLeft() ? this->LeftChild() : this->RightChild();
    }
    /*! \brief feature index of split condition */
    unsigned SplitIndex() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    bool DefaultLeft() const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    bool IsLeaf() const {
      return cleft_ == -1;
    }
    /*! \return get leaf value of leaf node */
    bst_float LeafValue() const {
      return (this->info_).leaf_value;
    }
    /*! \return get split condition of the node */
    SplitCondT SplitCond() const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    int Parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    bool IsLeftChild() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether this node is deleted */
    bool IsDeleted() const {
      return sindex_ == std::numeric_limits<unsigned>::max();
    }
    /*! \brief whether current node is root */
    bool IsRoot() const {
      return parent_ == -1;
    }
    /*!
     * \brief set the left child
     * \param nid node id to right child
     */
    void SetLeftChild(int nid) {
      this->cleft_ = nid;
    }
    /*!
     * \brief set the right child
     * \param nid node id to right child
     */
    void SetRightChild(int nid) {
      this->cright_ = nid;
    }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    void SetSplit(unsigned split_index, SplitCondT split_cond,
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
    void SetLeaf(bst_float value, int right = -1) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = right;
    }
    /*! \brief mark that this node is deleted */
    void MarkDelete() {
      this->sindex_ = std::numeric_limits<unsigned>::max();
    }
    // set parent
    void SetParent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
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
    int parent_;
    // pointer to left, right
    int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    unsigned sindex_{0};
    // extra info
    Info info_;
  };

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

  /*! \brief model parameter */
  TreeParam param;
  /*! \brief constructor */
  RegTree() {
    param.num_nodes = 1;
    param.num_roots = 1;
    param.num_deleted = 0;
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    for (int i = 0; i < param.num_nodes; i ++) {
      nodes_[i].SetLeaf(0.0f);
      nodes_[i].SetParent(-1);
    }
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
  void Load(dmlc::Stream* fi) {
    CHECK_EQ(fi->Read(&param, sizeof(TreeParam)), sizeof(TreeParam));
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    CHECK_NE(param.num_nodes, 0);
    CHECK_EQ(fi->Read(dmlc::BeginPtr(nodes_), sizeof(Node) * nodes_.size()),
             sizeof(Node) * nodes_.size());
    CHECK_EQ(fi->Read(dmlc::BeginPtr(stats_), sizeof(RTreeNodeStat) * stats_.size()),
             sizeof(RTreeNodeStat) * stats_.size());
    // chg deleted nodes
    deleted_nodes_.resize(0);
    for (int i = param.num_roots; i < param.num_nodes; ++i) {
      if (nodes_[i].IsDeleted()) deleted_nodes_.push_back(i);
    }
    CHECK_EQ(static_cast<int>(deleted_nodes_.size()), param.num_deleted);
  }
  /*!
   * \brief save model to stream
   * \param fo output stream
   */
  void Save(dmlc::Stream* fo) const {
    CHECK_EQ(param.num_nodes, static_cast<int>(nodes_.size()));
    CHECK_EQ(param.num_nodes, static_cast<int>(stats_.size()));
    fo->Write(&param, sizeof(TreeParam));
    CHECK_NE(param.num_nodes, 0);
    fo->Write(dmlc::BeginPtr(nodes_), sizeof(Node) * nodes_.size());
    fo->Write(dmlc::BeginPtr(stats_), sizeof(RTreeNodeStat) * nodes_.size());
  }

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
   */
  void ExpandNode(int nid, unsigned split_index, bst_float split_value,
                  bool default_left, bst_float base_weight,
                  bst_float left_leaf_weight, bst_float right_leaf_weight,
                  bst_float loss_change, float sum_hess) {
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
    // mark right child as 0, to indicate fresh leaf
    nodes_[pleft].SetLeaf(left_leaf_weight, 0);
    nodes_[pright].SetLeaf(right_leaf_weight, 0);

    this->Stat(nid).loss_chg = loss_change;
    this->Stat(nid).base_weight = base_weight;
    this->Stat(nid).sum_hess = sum_hess;
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
    int maxd = 0;
    for (int i = 0; i < param.num_roots; ++i) {
      maxd = std::max(maxd, MaxDepth(i));
    }
    return maxd;
  }

  /*! \brief number of extra nodes besides the root */
  int NumExtraNodes() const {
    return param.num_nodes - param.num_roots - param.num_deleted;
  }

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
    bst_float Fvalue(size_t i) const;
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
  /*!
   * \brief get the leaf index
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \return the leaf index of the given feature
   */
  int GetLeafIndex(const FVec& feat, unsigned root_id = 0) const;
  /*!
   * \brief calculate the feature contributions (https://arxiv.org/abs/1706.06060) for the tree
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \param out_contribs output vector to hold the contributions
   * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
   * \param condition_feature the index of the feature to fix
   */
  void CalculateContributions(const RegTree::FVec& feat, unsigned root_id,
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
   * \param root_id starting root index of the instance
   * \param out_contribs output vector to hold the contributions
   */
  void CalculateContributionsApprox(const RegTree::FVec& feat, unsigned root_id,
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

 private:
  // vector of nodes
  std::vector<Node> nodes_;
  // free node space, used during training process
  std::vector<int>  deleted_nodes_;
  // stats of nodes
  std::vector<RTreeNodeStat> stats_;
  std::vector<bst_float> node_mean_values_;
  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  int AllocNode() {
    if (param.num_deleted != 0) {
      int nd = deleted_nodes_.back();
      deleted_nodes_.pop_back();
      --param.num_deleted;
      return nd;
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
    CHECK_GE(nid, param.num_roots);
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
  for (bst_uint i = 0; i < inst.size(); ++i) {
    if (inst[i].index >= data_.size()) continue;
    data_[inst[i].index].fvalue = inst[i].fvalue;
  }
}

inline void RegTree::FVec::Drop(const SparsePage::Inst& inst) {
  for (bst_uint i = 0; i < inst.size(); ++i) {
    if (inst[i].index >= data_.size()) continue;
    data_[inst[i].index].flag = -1;
  }
}

inline size_t RegTree::FVec::Size() const {
  return data_.size();
}

inline bst_float RegTree::FVec::Fvalue(size_t i) const {
  return data_[i].fvalue;
}

inline bool RegTree::FVec::IsMissing(size_t i) const {
  return data_[i].flag == -1;
}

inline int RegTree::GetLeafIndex(const RegTree::FVec& feat,
                                 unsigned root_id) const {
  auto pid = static_cast<int>(root_id);
  while (!(*this)[pid].IsLeaf()) {
    unsigned split_index = (*this)[pid].SplitIndex();
    pid = this->GetNext(pid, feat.Fvalue(split_index), feat.IsMissing(split_index));
  }
  return pid;
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
