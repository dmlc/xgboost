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
#include "./base.h"
#include "./data.h"
#include "./logging.h"
#include "./feature_map.h"

namespace xgboost {

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

/*!
 * \brief template class of TreeModel
 * \tparam TSplitCond data type to indicate split condition
 * \tparam TNodeStat auxiliary statistics of node to help tree building
 */
template<typename TSplitCond, typename TNodeStat>
class TreeModel {
 public:
  /*! \brief data type to indicate split condition */
  typedef TNodeStat  NodeStat;
  /*! \brief auxiliary statistics of node to help tree building */
  typedef TSplitCond SplitCond;
  /*! \brief tree node */
  class Node {
   public:
    Node() : sindex_(0) {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    /*! \brief index of left child */
    inline int cleft() const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    inline int cright() const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int cdefault() const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    /*! \brief feature index of split condition */
    inline unsigned split_index() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool default_left() const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    inline bool is_leaf() const {
      return cleft_ == -1;
    }
    /*! \return get leaf value of leaf node */
    inline bst_float leaf_value() const {
      return (this->info_).leaf_value;
    }
    /*! \return get split condition of the node */
    inline TSplitCond split_cond() const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    inline int parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    inline bool is_left_child() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether this node is deleted */
    inline bool is_deleted() const {
      return sindex_ == std::numeric_limits<unsigned>::max();
    }
    /*! \brief whether current node is root */
    inline bool is_root() const {
      return parent_ == -1;
    }
    /*!
     * \brief set the right child
     * \param nid node id to right child
     */
    inline void set_right_child(int nid) {
      this->cright_ = nid;
    }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    inline void set_split(unsigned split_index, TSplitCond split_cond,
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
    inline void set_leaf(bst_float value, int right = -1) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = right;
    }
    /*! \brief mark that this node is deleted */
    inline void mark_delete() {
      this->sindex_ = std::numeric_limits<unsigned>::max();
    }

   private:
    friend class TreeModel<TSplitCond, TNodeStat>;
    /*!
     * \brief in leaf node, we have weights, in non-leaf nodes,
     *        we have split condition
     */
    union Info{
      bst_float leaf_value;
      TSplitCond split_cond;
    };
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    int parent_;
    // pointer to left, right
    int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    unsigned sindex_;
    // extra info
    Info info_;
    // set parent
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
  };

 protected:
  // vector of nodes
  std::vector<Node> nodes;
  // free node space, used during training process
  std::vector<int>  deleted_nodes;
  // stats of nodes
  std::vector<TNodeStat> stats;
  // leaf vector, that is used to store additional information
  std::vector<bst_float> leaf_vector;
  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  inline int AllocNode() {
    if (param.num_deleted != 0) {
      int nd = deleted_nodes.back();
      deleted_nodes.pop_back();
      --param.num_deleted;
      return nd;
    }
    int nd = param.num_nodes++;
    CHECK_LT(param.num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    leaf_vector.resize(param.num_nodes * param.size_leaf_vector);
    return nd;
  }
  // delete a tree node, keep the parent field to allow trace back
  inline void DeleteNode(int nid) {
    CHECK_GE(nid, param.num_roots);
    deleted_nodes.push_back(nid);
    nodes[nid].mark_delete();
    ++param.num_deleted;
  }

 public:
  /*!
   * \brief change a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  inline void ChangeToLeaf(int rid, bst_float value) {
    CHECK(nodes[nodes[rid].cleft() ].is_leaf());
    CHECK(nodes[nodes[rid].cright()].is_leaf());
    this->DeleteNode(nodes[rid].cleft());
    this->DeleteNode(nodes[rid].cright());
    nodes[rid].set_leaf(value);
  }
  /*!
   * \brief collapse a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  inline void CollapseToLeaf(int rid, bst_float value) {
    if (nodes[rid].is_leaf()) return;
    if (!nodes[nodes[rid].cleft() ].is_leaf()) {
      CollapseToLeaf(nodes[rid].cleft(), 0.0f);
    }
    if (!nodes[nodes[rid].cright() ].is_leaf()) {
      CollapseToLeaf(nodes[rid].cright(), 0.0f);
    }
    this->ChangeToLeaf(rid, value);
  }

 public:
  /*! \brief model parameter */
  TreeParam param;
  /*! \brief constructor */
  TreeModel() {
    param.num_nodes = 1;
    param.num_roots = 1;
    param.num_deleted = 0;
    nodes.resize(1);
  }
  /*! \brief get node given nid */
  inline Node& operator[](int nid) {
    return nodes[nid];
  }
  /*! \brief get node given nid */
  inline const Node& operator[](int nid) const {
    return nodes[nid];
  }

  /*! \brief get const reference to nodes */
  inline const std::vector<Node>& GetNodes() const { return nodes; }

  /*! \brief get node statistics given nid */
  inline NodeStat& stat(int nid) {
    return stats[nid];
  }
  /*! \brief get node statistics given nid */
  inline const NodeStat& stat(int nid) const {
    return stats[nid];
  }
  /*! \brief get leaf vector given nid */
  inline bst_float* leafvec(int nid) {
    if (leaf_vector.size() == 0) return nullptr;
    return& leaf_vector[nid * param.size_leaf_vector];
  }
  /*! \brief get leaf vector given nid */
  inline const bst_float* leafvec(int nid) const {
    if (leaf_vector.size() == 0) return nullptr;
    return& leaf_vector[nid * param.size_leaf_vector];
  }
  /*! \brief initialize the model */
  inline void InitModel() {
    param.num_nodes = param.num_roots;
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    leaf_vector.resize(param.num_nodes * param.size_leaf_vector, 0.0f);
    for (int i = 0; i < param.num_nodes; i ++) {
      nodes[i].set_leaf(0.0f);
      nodes[i].set_parent(-1);
    }
  }
  /*!
   * \brief load model from stream
   * \param fi input stream
   */
  inline void Load(dmlc::Stream* fi) {
    CHECK_EQ(fi->Read(&param, sizeof(TreeParam)), sizeof(TreeParam));
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    CHECK_NE(param.num_nodes, 0);
    CHECK_EQ(fi->Read(dmlc::BeginPtr(nodes), sizeof(Node) * nodes.size()),
             sizeof(Node) * nodes.size());
    CHECK_EQ(fi->Read(dmlc::BeginPtr(stats), sizeof(NodeStat) * stats.size()),
             sizeof(NodeStat) * stats.size());
    if (param.size_leaf_vector != 0) {
      CHECK(fi->Read(&leaf_vector));
    }
    // chg deleted nodes
    deleted_nodes.resize(0);
    for (int i = param.num_roots; i < param.num_nodes; ++i) {
      if (nodes[i].is_deleted()) deleted_nodes.push_back(i);
    }
    CHECK_EQ(static_cast<int>(deleted_nodes.size()), param.num_deleted);
  }
  /*!
   * \brief save model to stream
   * \param fo output stream
   */
  inline void Save(dmlc::Stream* fo) const {
    CHECK_EQ(param.num_nodes, static_cast<int>(nodes.size()));
    CHECK_EQ(param.num_nodes, static_cast<int>(stats.size()));
    fo->Write(&param, sizeof(TreeParam));
    CHECK_NE(param.num_nodes, 0);
    fo->Write(dmlc::BeginPtr(nodes), sizeof(Node) * nodes.size());
    fo->Write(dmlc::BeginPtr(stats), sizeof(NodeStat) * nodes.size());
    if (param.size_leaf_vector != 0) fo->Write(leaf_vector);
  }
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid) {
    int pleft  = this->AllocNode();
    int pright = this->AllocNode();
    nodes[nid].cleft_  = pleft;
    nodes[nid].cright_ = pright;
    nodes[nodes[nid].cleft() ].set_parent(nid, true);
    nodes[nodes[nid].cright()].set_parent(nid, false);
  }
  /*!
   * \brief only add a right child to a leaf node
   * \param nid node id to add right child
   */
  inline void AddRightChild(int nid) {
    int pright = this->AllocNode();
    nodes[nid].right  = pright;
    nodes[nodes[nid].right].set_parent(nid, false);
  }
  /*!
   * \brief get current depth
   * \param nid node id
   * \param pass_rchild whether right child is not counted in depth
   */
  inline int GetDepth(int nid, bool pass_rchild = false) const {
    int depth = 0;
    while (!nodes[nid].is_root()) {
      if (!pass_rchild || nodes[nid].is_left_child()) ++depth;
      nid = nodes[nid].parent();
    }
    return depth;
  }
  /*!
   * \brief get maximum depth
   * \param nid node id
   */
  inline int MaxDepth(int nid) const {
    if (nodes[nid].is_leaf()) return 0;
    return std::max(MaxDepth(nodes[nid].cleft())+1,
                     MaxDepth(nodes[nid].cright())+1);
  }
  /*!
   * \brief get maximum depth
   */
  inline int MaxDepth() {
    int maxd = 0;
    for (int i = 0; i < param.num_roots; ++i) {
      maxd = std::max(maxd, MaxDepth(i));
    }
    return maxd;
  }
  /*! \brief number of extra nodes besides the root */
  inline int num_extra_nodes() const {
    return param.num_nodes - param.num_roots - param.num_deleted;
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
class RegTree: public TreeModel<bst_float, RTreeNodeStat> {
 public:
  /*!
   * \brief dense feature vector that can be taken by RegTree
   * and can be construct from sparse feature vector.
   */
  struct FVec {
   public:
    /*!
     * \brief initialize the vector with size vector
     * \param size The size of the feature vector.
     */
    inline void Init(size_t size);
    /*!
     * \brief fill the vector with sparse vector
     * \param inst The sparse instance to fill.
     */
    inline void Fill(const RowBatch::Inst& inst);
    /*!
     * \brief drop the trace after fill, must be called after fill.
     * \param inst The sparse instance to drop.
     */
    inline void Drop(const RowBatch::Inst& inst);
    /*!
     * \brief returns the size of the feature vector
     * \return the size of the feature vector
     */
    inline size_t size() const;
    /*!
     * \brief get ith value
     * \param i feature index.
     * \return the i-th feature value
     */
    inline bst_float fvalue(size_t i) const;
    /*!
     * \brief check whether i-th entry is missing
     * \param i feature index.
     * \return whether i-th value is missing.
     */
    inline bool is_missing(size_t i) const;

   private:
    /*!
     * \brief a union value of value and flag
     *  when flag == -1, this indicate the value is missing
     */
    union Entry {
      bst_float fvalue;
      int flag;
    };
    std::vector<Entry> data;
  };
  /*!
   * \brief get the leaf index
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \return the leaf index of the given feature
   */
  inline int GetLeafIndex(const FVec& feat, unsigned root_id = 0) const;
  /*!
   * \brief get the prediction of regression tree, only accepts dense feature vector
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \return the leaf index of the given feature
   */
  inline bst_float Predict(const FVec& feat, unsigned root_id = 0) const;
  /*!
   * \brief calculate the feature contributions for the given root
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \param out_contribs output vector to hold the contributions
   */
  inline void CalculateContributions(const RegTree::FVec& feat, unsigned root_id,
                                     bst_float *out_contribs) const;
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
  inline void FillNodeMeanValues();

 private:
  inline bst_float FillNodeMeanValue(int nid);

  std::vector<bst_float> node_mean_values;
};

// implementations of inline functions
// do not need to read if only use the model
inline void RegTree::FVec::Init(size_t size) {
  Entry e; e.flag = -1;
  data.resize(size);
  std::fill(data.begin(), data.end(), e);
}

inline void RegTree::FVec::Fill(const RowBatch::Inst& inst) {
  for (bst_uint i = 0; i < inst.length; ++i) {
    if (inst[i].index >= data.size()) continue;
    data[inst[i].index].fvalue = inst[i].fvalue;
  }
}

inline void RegTree::FVec::Drop(const RowBatch::Inst& inst) {
  for (bst_uint i = 0; i < inst.length; ++i) {
    if (inst[i].index >= data.size()) continue;
    data[inst[i].index].flag = -1;
  }
}

inline size_t RegTree::FVec::size() const {
  return data.size();
}

inline bst_float RegTree::FVec::fvalue(size_t i) const {
  return data[i].fvalue;
}

inline bool RegTree::FVec::is_missing(size_t i) const {
  return data[i].flag == -1;
}

inline int RegTree::GetLeafIndex(const RegTree::FVec& feat, unsigned root_id) const {
  int pid = static_cast<int>(root_id);
  while (!(*this)[pid].is_leaf()) {
    unsigned split_index = (*this)[pid].split_index();
    pid = this->GetNext(pid, feat.fvalue(split_index), feat.is_missing(split_index));
  }
  return pid;
}

inline bst_float RegTree::Predict(const RegTree::FVec& feat, unsigned root_id) const {
  int pid = this->GetLeafIndex(feat, root_id);
  return (*this)[pid].leaf_value();
}

inline void RegTree::FillNodeMeanValues() {
  size_t num_nodes = this->param.num_nodes;
  if (this->node_mean_values.size() == num_nodes) {
    return;
  }
  this->node_mean_values.resize(num_nodes);
  for (int root_id = 0; root_id < param.num_roots; ++root_id) {
    this->FillNodeMeanValue(root_id);
  }
}

inline bst_float RegTree::FillNodeMeanValue(int nid) {
  bst_float result;
  auto& node = (*this)[nid];
  if (node.is_leaf()) {
    result = node.leaf_value();
  } else {
    result  = this->FillNodeMeanValue(node.cleft()) * this->stat(node.cleft()).sum_hess;
    result += this->FillNodeMeanValue(node.cright()) * this->stat(node.cright()).sum_hess;
    result /= this->stat(nid).sum_hess;
  }
  this->node_mean_values[nid] = result;
  return result;
}

inline void RegTree::CalculateContributions(const RegTree::FVec& feat, unsigned root_id,
                                            bst_float *out_contribs) const {
  CHECK_GT(this->node_mean_values.size(), 0U);
  // this follows the idea of http://blog.datadive.net/interpreting-random-forests/
  bst_float node_value;
  unsigned split_index;
  int pid = static_cast<int>(root_id);
  // update bias value
  node_value = this->node_mean_values[pid];
  out_contribs[feat.size()] += node_value;
  if ((*this)[pid].is_leaf()) {
    // nothing to do anymore
    return;
  }
  while (!(*this)[pid].is_leaf()) {
    split_index = (*this)[pid].split_index();
    pid = this->GetNext(pid, feat.fvalue(split_index), feat.is_missing(split_index));
    bst_float new_value = this->node_mean_values[pid];
    // update feature weight
    out_contribs[split_index] += new_value - node_value;
    node_value = new_value;
  }
  bst_float leaf_value = (*this)[pid].leaf_value();
  // update leaf feature weight
  out_contribs[split_index] += leaf_value - node_value;
}

/*! \brief get next position of the tree given current pid */
inline int RegTree::GetNext(int pid, bst_float fvalue, bool is_unknown) const {
  bst_float split_value = (*this)[pid].split_cond();
  if (is_unknown) {
    return (*this)[pid].cdefault();
  } else {
    if (fvalue < split_value) {
      return (*this)[pid].cleft();
    } else {
      return (*this)[pid].cright();
    }
  }
}
}  // namespace xgboost
#endif  // XGBOOST_TREE_MODEL_H_
