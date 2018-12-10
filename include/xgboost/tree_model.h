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
#include "./feature_map.h"

namespace xgboost {

// Forward declarations
struct PathElement;
struct DenseFeatureVector;

/*! \brief meta parameters of the tree */
struct TreeParam : public dmlc::Parameter<TreeParam> {
  /*! \brief number of start root */
  int num_roots{1};  // DEPRECATED - always 1
  /*! \brief total number of nodes */
  int num_nodes;
  /*!\brief number of deleted nodes */
  int num_deleted;
  /*! \brief maximum depth, this is a statistics of the tree */
  int max_depth;
  /*! \brief number of features used for tree construction */
  int num_feature;  // DEPRECATED
  /*!
   * \brief leaf vector size, used for vector tree
   * used to store more than one dimensional information in tree
   */
  int size_leaf_vector;  // DEPRECATED
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
  DMLC_DECLARE_PARAMETER(TreeParam) {}
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

class RegressionTree {
 public:
  /*! \brief auxiliary statistics of node to help tree building */
  using SplitCondT = float;
  /*! \brief tree node */
  class Node {
   public:
    Node()  {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info),
                    "Node: 64 bit align");
    }
    /*! \brief index of left child */
    inline int LeftChild() const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    inline int RightChild() const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int DefaultChild() const {
      return this->DefaultLeft() ? this->LeftChild() : this->RightChild();
    }
    /*! \brief feature index of split condition */
    inline unsigned SplitIndex() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool DefaultLeft() const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    inline bool IsLeaf() const {
      return cleft_ == -1;
    }
    /*! \return get leaf value of leaf node */
    inline bst_float LeafValue() const {
      return (this->info_).leaf_value;
    }
    /*! \return get split condition of the node */
    inline SplitCondT SplitCond() const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    inline int Parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    inline bool IsLeftChild() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether this node is deleted */
    inline bool IsDeleted() const {
      return sindex_ == std::numeric_limits<unsigned>::max();
    }
    /*! \brief whether current node is root */
    inline bool IsRoot() const {
      return parent_ == -1;
    }
    /*!
     * \brief set the right child
     * \param nid node id to right child
     */
    inline void SetRightChild(int nid) {
      this->cright_ = nid;
    }
    /*!
     * \brief set the left child
     * \param nid node id to leftchild
     */
    inline void SetLeftChild(int nid) {
      this->cleft_ = nid;
    }
    /*!
     * \brief set split condition of current node
     * \param split_index feature index to split
     * \param split_cond  split condition
     * \param default_left the default direction when feature is unknown
     */
    inline void SetSplit(unsigned split_index, SplitCondT split_cond,
                          bool default_left = false) {
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).split_cond = split_cond;
    }
    // set parent
    inline void SetParent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     * \param right right index, could be used to store
     *        additional information
     */
    inline void SetLeaf(bst_float value, int right = -1) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = right;
    }
    /*! \brief mark that this node is deleted */
    inline void MarkDelete() {
      this->sindex_ = std::numeric_limits<unsigned>::max();
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

 private:
  // vector of nodes
  std::vector<Node> nodes_;
  // stats of nodes
  std::vector<RTreeNodeStat> stats_;

  std::vector<bst_float>
      node_mean_values_;  // Cache mean values for generating contributions
  // allocate a new node,
  // !!!!!! NOTE: may cause BUG here, nodes.resize
  inline int AllocNode() {
    int nd = param.num_nodes++;
    CHECK_LT(param.num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    return nd;
  }
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
  void TreeShap(const DenseFeatureVector& feat, bst_float *phi,
                       unsigned node_index, unsigned unique_depth,
                       PathElement *parent_unique_path, bst_float parent_zero_fraction,
                       bst_float parent_one_fraction, int parent_feature_index,
                       int condition, unsigned condition_feature,
                       bst_float condition_fraction) const;
  // delete a tree node, keep the parent field to allow trace back
  inline void DeleteNode(int nid) {
    CHECK_GE(nid, param.num_roots);
    nodes_[nid].MarkDelete();
    ++param.num_deleted;
  }

  bst_float FillNodeMeanValue(int nid);

 public:
  /*! \brief model parameter */
  TreeParam param;
  /*! \brief constructor */
  RegressionTree() {
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
  /*!
   * \brief change a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param value new leaf value
   */
  inline void ChangeToLeaf(int rid, bst_float value) {
    CHECK(nodes_[nodes_[rid].LeftChild()].IsLeaf());
    CHECK(nodes_[nodes_[rid].RightChild()].IsLeaf());
    this->DeleteNode(nodes_[rid].LeftChild());
    this->DeleteNode(nodes_[rid].RightChild());
    nodes_[rid].SetLeaf(value);
  }

  /*! \brief get node given nid */
  inline const Node& GetNode(int nid) const {
    return nodes_[nid];
  }

  /*! \brief get node given nid */
  inline Node& GetNode(int nid) {
    return nodes_[nid];
  }

  /*! \brief get const reference to nodes */
  inline const std::vector<Node>& GetNodes() const { return nodes_; }

  /*! \brief get node statistics given nid */
  inline RTreeNodeStat& Stat(int nid) {
    return stats_[nid];
  }
  /*! \brief get node statistics given nid */
  inline const RTreeNodeStat& Stat(int nid) const {
    return stats_[nid];
  }
  /*!
   * \brief load model from stream
   * \param fi input stream
   */
  inline void Load(dmlc::Stream* fi) {
    CHECK_EQ(fi->Read(&param, sizeof(TreeParam)), sizeof(TreeParam));
    nodes_.resize(param.num_nodes);
    stats_.resize(param.num_nodes);
    CHECK_NE(param.num_nodes, 0);
    CHECK_EQ(fi->Read(dmlc::BeginPtr(nodes_), sizeof(Node) * nodes_.size()),
             sizeof(Node) * nodes_.size());
    CHECK_EQ(fi->Read(dmlc::BeginPtr(stats_), sizeof(RTreeNodeStat) * stats_.size()),
             sizeof(RTreeNodeStat) * stats_.size());
  }
  /*!
   * \brief save model to stream
   * \param fo output stream
   */
  inline void Save(dmlc::Stream* fo) const {
    CHECK_EQ(param.num_nodes, static_cast<int>(nodes_.size()));
    CHECK_EQ(param.num_nodes, static_cast<int>(stats_.size()));
    fo->Write(&param, sizeof(TreeParam));
    CHECK_NE(param.num_nodes, 0);
    fo->Write(dmlc::BeginPtr(nodes_), sizeof(Node) * nodes_.size());
    fo->Write(dmlc::BeginPtr(stats_), sizeof(RTreeNodeStat) * nodes_.size());
  }
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid) {
    int pleft  = this->AllocNode();
    int pright = this->AllocNode();
    nodes_[nid].SetLeftChild(pleft);
    nodes_[nid].SetRightChild(pright);
    nodes_[nodes_[nid].LeftChild() ].SetParent(nid, true);
    nodes_[nodes_[nid].RightChild()].SetParent(nid, false);
  }
  /*!
   * \brief get current depth
   * \param nid node id
   */
  inline int GetDepth(int nid) const {
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
  inline int MaxDepth(int nid = 0) const {
    if (nodes_[nid].IsLeaf()) return 0;
    return std::max(MaxDepth(nodes_[nid].LeftChild())+1,
                     MaxDepth(nodes_[nid].RightChild())+1);
  }
  /*! \brief number of extra nodes besides the root */
  inline int NumExtraNodes() const {
    return param.num_nodes - param.num_roots - param.num_deleted;
  }
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
   * \brief get the leaf index
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \return the leaf index of the given feature
   */
  int GetLeafIndex(const DenseFeatureVector& feat, unsigned root_id = 0) const;

  /*!
   * \brief get next position of the tree given current pid
   * \param pid Current node id.
   * \param fvalue feature value if not missing.
   * \param is_unknown Whether current required feature is missing.
   */
  int GetNext(int pid, bst_float fvalue, bool is_unknown) const;
  /*!
   * \brief Get the mean value for node, required for feature contributions
   */
  float GetNodeMeanValue(int nid);
  /*!
   * \brief calculate the feature contributions (https://arxiv.org/abs/1706.06060) for the tree
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \param out_contribs output vector to hold the contributions
   * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
   * \param condition_feature the index of the feature to fix
   */
  void CalculateContributions(const DenseFeatureVector& feat,
                                     unsigned root_id, bst_float* out_contribs,
                                     int condition = 0,
                                     unsigned condition_feature = 0);
  /*!
   * \brief calculate the approximate feature contributions for the given root
   * \param feat dense feature vector, if the feature is missing the field is set to NaN
   * \param root_id starting root index of the instance
   * \param out_contribs output vector to hold the contributions
   */
  void CalculateContributionsApprox(const DenseFeatureVector& feat,
                                    unsigned root_id, bst_float* out_contribs);
};

/** \brief Dense feature vector, used for efficient lookup of sparse data for prediction */
struct DenseFeatureVector {
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

}  // namespace xgboost
#endif  // XGBOOST_TREE_MODEL_H_
