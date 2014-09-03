#ifndef XGBOOST_TREE_MODEL_H_
#define XGBOOST_TREE_MODEL_H_
/*!
 * \file model.h
 * \brief model structure for tree
 * \author Tianqi Chen
 */
#include <string>
#include <cstring>
#include <sstream>
#include <limits>
#include <algorithm>
#include <vector>
#include <cmath>
#include "../utils/io.h"
#include "../utils/fmap.h"
#include "../utils/utils.h"

namespace xgboost {
namespace tree {
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
  /*! \brief parameters of the tree */
  struct Param{
    /*! \brief number of start root */
    int num_roots;
    /*! \brief total number of nodes */
    int num_nodes;
    /*!\brief number of deleted nodes */
    int num_deleted;
    /*! \brief maximum depth, this is a statistics of the tree */
    int max_depth;
    /*! \brief  number of features used for tree construction */
    int num_feature;
    /*! 
     * \brief leaf vector size, used for vector tree
     * used to store more than one dimensional information in tree
     */
    int size_leaf_vector;
    /*! \brief reserved part */
    int reserved[31];
    /*! \brief constructor */
    Param(void) {
      max_depth = 0;
      size_leaf_vector = 0;
      std::memset(reserved, 0, sizeof(reserved));
    }
    /*! 
     * \brief set parameters from outside 
     * \param name name of the parameter
     * \param val  value of the parameter
     */
    inline void SetParam(const char *name, const char *val) {
      using namespace std;
      if (!strcmp("num_roots", name)) num_roots = atoi(val);
      if (!strcmp("num_feature", name)) num_feature = atoi(val);
      if (!strcmp("size_leaf_vector", name)) size_leaf_vector = atoi(val);
    }
  };
  /*! \brief tree node */
  class Node{
   public:
    /*! \brief index of left child */
    inline int cleft(void) const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    inline int cright(void) const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int cdefault(void) const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    /*! \brief feature index of split condition */
    inline unsigned split_index(void) const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool default_left(void) const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    inline bool is_leaf(void) const {
      return cleft_ == -1;
    }
    /*! \brief get leaf value of leaf node */
    inline float leaf_value(void) const {
      return (this->info_).leaf_value;
    }
    /*! \brief get split condition of the node */
    inline TSplitCond split_cond(void) const {
      return (this->info_).split_cond;
    }
    /*! \brief get parent of the node */
    inline int parent(void) const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    inline bool is_left_child(void) const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether current node is root */
    inline bool is_root(void) const {
      return parent_ == -1;
    }
    /*! 
     * \brief set the right child 
     * \param nide node id to right child
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
    inline void set_leaf(float value, int right = -1) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = right;
    }

   private:
    friend class TreeModel<TSplitCond, TNodeStat>;
    /*! 
     * \brief in leaf node, we have weights, in non-leaf nodes, 
     *        we have split condition 
     */
    union Info{
      float leaf_value;
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
  inline int AllocNode(void) {
    if (param.num_deleted != 0) {
      int nd = deleted_nodes.back();
      deleted_nodes.pop_back();
      --param.num_deleted;
      return nd;
    }
    int nd = param.num_nodes++;
    utils::Check(param.num_nodes < std::numeric_limits<int>::max(),
                 "number of nodes in the tree exceed 2^31");
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    leaf_vector.resize(param.num_nodes * param.size_leaf_vector); 
    return nd;
  }
  // delete a tree node
  inline void DeleteNode(int nid) {
    utils::Assert(nid >= param.num_roots, "can not delete root");
    deleted_nodes.push_back(nid);
    nodes[nid].set_parent(-1);
    ++param.num_deleted;
  }

 public:
  /*! 
   * \brief change a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param new leaf value
   */
  inline void ChangeToLeaf(int rid, float value) {
    utils::Assert(nodes[nodes[rid].cleft() ].is_leaf(),
                  "can not delete a non termial child");
    utils::Assert(nodes[nodes[rid].cright()].is_leaf(),
                  "can not delete a non termial child");
    this->DeleteNode(nodes[rid].cleft());
    this->DeleteNode(nodes[rid].cright());
    nodes[rid].set_leaf(value);
  }
  /*! 
   * \brief collapse a non leaf node to a leaf node, delete its children
   * \param rid node id of the node
   * \param new leaf value
   */
  inline void CollapseToLeaf(int rid, float value) {
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
  Param param;
  /*! \brief constructor */
  TreeModel(void) {
    param.num_nodes = 1;
    param.num_roots = 1;
    param.num_deleted = 0;
    nodes.resize(1);
  }
  /*! \brief get node given nid */
  inline Node &operator[](int nid) {
    return nodes[nid];
  }
  /*! \brief get node given nid */
  inline const Node &operator[](int nid) const {
    return nodes[nid];
  }
  /*! \brief get node statistics given nid */
  inline NodeStat &stat(int nid) {
    return stats[nid];
  }
  /*! \brief get leaf vector given nid */
  inline bst_float* leafvec(int nid) {
    if (leaf_vector.size() == 0) return NULL;
    return &leaf_vector[nid * param.size_leaf_vector];
  }
  /*! \brief get leaf vector given nid */
  inline const bst_float* leafvec(int nid) const{
    if (leaf_vector.size() == 0) return NULL;
    return &leaf_vector[nid * param.size_leaf_vector];
  }
  /*! \brief initialize the model */
  inline void InitModel(void) {
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
  inline void LoadModel(utils::IStream &fi) {
    utils::Check(fi.Read(&param, sizeof(Param)) > 0,
                 "TreeModel: wrong format");
    nodes.resize(param.num_nodes); stats.resize(param.num_nodes);
    utils::Check(fi.Read(&nodes[0], sizeof(Node) * nodes.size()) > 0,
                 "TreeModel: wrong format");
    utils::Check(fi.Read(&stats[0], sizeof(NodeStat) * stats.size()) > 0,
                 "TreeModel: wrong format");
    if (param.size_leaf_vector != 0) {
      utils::Check(fi.Read(&leaf_vector), "TreeModel: wrong format");
    }
    // chg deleted nodes
    deleted_nodes.resize(0);
    for (int i = param.num_roots; i < param.num_nodes; i ++) {
      if (nodes[i].is_root()) deleted_nodes.push_back(i);
    }
    utils::Assert(static_cast<int>(deleted_nodes.size()) == param.num_deleted,
                  "number of deleted nodes do not match");
  }
  /*! 
   * \brief save model to stream
   * \param fo output stream
   */
  inline void SaveModel(utils::IStream &fo) const {
    utils::Assert(param.num_nodes == static_cast<int>(nodes.size()),
                  "Tree::SaveModel");
    utils::Assert(param.num_nodes == static_cast<int>(stats.size()),
                  "Tree::SaveModel");
    fo.Write(&param, sizeof(Param));
    fo.Write(&nodes[0], sizeof(Node) * nodes.size());
    fo.Write(&stats[0], sizeof(NodeStat) * nodes.size());
    if (param.size_leaf_vector != 0) fo.Write(leaf_vector);
  }
  /*! 
   * \brief add child nodes to node
   * \param nid node id to add childs
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
   * \param node id to add right child
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
  inline int MaxDepth(void) {
    int maxd = 0;
    for (int i = 0; i < param.num_roots; ++i) {
      maxd = std::max(maxd, MaxDepth(i));
    }
    return maxd;
  }
  /*! \brief number of extra nodes besides the root */
  inline int num_extra_nodes(void) const {
    return param.num_nodes - param.num_roots - param.num_deleted;
  }
  /*! 
   * \brief dump model to text string
   * \param fmap feature map of feature types
   * \param with_stats whether dump out statistics as well
   * \return the string of dumped model
   */
  inline std::string DumpModel(const utils::FeatMap& fmap, bool with_stats) {
    std::stringstream fo("");
    for (int i = 0; i < param.num_roots; ++i) {
      this->Dump(i, fo, fmap, 0, with_stats);
    }
    return fo.str();
  }

 private:
  void Dump(int nid, std::stringstream &fo,
            const utils::FeatMap& fmap, int depth, bool with_stats) {
    for (int i = 0;  i < depth; ++i) {
      fo << '\t';
    }
    if (nodes[nid].is_leaf()) {
      fo << nid << ":leaf=" << nodes[nid].leaf_value();
      if (with_stats) {
        stat(nid).Print(fo, true);
      }
      fo << '\n';
    } else {
      // right then left,
      TSplitCond cond = nodes[nid].split_cond();
      const unsigned split_index = nodes[nid].split_index();
      if (split_index < fmap.size()) {
        switch (fmap.type(split_index)) {
          case utils::FeatMap::kIndicator: {
            int nyes = nodes[nid].default_left() ?
                nodes[nid].cright() : nodes[nid].cleft();
            fo << nid << ":[" << fmap.name(split_index) << "] yes=" << nyes
               << ",no=" << nodes[nid].cdefault();
            break;
          }
          case utils::FeatMap::kInteger: {
            fo << nid << ":[" << fmap.name(split_index) << "<"
               << int(float(cond)+1.0f)
               << "] yes=" << nodes[nid].cleft()
               << ",no=" << nodes[nid].cright()
               << ",missing=" << nodes[nid].cdefault();
            break;
          }
          case utils::FeatMap::kFloat:
          case utils::FeatMap::kQuantitive: {
            fo << nid << ":[" << fmap.name(split_index) << "<"<< float(cond)
               << "] yes=" << nodes[nid].cleft()
               << ",no=" << nodes[nid].cright()
               << ",missing=" << nodes[nid].cdefault();
            break;
          }
          default: utils::Error("unknown fmap type");
        }
      } else {
        fo << nid << ":[f" << split_index << "<"<< float(cond)
           << "] yes=" << nodes[nid].cleft()
           << ",no=" << nodes[nid].cright()
           << ",missing=" << nodes[nid].cdefault();
      }
      if (with_stats) {
        fo << ' ';
        stat(nid).Print(fo, false);
      }
      fo << '\n';
      this->Dump(nodes[nid].cleft(), fo, fmap, depth+1, with_stats);
      this->Dump(nodes[nid].cright(), fo, fmap, depth+1, with_stats);
    }
  }
};

/*! \brief node statistics used in regression tree */
struct RTreeNodeStat {
  /*! \brief loss chg caused by current split */
  float loss_chg;
  /*! \brief sum of hessian values, used to measure coverage of data */
  float sum_hess;
  /*! \brief weight of current node */
  float base_weight;
  /*! \brief number of child that is leaf node known up to now */
  int   leaf_child_cnt;
  /*! \brief print information of current stats to fo */
  inline void Print(std::stringstream &fo, bool is_leaf) const {
    if (!is_leaf) {
      fo << "gain=" << loss_chg << ",cover=" << sum_hess;
    } else {
      fo << "cover=" << sum_hess;
    }
  }
};

/*! \brief define regression tree to be the most common tree model */
class RegTree: public TreeModel<bst_float, RTreeNodeStat>{
 public:
  /*! 
   * \brief dense feature vector that can be taken by RegTree
   * to do tranverse efficiently
   * and can be construct from sparse feature vector
   */
  struct FVec {
    /*! 
     * \brief a union value of value and flag
     * when flag == -1, this indicate the value is missing
     */
    union Entry{
      float fvalue;
      int flag;
    };
    std::vector<Entry> data;
    /*! \brief intialize the vector with size vector */
    inline void Init(size_t size) {
      Entry e; e.flag = -1;
      data.resize(size);
      std::fill(data.begin(), data.end(), e);
    }
    /*! \brief fill the vector with sparse vector */
    inline void Fill(const RowBatch::Inst &inst) {
      for (bst_uint i = 0; i < inst.length; ++i) {
        data[inst[i].index].fvalue = inst[i].fvalue;
      }
    }
    /*! \brief drop the trace after fill, must be called after fill */
    inline void Drop(const RowBatch::Inst &inst) {      
      for (bst_uint i = 0; i < inst.length; ++i) {
        data[inst[i].index].flag = -1;
      }
    }
    /*! \brief get ith value */
    inline float fvalue(size_t i) const {
      return data[i].fvalue;
    }
    /*! \brief check whether i-th entry is missing */
    inline bool is_missing(size_t i) const {
      return data[i].flag == -1;
    }
  };
  /*!
   * \brief get the leaf index 
   * \param feats dense feature vector, if the feature is missing the field is set to NaN
   * \param root_gid starting root index of the instance
   * \return the leaf index of the given feature 
   */
  inline int GetLeafIndex(const FVec&feat, unsigned root_id = 0) const {
    // start from groups that belongs to current data
    int pid = static_cast<int>(root_id);
    // tranverse tree
    while (!(*this)[ pid ].is_leaf()) {
      unsigned split_index = (*this)[pid].split_index();
      pid = this->GetNext(pid, feat.fvalue(split_index), feat.is_missing(split_index));
    }
    return pid;
  }
  /*!
   * \brief get the prediction of regression tree, only accepts dense feature vector
   * \param feats dense feature vector, if the feature is missing the field is set to NaN
   * \param root_gid starting root index of the instance
   * \return the leaf index of the given feature 
   */
  inline float Predict(const FVec &feat, unsigned root_id = 0) const {
    int pid = this->GetLeafIndex(feat, root_id);
    return (*this)[pid].leaf_value();
  }
  /*! \brief get next position of the tree given current pid */
  inline int GetNext(int pid, float fvalue, bool is_unknown) const {
    float split_value = (*this)[pid].split_cond();
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
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_MODEL_H_
