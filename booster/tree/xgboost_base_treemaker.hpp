#ifndef XGBOOST_BASE_TREEMAKER_HPP
#define XGBOOST_BASE_TREEMAKER_HPP
/*!
 * \file xgboost_base_treemaker.hpp
 * \brief implementation of base data structure for regression tree maker,
 *         gives common operations of tree construction steps template 
 * 
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
#include <vector>
#include "xgboost_tree_model.h"

namespace xgboost{
    namespace booster{
        class BaseTreeMaker{
        protected:
            BaseTreeMaker( RegTree &tree,
                           const TreeParamTrain &param )
                : tree( tree ), param( param ){}
        protected:
            // statistics that is helpful to decide a split
            struct SplitEntry{
                /*! \brief loss change after split this node */
                float  loss_chg;
                /*! \brief split index */
                unsigned  sindex;
                /*! \brief split value */
                float     split_value;
                /*! \brief constructor */
                SplitEntry( void ){
                    loss_chg = 0.0f;
                    split_value = 0.0f; sindex = 0;
                }
                // This function gives better priority to lower index when loss_chg equals
                // not the best way, but helps to give consistent result during multi-thread execution
                inline bool NeedReplace( float loss_chg, unsigned split_index ) const{
                    if( this->split_index() <= split_index ){
                        return loss_chg > this->loss_chg; 
                    }else{
                        return !(this->loss_chg > loss_chg);
                    }
                }
                inline bool Update( const SplitEntry &e ){
                    if( this->NeedReplace( e.loss_chg, e.split_index() ) ){
                        this->loss_chg = e.loss_chg;
                        this->sindex = e.sindex;
                        this->split_value = e.split_value;
                        return true;
                    } else{
                        return false;
                    }
                }
                inline bool Update( float loss_chg, unsigned split_index, float split_value, bool default_left ){                    
                    if( this->NeedReplace( loss_chg, split_index ) ){
                        this->loss_chg = loss_chg;
                        if( default_left ) split_index |= (1U << 31);
                        this->sindex = split_index;
                        this->split_value = split_value;
                        return true;
                    }else{
                        return false;
                    }
                }
                inline unsigned split_index( void ) const{
                    return sindex & ( (1U<<31) - 1U );
                }
                inline bool default_left( void ) const{
                    return (sindex >> 31) != 0;
                }
            };
            struct NodeEntry{
                /*! \brief sum gradient statistics */
                double sum_grad;
                /*! \brief sum hessian statistics */
                double sum_hess;
                /*! \brief loss of this node, without split */
                float  root_gain;
                /*! \brief weight calculated related to current data */
                float  weight;
                /*! \brief current best solution */
                SplitEntry best;
                NodeEntry( void ){
                    sum_grad = sum_hess = 0.0;
                    weight = root_gain = 0.0f;
                }
            };
        private:
            // try to prune off current leaf, return true if successful
            inline void TryPruneLeaf( int nid, int depth ){
                if( tree[ nid ].is_root() ) return;
                int pid = tree[ nid ].parent();
                RegTree::NodeStat &s = tree.stat( pid );
                ++ s.leaf_child_cnt;
                
                if( s.leaf_child_cnt >= 2 && param.need_prune( s.loss_chg, depth - 1 ) ){
                    this->stat_num_pruned += 2;
                    // need to be pruned
                    tree.ChangeToLeaf( pid, param.learning_rate * s.base_weight );
                    // tail recursion
                    this->TryPruneLeaf( pid, depth - 1 );
                }
            }
        protected:
            /*! \brief do prunning of a tree */
            inline int DoPrune( void ){
                this->stat_num_pruned = 0;
                // initialize auxiliary statistics
                for( int nid = 0; nid < tree.param.num_nodes; ++ nid ){
                    tree.stat( nid ).leaf_child_cnt = 0;
                    tree.stat( nid ).loss_chg = snode[ nid ].best.loss_chg;
                    tree.stat( nid ).sum_hess = static_cast<float>( snode[ nid ].sum_hess );
                }
                for( int nid = 0; nid < tree.param.num_nodes; ++ nid ){
                    if( tree[ nid ].is_leaf() ) this->TryPruneLeaf( nid, tree.GetDepth(nid) );
                }
                return this->stat_num_pruned;
            }
        protected:
            /*! \brief update queue expand add in new leaves */
            inline void UpdateQueueExpand( std::vector<int> &qexpand ){
                std::vector<int> newnodes;
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[i];
                    if( !tree[ nid ].is_leaf() ){
                        newnodes.push_back( tree[nid].cleft() );
                        newnodes.push_back( tree[nid].cright() );
                    }
                }
                // use new nodes for qexpand
                qexpand = newnodes;
            }
        protected:
            // local helper tmp data structure
            // statistics
            int stat_num_pruned;
            /*! \brief queue of nodes to be expanded */
            std::vector<int> qexpand;
            /*! \brief TreeNode Data: statistics for each constructed node, the derived class must maintain this */
            std::vector<NodeEntry> snode;
        protected:
            // original data that supports tree construction
            RegTree &tree;
            const TreeParamTrain &param;
        };
    }; // namespace booster
}; // namespace xgboost
#endif // XGBOOST_BASE_TREEMAKER_HPP
