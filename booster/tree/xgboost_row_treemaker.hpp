#ifndef XGBOOST_ROW_TREEMAKER_HPP
#define XGBOOST_ROW_TREEMAKER_HPP
/*!
 * \file xgboost_row_treemaker.hpp
 * \brief implementation of regression tree maker,
 *        use a row based approach
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
// use openmp
#include <vector>
#include "xgboost_tree_model.h"
#include "../../utils/xgboost_omp.h"
#include "../../utils/xgboost_random.h"
#include "xgboost_base_treemaker.hpp"

namespace xgboost{
    namespace booster{
        template<typename FMatrix>
        class RowTreeMaker : protected BaseTreeMaker{
        public:
            RowTreeMaker( RegTree &tree,
                          const TreeParamTrain &param, 
                          const std::vector<float> &grad,
                          const std::vector<float> &hess,
                          const FMatrix &smat, 
                          const std::vector<unsigned> &root_index )
                : BaseTreeMaker( tree, param ), 
                  grad(grad), hess(hess), 
                  smat(smat), root_index(root_index) {
                utils::Assert( grad.size() == hess.size(), "booster:invalid input" );
                utils::Assert( smat.NumRow() == hess.size(), "booster:invalid input" );
                utils::Assert( root_index.size() == 0 || root_index.size() == hess.size(), "booster:invalid input" ); 
            }
            inline void Make( int& stat_max_depth, int& stat_num_pruned ){
                this->InitData();
                this->InitNewNode( this->qexpand );
                stat_max_depth = 0;
                
                for( int depth = 0; depth < param.max_depth; ++ depth ){
                    //this->FindSplit( this->qexpand );
                    this->UpdateQueueExpand( this->qexpand );
                    this->InitNewNode( this->qexpand );
                    // if nothing left to be expand, break
                    if( qexpand.size() == 0 ) break;
                    stat_max_depth = depth + 1;
                }
                // set all the rest expanding nodes to leaf
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[i];
                    tree[ nid ].set_leaf( snode[nid].weight * param.learning_rate );
                }
                // start prunning the tree
                stat_num_pruned = this->DoPrune();
            }
        private:
            // make leaf nodes for all qexpand, update node statistics, mark leaf value
            inline void InitNewNode( const std::vector<int> &qexpand ){
                snode.resize( tree.param.num_nodes, NodeEntry() );

                for( size_t j = 0; j < qexpand.size(); ++ j ){
                    const int nid = qexpand[ j ];
                    double sum_grad = 0.0, sum_hess = 0.0;
                    // TODO: get sum statistics for nid

                    // update node statistics
                    snode[nid].sum_grad = sum_grad; 
                    snode[nid].sum_hess = sum_hess;
                    snode[nid].root_gain = param.CalcRootGain( sum_grad, sum_hess );
                    if( !tree[nid].is_root() ){
                        snode[nid].weight = param.CalcWeight( sum_grad, sum_hess, snode[ tree[nid].parent() ].weight );
                    }else{
                        snode[nid].weight = param.CalcWeight( sum_grad, sum_hess, 0.0f );
                    }
                }
            }
            // find splits at current level
            inline void FindSplit( int nid ){
                // TODO

            }
        private:
            // initialize temp data structure
            inline void InitData( void ){
                std::vector<bst_uint> valid_index;
                for( size_t i = 0; i < grad.size(); ++i ){
                    if( hess[ i ] < 0.0f ) continue;
                    if( param.subsample > 1.0f-1e-6f || random::SampleBinary( param.subsample ) != 0 ){
                        valid_index.push_back( static_cast<bst_uint>(i) );
                    }
                }
                node_bound.resize( tree.param.num_roots );

                if( root_index.size() == 0 ){
                    row_index_set = valid_index;
                    // set bound of root node
                    node_bound[0] = std::make_pair( 0, (bst_uint)row_index_set.size() );
                }else{                    
                    std::vector<size_t>   rptr;
                    utils::SparseCSRMBuilder<bst_uint> builder( rptr, row_index_set );
                    builder.InitBudget( tree.param.num_roots );
                    for( size_t i = 0; i < valid_index.size(); ++i ){
                        const bst_uint rid = valid_index[ i ];
                        utils::Assert( root_index[ rid ] < (unsigned)tree.param.num_roots, "root id exceed number of roots" );
                        builder.AddBudget( root_index[ rid ] );
                    }
                    builder.InitStorage();
                    for( size_t i = 0; i < valid_index.size(); ++i ){
                        const bst_uint rid = valid_index[ i ];
                        builder.PushElem( root_index[ rid ], rid );
                    }
                    for( size_t i = 1; i < rptr.size(); ++ i ){
                        node_bound[i-1] = std::make_pair( rptr[ i - 1 ], rptr[ i ] );
                    }
                }
                
                {// setup temp space for each thread
                    if( param.nthread != 0 ){
                        omp_set_num_threads( param.nthread );
                    }
                    #pragma omp parallel
                    {
                        this->nthread = omp_get_num_threads();
                    }
                    snode.reserve( 256 );
                }
                
                {// expand query
                    qexpand.reserve( 256 ); qexpand.clear();
                    for( int i = 0; i < tree.param.num_roots; ++ i ){
                        qexpand.push_back( i );
                    }
                }
            }
        private:
            // number of omp thread used during training
            int nthread;
            // Instance row indexes corresponding to each node
            std::vector<bst_uint> row_index_set;
            // lower and upper bound of each nodes' row_index
            std::vector< std::pair<bst_uint, bst_uint> > node_bound;
        private:
            const std::vector<float> &grad;
            const std::vector<float> &hess;
            const FMatrix            &smat;
            const std::vector<unsigned> &root_index;
        };
    };
};
#endif
