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
#include "../../utils/xgboost_fmap.h"
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
                          const std::vector<unsigned> &root_index, 
                          const utils::FeatConstrain &constrain )
                : BaseTreeMaker( tree, param ), 
                  grad(grad), hess(hess), 
                  smat(smat), root_index(root_index), constrain(constrain) {
                utils::Assert( grad.size() == hess.size(), "booster:invalid input" );
                utils::Assert( smat.NumRow() == hess.size(), "booster:invalid input" );
                utils::Assert( root_index.size() == 0 || root_index.size() == hess.size(), "booster:invalid input" ); 
                {// setup temp space for each thread
                    if( param.nthread != 0 ){
                        omp_set_num_threads( param.nthread );
                    }
                    #pragma omp parallel
                    {
                        this->nthread = omp_get_num_threads();
                    }
                    tmp_rptr.resize( this->nthread, std::vector<size_t>() );
                    snode.reserve( 256 );
                }
            }
            inline void Make( int& stat_max_depth, int& stat_num_pruned ){
                this->InitData();
                this->InitNewNode( this->qexpand );
                stat_max_depth = 0;
                
                for( int depth = 0; depth < param.max_depth; ++ depth ){                                        
                    this->FindSplit( this->qexpand, depth );
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
            // expand a specific node
            inline bool Expand( const std::vector<bst_uint> &valid_index, int nid ){
                if( valid_index.size() == 0 ) return false;
                this->InitDataExpand( valid_index, nid );
                this->InitNewNode( this->qexpand );
                this->FindSplit( nid, tmp_rptr[0] );

                // update node statistics
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[i];
                    tree.stat( nid ).loss_chg = snode[ nid ].best.loss_chg;
                    tree.stat( nid ).sum_hess = static_cast<float>( snode[ nid ].sum_hess );
                }
                // change the leaf
                this->UpdateQueueExpand( this->qexpand );
                this->InitNewNode( this->qexpand );
                
                // set all the rest expanding nodes to leaf
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[i];
                    
                    tree[ nid ].set_leaf( snode[nid].weight * param.learning_rate );
                    tree.stat( nid ).loss_chg = 0.0f;
                    tree.stat( nid ).sum_hess = static_cast<float>( snode[ nid ].sum_hess );
                    tree.param.max_depth = std::max( tree.param.max_depth, tree.GetDepth( nid ) );
                }
                if( qexpand.size() != 0 ) {
                    return true;
                }else{
                    return false;
                }
            }
            // collapse specific node
            inline void Collapse( const std::vector<bst_uint> &valid_index, int nid ){
                if( valid_index.size() == 0 ) return;
                this->InitDataExpand( valid_index, nid );
                this->InitNewNode( this->qexpand );
                tree.stat( nid ).loss_chg = 0.0f;
                tree.stat( nid ).sum_hess = static_cast<float>( snode[ nid ].sum_hess );
                tree.CollapseToLeaf( nid, snode[nid].weight * param.learning_rate );
            }
        private:
            // make leaf nodes for all qexpand, update node statistics, mark leaf value
            inline void InitNewNode( const std::vector<int> &qexpand ){
                snode.resize( tree.param.num_nodes, NodeEntry() );

                for( size_t j = 0; j < qexpand.size(); ++j ){
                    const int nid = qexpand[ j ];
                    double sum_grad = 0.0, sum_hess = 0.0;

                    for( bst_uint i = node_bound[nid].first; i < node_bound[nid].second; ++i ){
                        const bst_uint ridx = row_index_set[i];
                        sum_grad += grad[ridx]; sum_hess += hess[ridx];
                    }
                    // update node statistics
                    snode[nid].sum_grad = sum_grad; 
                    snode[nid].sum_hess = sum_hess;

                    snode[nid].root_gain = param.CalcRootGain( sum_grad, sum_hess );
                    if( !tree[nid].is_root() ){
                        snode[nid].weight = param.CalcWeight( sum_grad, sum_hess, tree.stat( tree[nid].parent() ).base_weight );
                        tree.stat(nid).base_weight = snode[nid].weight;
                    }else{
                        snode[nid].weight = param.CalcWeight( sum_grad, sum_hess, 0.0f );
                        tree.stat(nid).base_weight = snode[nid].weight;
                    }
                }
            }
        private:
            // enumerate the split values of specific feature
            template<typename Iter>
            inline void EnumerateSplit( Iter it, SplitEntry &best, const int nid, const unsigned fid, bool is_forward_search ){
                float last_fvalue = 0.0f;
                double sum_hess = 0.0, sum_grad = 0.0;
                const NodeEntry enode = snode[ nid ];

                while( it.Next() ){
                    const bst_uint ridx = it.rindex();
                    const float fvalue = it.fvalue();           
                    
                    if( sum_hess == 0.0 ){
                        sum_grad = grad[ ridx ];
                        sum_hess = hess[ ridx ];
                        last_fvalue = fvalue;
                    }else{
                        // try to find a split
                        if( fabsf(fvalue - last_fvalue) > rt_2eps && sum_hess >= param.min_child_weight ){
                            const double csum_hess = enode.sum_hess - sum_hess;
                            if( csum_hess >= param.min_child_weight ){
                                const double csum_grad = enode.sum_grad - sum_grad; 
                                const double loss_chg = 
                                    + param.CalcGain(  sum_grad,  sum_hess, enode.weight ) 
                                    + param.CalcGain( csum_grad, csum_hess, enode.weight )
                                    - enode.root_gain;
                                best.Update( loss_chg, fid, (fvalue + last_fvalue) * 0.5f, !is_forward_search );
                            }else{
                                // the rest part doesn't meet split condition anyway, return 
                                return;
                            }
                        }
                        // update the statistics
                        sum_grad += grad[ ridx ];
                        sum_hess += hess[ ridx ];
                        last_fvalue = fvalue;
                    }                    
                }

                const double csum_hess = enode.sum_hess - sum_hess;
                if( sum_hess >= param.min_child_weight && csum_hess >= param.min_child_weight ){
                    const double csum_grad = enode.sum_grad - sum_grad; 
                    const double loss_chg = 
                        + param.CalcGain(   sum_grad,   sum_hess, enode.weight ) 
                        + param.CalcGain(  csum_grad,  csum_hess, enode.weight )
                        - snode[nid].root_gain;
                    const float delta = is_forward_search ? rt_eps:-rt_eps;
                    best.Update( loss_chg, fid, last_fvalue + delta, !is_forward_search );
                }
            }
        private:
            inline void FindSplit( const std::vector<int> &qexpand, int depth ){
                int nexpand = (int)qexpand.size();
                if( depth < 3 ){ 
                    for( int i = 0; i < nexpand; ++ i ){
                        this->FindSplit( qexpand[i], tmp_rptr[0] );
                    }
                }else{
                    // if get to enough depth, parallelize over node
                    #pragma omp parallel for schedule(dynamic,1)
                    for( int i = 0; i < nexpand; ++ i ){
                        const int tid = omp_get_thread_num();
                        utils::Assert( tid < (int)tmp_rptr.size(), "BUG: FindSplit, tid exceed tmp_rptr size" );
                        this->FindSplit( qexpand[i], tmp_rptr[tid] );
                    }
                }
            }
        private:
            inline void MakeSplit( int nid, unsigned gid ){
                node_bound.resize( tree.param.num_nodes );
                // re-organize the row_index_set after split on nid
                const unsigned split_index = tree[nid].split_index();
                const float    split_value = tree[nid].split_cond();

                std::vector<bst_uint> right;
                bst_uint top = node_bound[nid].first;
                for( bst_uint i = node_bound[ nid ].first; i < node_bound[ nid ].second; ++i ){
                    const bst_uint ridx = row_index_set[i];                    
                    bool goleft = tree[ nid ].default_left();                    
                    for( typename FMatrix::RowIter it = smat.GetRow(ridx,gid); it.Next(); ){
                        if( it.findex() == split_index ){
                            if( it.fvalue() < split_value ){
                                goleft = true;  break;
                            }else{
                                goleft = false; break;
                            }
                        }
                    }
                    if( goleft ) {
                        row_index_set[ top ++ ] = ridx;
                    }else{
                        right.push_back( ridx );
                    }
                }
                node_bound[ tree[nid].cleft() ]  = std::make_pair( node_bound[nid].first, top );
                node_bound[ tree[nid].cright() ] = std::make_pair( top, node_bound[nid].second );

                utils::Assert( node_bound[nid].second - top == (bst_uint)right.size(), "BUG:MakeSplit" );
                for( size_t i = 0; i < right.size(); ++ i ){
                    row_index_set[ top ++ ] = right[ i ];
                }
            }
                        
            // find splits at current level
            inline void FindSplit( int nid, std::vector<size_t> &tmp_rptr ){
                if( tmp_rptr.size() == 0 ){
                    tmp_rptr.resize( tree.param.num_feature + 1, 0 );
                }
                const bst_uint begin = node_bound[ nid ].first;
                const bst_uint end   = node_bound[ nid ].second;
                const unsigned ncgroup = smat.NumColGroup();
                unsigned best_group = 0;

                for( unsigned gid = 0; gid < ncgroup; ++gid ){
                    // records the columns
                    std::vector<FMatrixS::REntry> centry;
                    // records the active features
                    std::vector<size_t>  aclist;
                    utils::SparseCSRMBuilder<FMatrixS::REntry,true> builder( tmp_rptr, centry, aclist );
                    builder.InitBudget( tree.param.num_feature );
                    for( bst_uint i = begin; i < end; ++i ){
                        const bst_uint ridx = row_index_set[i];
                        for( typename FMatrix::RowIter it = smat.GetRow(ridx,gid); it.Next(); ){
                            const bst_uint findex = it.findex();
                            if( constrain.NotBanned( findex ) ) builder.AddBudget( findex );
                        }
                    }
                    builder.InitStorage();
                    for( bst_uint i = begin; i < end; ++i ){
                        const bst_uint ridx = row_index_set[i];
                        for( typename FMatrix::RowIter it = smat.GetRow(ridx,gid); it.Next(); ){
                            const bst_uint findex = it.findex();
                            if( constrain.NotBanned( findex ) ) {
                                builder.PushElem( findex, FMatrixS::REntry( ridx, it.fvalue() ) );
                            }
                        }
                    }
                    // --- end of building column major matrix ---                    
                    // after this point, tmp_rptr and entry is ready to use                    
                    int naclist = (int)aclist.size();
                    // best entry for each thread
                    SplitEntry nbest, tbest;
                    #pragma omp parallel private(tbest)
                    { 
                        #pragma omp for schedule(dynamic,1)
                        for( int j = 0; j < naclist; ++j ){
                            bst_uint findex = static_cast<bst_uint>( aclist[j] );
                            // local sort can be faster when the features are sparse
                            std::sort( centry.begin() + tmp_rptr[findex], centry.begin() + tmp_rptr[findex+1], FMatrixS::REntry::cmp_fvalue );
                            if( param.need_forward_search() ){
                                this->EnumerateSplit( FMatrixS::ColIter( &centry[tmp_rptr[findex]]-1, &centry[tmp_rptr[findex+1]] - 1 ),
                                                      tbest, nid, findex, true );
                            }
                            if( param.need_backward_search() ){
                                this->EnumerateSplit( FMatrixS::ColBackIter( &centry[tmp_rptr[findex+1]], &centry[tmp_rptr[findex]] ),
                                                      tbest, nid, findex, false );
                            }
                        }
                        #pragma omp critical 
                        {
                            nbest.Update( tbest );
                        }
                    }
                    // if current solution gives the best 
                    if( snode[nid].best.Update( nbest ) ){
                        best_group = gid;
                    }
                    // cleanup tmp_rptr for next usage
                    builder.Cleanup();    
                }
                
                // at this point, we already know the best split
                if( snode[nid].best.loss_chg > rt_eps ){
                    const SplitEntry &e = snode[nid].best;
                    tree.AddChilds( nid );
                    tree[ nid ].set_split( e.split_index(), e.split_value, e.default_left() );
                    this->MakeSplit( nid, best_group );
                }else{
                    tree[ nid ].set_leaf( snode[nid].weight * param.learning_rate );                    
                }
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

                {// expand query
                    qexpand.reserve( 256 ); qexpand.clear();
                    for( int i = 0; i < tree.param.num_roots; ++ i ){
                        qexpand.push_back( i );
                    }
                }
            }

            // initialize temp data structure
            inline void InitDataExpand( const std::vector<bst_uint> &valid_index, int nid ){
                row_index_set = valid_index;                
                node_bound.resize( tree.param.num_nodes );
                node_bound[ nid ] = std::make_pair( 0, (bst_uint)row_index_set.size() );
             
                qexpand.clear(); qexpand.push_back( nid );
            }
        private:
            // number of omp thread used during training
            int nthread;
            // tmp row pointer, per thread, used for tmp data construction
            std::vector< std::vector<size_t> > tmp_rptr;
            // Instance row indexes corresponding to each node
            std::vector<bst_uint> row_index_set;
            // lower and upper bound of each nodes' row_index
            std::vector< std::pair<bst_uint, bst_uint> > node_bound;
        private:
            const std::vector<float> &grad;
            const std::vector<float> &hess;
            const FMatrix            &smat;
            const std::vector<unsigned> &root_index;
            const utils::FeatConstrain  &constrain;
        };
    };
};
#endif
