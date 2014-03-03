#ifndef XGBOOST_COL_TREEMAKER_HPP
#define XGBOOST_COL_TREEMAKER_HPP
/*!
 * \file xgboost_col_treemaker.hpp
 * \brief implementation of regression tree maker,
 *        use a column based approach, with OpenMP 
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
// use openmp
#include <omp.h>
#include <vector>
#include "xgboost_tree_model.h"
#include "../../utils/xgboost_random.h"

namespace xgboost{
    namespace booster{
        template<typename FMatrix>
        class ColTreeMaker{
        public:
            ColTreeMaker( RegTree &tree,
                          const TreeParamTrain &param, 
                          const std::vector<float> &grad,
                          const std::vector<float> &hess,
                          const FMatrix &smat, 
                          const std::vector<unsigned> &root_index ):
                tree( tree ), param( param ), grad( grad ), hess( hess ),
                smat( smat ), root_index( root_index ){
                utils::Assert( grad.size() == hess.size(), "booster:invalid input" );
                utils::Assert( smat.NumRow() == hess.size(), "booster:invalid input" );
                utils::Assert( root_index.size() == 0 || root_index.size() == hess.size(), "booster:invalid input" );
                utils::Assert( smat.HaveColAccess(), "ColTreeMaker: need column access matrix" );
            }
            inline void Make( int& stat_max_depth, int& stat_num_pruned ){
                this->InitData();
                this->InitNewNode( this->qexpand );
                stat_max_depth = 0;
                
                for( int depth = 0; depth < param.max_depth; ++ depth ){
                    this->FindSplit( depth );
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
                inline void Update( const SplitEntry &e ){
                    if( this->NeedReplace( e.loss_chg, e.split_index() ) ){
                        this->loss_chg = e.loss_chg;
                        this->sindex = e.sindex;
                        this->split_value = e.split_value;
                    } 
                }
                inline void Update( float loss_chg, unsigned split_index, float split_value, bool default_left ){
                    if( this->NeedReplace( loss_chg, split_index ) ){
                        this->loss_chg = loss_chg;
                        if( default_left ) split_index |= (1U << 31);
                        this->sindex = split_index;
                        this->split_value = split_value;
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

            /*! \brief per thread x per node entry to store tmp data */
            struct ThreadEntry{
                /*! \brief sum gradient statistics */
                double sum_grad;
                /*! \brief sum hessian statistics */
                double sum_hess;
                /*! \brief last feature value scanned */
                float  last_fvalue;
                /*! \brief current best solution */
                SplitEntry best;
                /*! \brief constructor */
                ThreadEntry( void ){                    
                    this->ClearStats();
                }
                /*! \brief clear statistics */
                inline void ClearStats( void ){
                    sum_grad = sum_hess = 0.0;
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
                    tree.ChangeToLeaf( pid, param.learning_rate * snode[pid].weight );
                    // tail recursion
                    this->TryPruneLeaf( pid, depth - 1 );
                }
            }
            // prune tree
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
        private:
            // update queue expand 
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

            // make leaf nodes for all qexpand, update node statistics, mark leaf value
            inline void InitNewNode( const std::vector<int> &qexpand ){
                {// setup statistics space for each tree node
                   for( size_t i = 0; i < stemp.size(); ++ i ){
                        stemp[i].resize( tree.param.num_nodes, ThreadEntry() );
                   }
                    snode.resize( tree.param.num_nodes, NodeEntry() );
                }

                const unsigned ndata = static_cast<unsigned>( position.size() );
                
                #pragma omp parallel for schedule( static )
                for( unsigned i = 0; i < ndata; ++ i ){
                    const int tid = omp_get_thread_num();
                    if( position[i] < 0 ) continue; 
                    stemp[tid][ position[i] ].sum_grad += grad[i];
                    stemp[tid][ position[i] ].sum_hess += hess[i];
                }

                for( size_t j = 0; j < qexpand.size(); ++ j ){
                    const int nid = qexpand[ j ];
                    double sum_grad = 0.0, sum_hess = 0.0;
                    for( size_t tid = 0; tid < stemp.size(); tid ++ ){
                        sum_grad += stemp[tid][nid].sum_grad;
                        sum_hess += stemp[tid][nid].sum_hess;
                    }
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
        private:
            // enumerate the split values of specific feature
            template<typename Iter>
            inline void EnumerateSplit( Iter it, const unsigned fid, std::vector<ThreadEntry> &temp, bool is_forward_search ){
                // clear all the temp statistics
                for( size_t j = 0; j < qexpand.size(); ++ j ){
                    temp[ qexpand[j] ].ClearStats();
                }
                
                while( it.Next() ){
                    const bst_uint ridx = it.rindex();
                    const int nid = position[ ridx ];
                    if( nid < 0 ) continue;

                    const float fvalue = it.fvalue();           
                    ThreadEntry &e = temp[ nid ];

                    // test if first hit, this is fine, because we set 0 during init
                    if( e.sum_hess == 0.0 ){
                        e.sum_grad = grad[ ridx ];
                        e.sum_hess = hess[ ridx ];
                        e.last_fvalue = fvalue;
                    }else{
                        // try to find a split
                        if( fabsf(fvalue - e.last_fvalue) > rt_2eps && e.sum_hess >= param.min_child_weight ){
                            const double csum_hess = snode[ nid ].sum_hess - e.sum_hess;
                            if( csum_hess >= param.min_child_weight ){
                                const double csum_grad = snode[nid].sum_grad - e.sum_grad; 
                                const double loss_chg = 
                                    + param.CalcGain( e.sum_grad, e.sum_hess, snode[nid].weight ) 
                                    + param.CalcGain( csum_grad , csum_hess , snode[nid].weight )
                                    - snode[nid].root_gain;
                                e.best.Update( loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, !is_forward_search );
                            }
                        }
                        // update the statistics
                        e.sum_grad += grad[ ridx ];
                        e.sum_hess += hess[ ridx ];
                        e.last_fvalue = fvalue;
                    }
                }
                // finish updating all statistics, check if it is possible to include all sum statistics
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[ i ];
                    ThreadEntry &e = temp[ nid ];
                    const double csum_hess = snode[nid].sum_hess - e.sum_hess;

                    if( e.sum_hess >= param.min_child_weight && csum_hess >= param.min_child_weight ){
                        const double csum_grad = snode[nid].sum_grad - e.sum_grad; 
                        const double loss_chg = 
                            + param.CalcGain( e.sum_grad, e.sum_hess, snode[nid].weight ) 
                            + param.CalcGain(  csum_grad,  csum_hess, snode[nid].weight )
                            - snode[nid].root_gain;
                        const float delta = is_forward_search ? rt_eps:-rt_eps;
                        e.best.Update( loss_chg, fid, e.last_fvalue + delta, !is_forward_search );
                    }
                }
            }

            // find splits at current level
            inline void FindSplit( int depth ){
                const unsigned nsize = static_cast<unsigned>( feat_index.size() );
                
                #pragma omp parallel for schedule( dynamic, 1 )
                for( unsigned i = 0; i < nsize; ++ i ){
                    const unsigned fid = feat_index[i];
                    const int tid = omp_get_thread_num();
                    if( param.need_forward_search() ){
                        this->EnumerateSplit( smat.GetSortedCol(fid), fid, stemp[tid], true );
                    }
                    if( param.need_backward_search() ){
                        this->EnumerateSplit( smat.GetReverseSortedCol(fid), fid, stemp[tid], false );
                    }
                }

                // after this each thread's stemp will get the best candidates, aggregate results
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[ i ];
                    NodeEntry &e = snode[ nid ];
                    for( int tid = 0; tid < this->nthread; ++ tid ){
                        e.best.Update( stemp[ tid ][ nid ].best );
                    }
                    
                    // now we know the solution in snode[ nid ], set split
                    if( e.best.loss_chg > rt_eps ){
                        tree.AddChilds( nid );
                        tree[ nid ].set_split( e.best.split_index(), e.best.split_value, e.best.default_left() );
                    } else{
                        tree[ nid ].set_leaf( e.weight * param.learning_rate );
                    }  
                }

                {// reset position 
                    // step 1, set default direct nodes to default, and leaf nodes to -1, 
                    const unsigned ndata = static_cast<unsigned>( position.size() );
                    #pragma omp parallel for schedule( static )
                    for( unsigned i = 0; i < ndata; ++ i ){
                        const int nid = position[i];
                        if( nid >= 0 ){
                            if( tree[ nid ].is_leaf() ){
                                position[i] = -1;
                            }else{
                                // push to default branch, correct latter
                                position[i] = tree[nid].default_left() ? tree[nid].cleft(): tree[nid].cright();
                            }
                        }
                    }

                    // step 2, classify the non-default data into right places
                    std::vector<unsigned> fsplits;

                    for( size_t i = 0; i < qexpand.size(); ++ i ){
                        const int nid = qexpand[i];
                        if( !tree[nid].is_leaf() ) fsplits.push_back( tree[nid].split_index() );
                    }
                    std::sort( fsplits.begin(), fsplits.end() );
                    fsplits.resize( std::unique( fsplits.begin(), fsplits.end() ) - fsplits.begin() );

                    const unsigned nfeats = static_cast<unsigned>( fsplits.size() );
                    #pragma omp parallel for schedule( dynamic, 1 )
                    for( unsigned i = 0; i < nfeats; ++ i ){
                        const unsigned fid = fsplits[i];
                        for( typename FMatrix::ColIter it = smat.GetSortedCol( fid ); it.Next(); ){
                            const bst_uint ridx = it.rindex();
                            int nid = position[ ridx ];
                            if( nid == -1 ) continue;
                            // go back to parent, correct those who are not default
                            nid = tree[ nid ].parent();
                            if( tree[ nid ].split_index() == fid ){
                                if( it.fvalue() < tree[nid].split_cond() ){
                                    position[ ridx ] = tree[ nid ].cleft();
                                }else{
                                    position[ ridx ] = tree[ nid ].cright();
                                }
                            }
                        }
                    }
                }
            }
        private:
            // initialize temp data structure
            inline void InitData( void ){
                {
                    position.resize( grad.size() );
                    if( root_index.size() == 0 ){
                        std::fill( position.begin(), position.end(), 0 );
                    }else{
                        for( size_t i = 0; i < root_index.size(); ++ i ){
                            position[i] = root_index[i];
                            utils::Assert( root_index[i] < (unsigned)tree.param.num_roots, "root index exceed setting" );
                        }
                    }
                    // mark delete for the deleted datas
                    for( size_t i = 0; i < grad.size(); ++ i ){
                        if( hess[i] < 0.0f ) position[i] = -1;
                    }
                    if( param.subsample < 1.0f - 1e-6f ){
                        for( size_t i = 0; i < grad.size(); ++ i ){
                            if( hess[i] < 0.0f ) continue;
                            if( random::SampleBinary( param.subsample) == 0 ){
                                position[ i ] = -1;
                            }
                        }
                    }
                }
                
                {// initialize feature index
                    int ncol = static_cast<int>( smat.NumCol() );
                    for( int i = 0; i < ncol; i ++ ){
                        if( smat.GetSortedCol(i).Next() ){
                            feat_index.push_back( i );
                        }
                    }                    
                    random::Shuffle( feat_index );
                }
                {// setup temp space for each thread
                    if( param.nthread != 0 ){
                        omp_set_num_threads( param.nthread );
                    }
                    #pragma omp parallel
                    {
                        this->nthread = omp_get_num_threads();
                    }

                    // reserve a small space
                    stemp.resize( this->nthread, std::vector<ThreadEntry>() );
                    for( size_t i = 0; i < stemp.size(); ++ i ){
                        stemp[i].reserve( 256 );
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
            // local helper tmp data structure
            // statistics
            int stat_num_pruned;
            // number of omp thread used during training
            int nthread;
            // queue of nodes to be expanded
            std::vector<int> qexpand;
            // Per feature: shuffle index of each feature index
            std::vector<int> feat_index;
            // Instance Data: current node position in the tree of each instance
            std::vector<int> position;
            // TreeNode Data: statistics for each constructed node
            std::vector<NodeEntry> snode;
            // PerThread x PerTreeNode: statistics for per thread construction
            std::vector< std::vector<ThreadEntry> > stemp;
        private:
            // original data that supports tree construction
            RegTree &tree;
            const TreeParamTrain &param;
            const std::vector<float> &grad;
            const std::vector<float> &hess;
            const FMatrix            &smat;
            const std::vector<unsigned> &root_index;
            
        };
    };
};
#endif
