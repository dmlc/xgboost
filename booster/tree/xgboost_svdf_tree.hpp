#ifndef XGBOOST_APEX_TREE_HPP
#define XGBOOST_APEX_TREE_HPP
/*!
 * \file xgboost_svdf_tree.hpp
 * \brief implementation of regression tree constructor, with layerwise support
 *        this file is adapted from GBRT implementation in SVDFeature project
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn, tianqi.tchen@gmail.com 
 */
#include <algorithm>
#include "xgboost_tree_model.h"
#include "../../utils/xgboost_random.h"
#include "../../utils/xgboost_matrix_csr.h"

namespace xgboost{
    namespace booster{        
        inline void assert_sorted( unsigned *idset, int len ){
            if( !rt_debug || !check_bug ) return;
            for( int i = 1; i < len; i ++ ){
                utils::Assert( idset[i-1] < idset[i], "idset not sorted" );
            }
        }
    };
    
    namespace booster{                
        // selecter of rtree to find the suitable candidate
        class RTSelecter{
        public:
            struct Entry{
                float  loss_chg;
                size_t start;
                int    len;
                unsigned sindex;
                float    split_value;
                Entry(){}
                Entry( float loss_chg, size_t start, int len, unsigned split_index, float split_value, bool default_left ){
                    this->loss_chg = loss_chg;
                    this->start    = start;
                    this->len      = len;
                    if( default_left ) split_index |= (1U << 31);
                    this->sindex = split_index;
                    this->split_value = split_value;
                }
                inline unsigned split_index( void ) const{
                    return sindex & ( (1U<<31) - 1U );
                }
                inline bool default_left( void ) const{
                    return (sindex >> 31) != 0;
                }
            };
        private:
            Entry best_entry;
        public:
            RTSelecter( void ){
                memset( &best_entry, 0, sizeof(best_entry) );
                best_entry.loss_chg = 0.0f;
            }
            inline void push_back( const Entry &e ){
                if( e.loss_chg > best_entry.loss_chg ) best_entry = e;
            }
            inline const Entry & select( void ){            
                return best_entry;                
            }
        };
        
        
        // updater of rtree, allows the parameters to be stored inside, key solver
        template<typename FMatrix>
        class RTreeUpdater{
        protected:
            // training task, element of single task
            struct Task{
                // node id in tree
                int nid;
                // idset pointer, instance id in [idset,idset+len)
                unsigned *idset;
                // length of idset
                unsigned len;            
                // base_weight of parent
                float parent_base_weight;
                Task(){}
                Task( int nid, unsigned *idset, unsigned len, float pweight = 0.0f ){
                    this->nid = nid;
                    this->idset = idset;
                    this->len = len;
                    this->parent_base_weight = pweight;
                }
            };
            
            // sparse column entry
            struct SCEntry{
                // feature value 
                float    fvalue;
                // row index in grad
                unsigned rindex;
                SCEntry(){}
                SCEntry( float fvalue, unsigned rindex ){
                    this->fvalue = fvalue; this->rindex = rindex;
                }
                inline bool operator<( const SCEntry &p ) const{
                    return fvalue < p.fvalue;
                }
            };
        private:
            // training parameter
            const TreeParamTrain &param;
            // parameters, reference
            RegTree &tree;
            std::vector<float> &grad;
            std::vector<float> &hess;
            const FMatrix &smat;
            const std::vector<unsigned> &group_id;
        private:
            // maximum depth up to now
            int max_depth;
            // number of nodes being pruned
            int num_pruned;
            // stack to store current task
            std::vector<Task> task_stack;
            // temporal space for index set
            std::vector<unsigned> idset;
        private:
            // task management: NOTE DFS here
            inline void add_task( Task tsk ){
                task_stack.push_back( tsk );
            }
            inline bool next_task( Task &tsk ){
                if( task_stack.size() == 0 ) return false;
                tsk = task_stack.back(); 
                task_stack.pop_back();
                return true;
            } 
        private:
            // try to prune off current leaf, return true if successful
            inline void try_prune_leaf( int nid, int depth ){
                if( tree[ nid ].is_root() ) return;
                int pid = tree[ nid ].parent();
                RegTree::NodeStat &s = tree.stat( pid );
                s.leaf_child_cnt ++;
                
                if( s.leaf_child_cnt >= 2 && param.need_prune( s.loss_chg, depth - 1 ) ){
                    // need to be pruned
                    tree.ChangeToLeaf( pid, param.learning_rate * s.base_weight );
                    // add statistics to number of nodes pruned
                    num_pruned += 2;
                    // tail recursion
                    this->try_prune_leaf( pid, depth - 1 );
                }
            }
            // make leaf for current node :)
            inline void make_leaf( Task tsk, double sum_grad, double sum_hess, bool compute ){
                for( unsigned i = 0; i < tsk.len; i ++ ){
                    const unsigned ridx = tsk.idset[i];
                    if( compute ){
                        sum_grad += grad[ ridx ];
                        sum_hess += hess[ ridx ];
                    }
                }
                tree.stat( tsk.nid ).sum_hess = static_cast<float>( sum_hess );
                tree[ tsk.nid ].set_leaf( param.learning_rate * param.CalcWeight( sum_grad, sum_hess, tsk.parent_base_weight ) );
                this->try_prune_leaf( tsk.nid, tree.GetDepth( tsk.nid ) );
            }
        private:
            // make split for current task, re-arrange positions in idset
            inline void make_split( Task tsk, const SCEntry *entry, int num, float loss_chg, double sum_hess, double base_weight ){
                // before split, first prepare statistics
                RegTree::NodeStat &s = tree.stat( tsk.nid );
                s.loss_chg = loss_chg; 
                s.leaf_child_cnt = 0;
                s.sum_hess = static_cast<float>( sum_hess );
                s.base_weight = static_cast<float>( base_weight );
                
                // add childs to current node
                tree.AddChilds( tsk.nid );
                // assert that idset is sorted
                assert_sorted( tsk.idset, tsk.len );
                // use merge sort style to get the solution
                std::vector<unsigned> qset;
                for( int i = 0; i < num; i ++ ){
                    qset.push_back( entry[i].rindex );
                }
                std::sort( qset.begin(), qset.end() );
                // do merge sort style, make the other set, remove elements in qset
                for( unsigned i = 0, top = 0; i < tsk.len; i ++ ){
                    if( top < qset.size() ){
                        if( tsk.idset[ i ] != qset[ top ] ){
                            tsk.idset[ i - top ] = tsk.idset[ i ];
                        }else{
                            top ++;
                        }
                    }else{
                        tsk.idset[ i - qset.size() ] = tsk.idset[ i ];
                    }
                }
                // get two parts 
                RegTree::Node &n = tree[ tsk.nid ];
                Task def_part( n.default_left() ? n.cleft() : n.cright(), tsk.idset, tsk.len - qset.size(), s.base_weight );
                Task spl_part( n.default_left() ? n.cright(): n.cleft() , tsk.idset + def_part.len, qset.size(), s.base_weight );  
                // fill back split part
                for( unsigned i = 0; i < spl_part.len; i ++ ){
                    spl_part.idset[ i ] = qset[ i ];
                }
                // add tasks to the queue
                this->add_task( def_part ); 
                this->add_task( spl_part );
            }
            
            // enumerate split point of the tree
            inline void enumerate_split( RTSelecter &sglobal, int tlen,
                                         double rsum_grad, double rsum_hess, double root_gain,
                                         const SCEntry *entry, size_t start, size_t end, 
                                         int findex, float parent_base_weight ){
                // local selecter
                RTSelecter slocal;
                
                if( param.need_forward_search() ){
                    // forward process, default right
                    double csum_grad = 0.0, csum_hess = 0.0;
                    for( size_t j = start; j < end; j ++ ){
                        const unsigned ridx = entry[ j ].rindex;
                        csum_grad += grad[ ridx ];
                        csum_hess += hess[ ridx ];
                        // check for split
                        if( j == end - 1 || entry[j].fvalue + rt_2eps < entry[ j + 1 ].fvalue ){
                            if( csum_hess < param.min_child_weight ) continue;
                            const double dsum_hess = rsum_hess - csum_hess;
                            if( dsum_hess < param.min_child_weight ) break;                        
                            // change of loss 
                            double loss_chg = 
                            param.CalcGain( csum_grad, csum_hess, parent_base_weight ) + 
                                param.CalcGain( rsum_grad - csum_grad, dsum_hess, parent_base_weight ) - root_gain;
                            
                            const int clen = static_cast<int>( j + 1 - start );
                            // add candidate to selecter
                            slocal.push_back( RTSelecter::Entry( loss_chg, start, clen, findex, 
                                                                 j == end - 1 ? entry[j].fvalue + rt_eps : 0.5 * (entry[j].fvalue+entry[j+1].fvalue),
                                                                 false ) );
                        }
                    }
                }
                
                if( param.need_backward_search() ){
                    // backward process, default left
                    double csum_grad = 0.0, csum_hess = 0.0;
                    for( size_t j = end; j > start; j -- ){
                        const unsigned ridx = entry[ j - 1 ].rindex;
                        csum_grad += grad[ ridx ];
                        csum_hess += hess[ ridx ];
                        // check for split
                        if( j == start + 1 || entry[ j - 2 ].fvalue + rt_2eps < entry[ j - 1 ].fvalue ){
                            if( csum_hess < param.min_child_weight ) continue;
                            const double dsum_hess = rsum_hess - csum_hess;
                            if( dsum_hess < param.min_child_weight ) break;
                            double loss_chg = param.CalcGain( csum_grad, csum_hess, parent_base_weight ) + 
                                param.CalcGain( rsum_grad - csum_grad, dsum_hess, parent_base_weight ) - root_gain;
                            const int clen = static_cast<int>( end - j + 1 );
                            // add candidate to selecter
                            slocal.push_back( RTSelecter::Entry( loss_chg, j - 1, clen, findex,
                                                                 j == start + 1 ? entry[j-1].fvalue - rt_eps : 0.5 * (entry[j-2].fvalue + entry[j-1].fvalue), 
                                                                 true ) );
                        }
                    }
                }
                sglobal.push_back( slocal.select() );
            }
            
        private:
            // temporal storage for expand column major
            std::vector<size_t>  tmp_rptr;        
            // find split for current task, another implementation of expand in column major manner
            // should be more memory frugal, avoid global sorting across feature       
            inline void expand( Task tsk ){
                // assert that idset is sorted
                // if reach maximum depth, make leaf from current node
                int depth = tree.GetDepth( tsk.nid );
                // update statistiss
                if( depth > max_depth ) max_depth = depth; 
                // if bigger than max depth
                if( depth >= param.max_depth ){
                    this->make_leaf( tsk, 0.0, 0.0, true ); return;
                }
                // convert to column major CSR format
                const int nrows = tree.param.num_feature;
                if( tmp_rptr.size() == 0 ){
                    // initialize tmp storage in first usage
                    tmp_rptr.resize( nrows + 1 ); 
                    std::fill( tmp_rptr.begin(), tmp_rptr.end(), 0 );
                }
                // records the columns
                std::vector<SCEntry> entry;
                // records the active features
                std::vector<size_t>  aclist;
                utils::SparseCSRMBuilder<SCEntry,true> builder( tmp_rptr, entry, aclist );
                builder.InitBudget( nrows );
                // statistics of root
                double rsum_grad = 0.0, rsum_hess = 0.0;            
                for( unsigned i = 0; i < tsk.len; i ++ ){
                    const unsigned ridx = tsk.idset[i];
                    rsum_grad  += grad[ ridx ];
                    rsum_hess  += hess[ ridx ];
                    
                    for( typename FMatrix::RowIter it = smat.GetRow(ridx); it.Next(); ){
                        builder.AddBudget( it.findex() );
                    }
                }
                
                // if minimum split weight is not meet
                if( param.cannot_split( rsum_hess, depth )  ){
                    this->make_leaf( tsk, rsum_grad, rsum_hess, false ); builder.Cleanup(); return; 
                }
                
                builder.InitStorage();
                for( unsigned i = 0; i < tsk.len; i ++ ){
                    const unsigned ridx = tsk.idset[i];
                    for( typename FMatrix::RowIter it = smat.GetRow(ridx); it.Next(); ){
                        builder.PushElem( it.findex(), SCEntry( it.fvalue(), ridx ) );
                    }
                }
                // --- end of building column major matrix ---
                // after this point, tmp_rptr and entry is ready to use
                
                // global selecter
                RTSelecter sglobal;
                // gain root 
                const double root_gain = param.CalcRootGain( rsum_grad, rsum_hess );
                // KEY: layerwise, weight of current node if it is leaf
                const double base_weight = param.CalcWeight( rsum_grad, rsum_hess, tsk.parent_base_weight );
                // enumerate feature index
                for( size_t i = 0; i < aclist.size(); i ++ ){
                    int findex = static_cast<int>( aclist[i] );                
                    size_t start = tmp_rptr[ findex ];
                    size_t end   = tmp_rptr[ findex + 1 ];
                    utils::Assert( start < end, "bug" );
                    // local sort can be faster when the features are sparse
                    std::sort( entry.begin() + start, entry.begin() + end );
                    // local selecter
                    this->enumerate_split( sglobal, tsk.len,
                                           rsum_grad, rsum_hess, root_gain,
                                           &entry[0], start, end, findex, base_weight );
                }
                // Cleanup tmp_rptr for next use
                builder.Cleanup();
                // get the best solution
                const RTSelecter::Entry &e = sglobal.select();
                // allowed to split
                if( e.loss_chg > rt_eps ){
                    // add splits
                    tree[ tsk.nid ].set_split( e.split_index(), e.split_value, e.default_left() );
                    // re-arrange idset, push tasks
                    this->make_split( tsk, &entry[ e.start ], e.len, e.loss_chg, rsum_hess, base_weight ); 
                }else{
                    // make leaf if we didn't meet requirement
                    this->make_leaf( tsk, rsum_grad, rsum_hess, false );
                }
            }
        private:
            // initialize the tasks
            inline void init_tasks( size_t ngrads ){
                // add group partition if necessary
                if( group_id.size() == 0 ){       
                    if( param.subsample > 1.0f - 1e-6f ){ 
                        idset.resize( 0 );
                        for( size_t i = 0; i < ngrads; i ++ ){
                            if( hess[i] < 0.0f ) continue;
                            idset.push_back( (unsigned)i );
                        } 
                    }else{
                        idset.resize( 0 );
                        for( size_t i = 0; i < ngrads; i ++ ){
                            if( random::SampleBinary( param.subsample ) != 0 ){
                                idset.push_back( (unsigned)i );
                            }
                        } 
                    }
                    this->add_task( Task( 0, &idset[0], idset.size() ) ); return;
                }
                
                utils::Assert( group_id.size() == ngrads, "number of groups must be exact" );            
                {// new method for grouping, use CSR builder
                    std::vector<size_t>   rptr;
                    utils::SparseCSRMBuilder<unsigned> builder( rptr, idset );
                    builder.InitBudget( tree.param.num_roots );
                    for( size_t i = 0; i < group_id.size(); i ++ ){
                        // drop invalid elements
                        if( hess[ i ] < 0.0f ) continue;
                        utils::Assert( group_id[ i ] < (unsigned)tree.param.num_roots, 
                                       "group id exceed number of roots" );
                        builder.AddBudget( group_id[ i ] );
                    }
                    builder.InitStorage();
                    for( size_t i = 0; i < group_id.size(); i ++ ){
                        // drop invalid elements
                        if( hess[ i ] < 0.0f ) continue;
                        builder.PushElem( group_id[ i ], static_cast<unsigned>(i) );
                    }
                    for( size_t i = 1; i < rptr.size(); i ++ ){
                        const size_t start = rptr[ i - 1 ], end = rptr[ i ];
                        if( start < end ){
                            this->add_task( Task( i - 1, &idset[ start ], end - start ) );
                        }
                    }
                }
            }
        public:
            RTreeUpdater( const TreeParamTrain &pparam, 
                          RegTree &ptree,
                          std::vector<float> &pgrad,
                          std::vector<float> &phess,
                          const FMatrix &psmat, 
                          const std::vector<unsigned> &pgroup_id ):
                param( pparam ), tree( ptree ), grad( pgrad ), hess( phess ),
                smat( psmat ), group_id( pgroup_id ){
            }
            inline int do_boost( int &num_pruned ){
                this->init_tasks( grad.size() );
                this->max_depth = 0;
                this->num_pruned = 0;
                Task tsk;
                while( this->next_task( tsk ) ){
                    this->expand( tsk );
                }
                num_pruned = this->num_pruned;
                return max_depth;
            }
        };
    };
};
#endif


