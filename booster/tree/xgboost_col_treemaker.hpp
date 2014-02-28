#ifndef _XGBOOST_COL_TREEMAKER_HPP_
#define _XGBOOST_COL_TREEMAKER_HPP_
/*!
 * \file xgboost_col_treemaker.hpp
 * \brief implementation of regression tree maker,
 *        use a column based approach, with OpenMP 
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
#include <vector>
#include <omp.h>
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
            inline void Make( void ){
            }
        private:
            // statistics that is helpful to decide a split
            struct SplitEntry{
                /*! \brief gain in terms of loss */
                float  loss_gain;
                /*! \brief weight calculated related to current data */
                float  weight;
                /*! \brief split index */
                unsigned  sindex;
                /*! \brief split value */
                float     split_value;
                /*! \brief constructor */
                SplitEntry( void ){
                    weight = loss_gain = 0.0f;
                    split_value = 0.0f; sindex = 0;
                }
                inline void SetSplit( unsigned split_index, float split_value, bool default_left ){
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
            /*! \brief per thread x per node entry to store tmp data */
            struct ThreadEntry{
                /*! \brief sum gradient statistics */
                double sum_grad;
                /*! \brief sum hessian statistics */
                double sum_hess;
                /*! \brief current best solution */
                SplitEntry best;
                ThreadEntry( void ){
                    sum_grad = sum_hess = 0;
                }
            };
        private:
            // find split at current level
            inline void FindSplit( int depth ){
                unsigned nsize = static_cast<unsigned>(feat_index.size());

                #pragma omp parallel for schedule( dynamic, 1 )
                for( unsigned i = 0; i < nsize; ++ i ){
                    const unsigned fid = feat_index[i];                    
                }
            }
            // initialize temp data structure
            inline void InitData( void ){
                position.resize( grad.size() );
                if( root_index.size() == 0 ){
                    std::fill( position.begin(), position.end(), 0 );
                }else{
                    for( size_t i = 0; i < root_index.size(); ++ i ){
                        position[i] = root_index[i];
                        utils::Assert( root_index[i] < (unsigned)tree.param.num_roots, "root index exceed setting" );
                    }
                }
                {// initialize feature index
                    for( int i = 0; i < tree.param.num_feature; i ++ ){
                        if( smat.GetSortedCol(i).Next() ){
                            feat_index.push_back( i );
                        }
                    }
                    random::Shuffle( feat_index );
                }
                {// setup temp space for each thread
                    int nthread;
                    #pragma omp parallel
                    {
                        nthread = omp_get_num_threads();
                    }                
                    // reserve a small space
                    stemp.resize( nthread, std::vector<ThreadEntry>() );
                    for( size_t i = 0; i < stemp.size(); ++ i ){
                        stemp[i].reserve( 256 );
                        stemp[i].resize( tree.param.num_roots, ThreadEntry() );
                    }
                }
                {// setup statistics space for each tree node
                    snode.resize( tree.param.num_roots, SplitEntry() );
                }
            }
        private:
            // local helper tmp data structure
            // Per feature: shuffle index of each feature index
            std::vector<int> feat_index;
            // Instance Data: current node position in the tree of each instance
            std::vector<int> position;                
            // TreeNode Data: statistics for each constructed node
            std::vector<SplitEntry> snode;
            // PerThread x PerTreeNode: statistics for per thread construction
            std::vector< std::vector<SplitEntry> > stemp;
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
