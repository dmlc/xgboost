#ifndef XGBOOST_TREE_HPP
#define XGBOOST_TREE_HPP
/*!
 * \file xgboost_tree.hpp
 * \brief implementation of regression tree
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
#include "xgboost_tree_model.h"

namespace xgboost{
    namespace booster{
        const bool rt_debug = false;
        // whether to check bugs
        const bool check_bug = false;
    
        const float rt_eps = 1e-5f;
        const float rt_2eps = rt_eps * 2.0f;
        
        inline double sqr( double a ){
            return a * a;
        }
    };
};
#include "../../utils/xgboost_fmap.h"
#include "xgboost_svdf_tree.hpp"
#include "xgboost_col_treemaker.hpp"
#include "xgboost_row_treemaker.hpp"

namespace xgboost{
    namespace booster{
        // regression tree, construction algorithm is seperated from this class
        // see RegTreeUpdater
        template<typename FMatrix>
        class RegTreeTrainer : public InterfaceBooster<FMatrix>{
        public:
            RegTreeTrainer( void ){ 
                silent = 0; tree_maker = 1; 
                // interact mode
                interact_type = 0;
                interact_node = 0;
                // normally we won't have more than 64 OpenMP threads
                threadtemp.resize( 64, ThreadEntry() );
            }
            virtual ~RegTreeTrainer( void ){}
        public:
            virtual void SetParam( const char *name, const char *val ){
                if( !strcmp( name, "silent") )      silent = atoi( val );
                if( !strcmp( name, "tree_maker") )  tree_maker = atoi( val );
                if( !strncmp( name, "interact:", 9) ){
                    const char *ename = name + 9;
                    interact_node = atoi( val );
                    if( !strcmp( ename, "expand") ) {
                        interact_type = 1;
                    }
                    if( !strcmp( ename, "remove") ) {
                        interact_type = 2;
                    }                    
                }
                param.SetParam( name, val );
                constrain.SetParam( name, val );
                tree.param.SetParam( name, val );
            }
            virtual void LoadModel( utils::IStream &fi ){
                tree.LoadModel( fi );
            }
            virtual void SaveModel( utils::IStream &fo ) const{
                tree.SaveModel( fo );
            }
            virtual void InitModel( void ){
                tree.InitModel();
            }
        public:
            virtual void DoBoost( std::vector<float> &grad, 
                                  std::vector<float> &hess,
                                  const FMatrix &smat,
                                  const std::vector<unsigned> &root_index ){
                utils::Assert( grad.size() < UINT_MAX, "number of instance exceed what we can handle" );

                // interactive update 
                if( interact_type != 0 ){
                    switch( interact_type ){
                    case 1: this->ExpandNode( grad, hess, smat, root_index, interact_node ); return;
                    case 2: this->CollapseNode( grad, hess, smat, root_index, interact_node ); return;
                    default: utils::Error("unknown interact type");
                    }
                }

                if( !silent ){
                    printf( "\nbuild GBRT with %u instances\n", (unsigned)grad.size() );
                }
                int num_pruned;
                switch( tree_maker ){
                case 0: {
                    utils::Assert( !constrain.HasConstrain(), "tree maker 0 does not support constrain" );
                    RTreeUpdater<FMatrix> updater( param, tree, grad, hess, smat, root_index );
                    tree.param.max_depth = updater.do_boost( num_pruned );
                    break;
                }
                case 1:{
                    ColTreeMaker<FMatrix> maker( tree, param, grad, hess, smat, root_index, constrain );
                    maker.Make( tree.param.max_depth, num_pruned );
                    break;
                }
                case 2:{
                    RowTreeMaker<FMatrix> maker( tree, param, grad, hess, smat, root_index, constrain );
                    maker.Make( tree.param.max_depth, num_pruned );
                    break;
                }                    
                default: utils::Error("unknown tree maker");
                }
                if( !silent ){
                    printf( "tree train end, %d roots, %d extra nodes, %d pruned nodes ,max_depth=%d\n", 
                            tree.param.num_roots, tree.num_extra_nodes(), num_pruned, tree.MaxDepth() );
                }
            }            
            virtual float Predict( const FMatrix &fmat, bst_uint ridx, unsigned gid = 0 ){
                ThreadEntry &e = this->InitTmp();
                this->PrepareTmp( fmat.GetRow(ridx), e );
                int pid = this->GetLeafIndex( e.feat, e.funknown, gid );
                this->DropTmp( fmat.GetRow(ridx), e );
                return tree[ pid ].leaf_value();          
            }
            virtual int GetLeafIndex( const std::vector<float> &feat,
                                      const std::vector<bool>  &funknown,
                                      unsigned gid = 0 ){
                // start from groups that belongs to current data
                int pid = (int)gid;
                // tranverse tree
                while( !tree[ pid ].is_leaf() ){
                    unsigned split_index = tree[ pid ].split_index();
                    pid = this->GetNext( pid, feat[ split_index ], funknown[ split_index ] );
                }
                return pid;
            }

            virtual void PredPath( std::vector<int> &path, const FMatrix &fmat, bst_uint ridx, unsigned gid = 0 ){
                path.clear();
                ThreadEntry &e = this->InitTmp();
                this->PrepareTmp( fmat.GetRow(ridx), e );
                
                int pid = (int)gid;
                path.push_back( pid );
                // tranverse tree
                while( !tree[ pid ].is_leaf() ){                    
                    unsigned split_index = tree[ pid ].split_index();
                    pid = this->GetNext( pid, e.feat[ split_index ], e.funknown[ split_index ] );
                    path.push_back( pid );
                }
                this->DropTmp( fmat.GetRow(ridx), e );
            }
            virtual float Predict( const std::vector<float> &feat, 
                                   const std::vector<bool>  &funknown,
                                   unsigned gid = 0 ){
                utils::Assert( feat.size() >= (size_t)tree.param.num_feature,
                               "input data smaller than num feature" );
                int pid = this->GetLeafIndex( feat, funknown, gid );
                return tree[ pid ].leaf_value();
            }            
            virtual void DumpModel( FILE *fo, const utils::FeatMap &fmap, bool with_stats ){
                tree.DumpModel( fo, fmap, with_stats );
            }
        private:
            inline void CollapseNode( std::vector<float> &grad, 
                                      std::vector<float> &hess,
                                      const FMatrix &fmat,
                                      const std::vector<unsigned> &root_index, 
                                      int nid ){
                std::vector<bst_uint> valid_index;
                for( size_t i = 0; i < grad.size(); i ++ ){
                    ThreadEntry &e = this->InitTmp();
                    this->PrepareTmp( fmat.GetRow(i), e );
                    int pid = root_index.size() == 0 ? 0 : (int)root_index[i];
                    // tranverse tree
                    while( !tree[ pid ].is_leaf() ){                    
                        unsigned split_index = tree[ pid ].split_index();
                        pid = this->GetNext( pid, e.feat[ split_index ], e.funknown[ split_index ] );
                        if( pid == nid ){
                            valid_index.push_back( static_cast<bst_uint>(i) );  break;
                        }
                    }
                    this->DropTmp( fmat.GetRow(i), e );
                }
                RowTreeMaker<FMatrix> maker( tree, param, grad, hess, fmat, root_index, constrain ); 
                maker.Collapse( valid_index, nid );
                if( !silent ){
                    printf( "tree collapse end, max_depth=%d\n", tree.param.max_depth );
                }                
            }
            inline void ExpandNode( std::vector<float> &grad, 
                                    std::vector<float> &hess,
                                    const FMatrix &fmat,
                                    const std::vector<unsigned> &root_index, 
                                    int nid ){
                std::vector<bst_uint> valid_index;
                for( size_t i = 0; i < grad.size(); i ++ ){
                    ThreadEntry &e = this->InitTmp();
                    this->PrepareTmp( fmat.GetRow(i), e );
                    unsigned rtidx = root_index.size() == 0 ? 0 : root_index[i]; 
                    int pid = this->GetLeafIndex( e.feat, e.funknown, rtidx );
                    this->DropTmp( fmat.GetRow(i), e );
                    if( pid == nid ) valid_index.push_back( static_cast<bst_uint>(i) ); 
                }
                RowTreeMaker<FMatrix> maker( tree, param, grad, hess, fmat, root_index, constrain ); 
                bool success =  maker.Expand( valid_index, nid );
                if( !silent ){
                    printf( "tree expand end, success=%d, max_depth=%d\n", (int)success, tree.MaxDepth() );
                }                
            }
        private:
            // silent 
            int silent;
            RegTree tree;
            TreeParamTrain param;
        private:
            // some training parameters
            // tree maker
            int tree_maker;
            // interaction
            int interact_type;
            int interact_node;         
            // feature constrain
            utils::FeatConstrain  constrain;   
        private:
            struct ThreadEntry{
                std::vector<float> feat;
                std::vector<bool>  funknown;
            };
            std::vector<ThreadEntry> threadtemp;
        private:
            inline ThreadEntry& InitTmp( void ){
                const int tid = omp_get_thread_num();
                utils::Assert( tid < (int)threadtemp.size(), "RTreeUpdater: threadtemp pool is too small" );
                ThreadEntry &e = threadtemp[ tid ];
                if( e.feat.size() != (size_t)tree.param.num_feature ){
                    e.feat.resize( tree.param.num_feature );
                    e.funknown.resize( tree.param.num_feature );
                    std::fill( e.funknown.begin(), e.funknown.end(), true );
                }
                return e;
            }
            inline void PrepareTmp( typename FMatrix::RowIter it, ThreadEntry &e ){
                while( it.Next() ){
                    const bst_uint findex = it.findex();
                    utils::Assert( findex < (unsigned)tree.param.num_feature , "input feature execeed bound" );
                    e.funknown[ findex ] = false;
                    e.feat[ findex ] = it.fvalue();
                } 
            }
            inline void DropTmp( typename FMatrix::RowIter it, ThreadEntry &e ){
                while( it.Next() ){
                    e.funknown[ it.findex() ] = true;
                }
            }

            inline int GetNext( int pid, float fvalue, bool is_unknown ){
                float split_value = tree[ pid ].split_cond();
                if( is_unknown ){ 
                    return tree[ pid ].cdefault();
                }else{
                    if( fvalue < split_value ) return tree[ pid ].cleft();
                    else return tree[ pid ].cright();
                }
            }
        };
    };
};

#endif
