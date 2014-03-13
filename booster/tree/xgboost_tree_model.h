#ifndef XGBOOST_TREE_MODEL_H
#define XGBOOST_TREE_MODEL_H
/*!
 * \file xgboost_tree_model.h
 * \brief generic definition of model structure used in tree models
 *        used to support learning of boosting tree
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <cstring>
#include "../../utils/xgboost_utils.h"
#include "../../utils/xgboost_stream.h"

namespace xgboost{
    namespace booster{
        /*!
         * \brief template class of TreeModel 
         * \tparam TSplitCond data type to indicate split condition
         * \tparam TNodeStat auxiliary statistics of node to help tree building
         */
        template<typename TSplitCond,typename TNodeStat>
        class TreeModel{
        public:
            /*! \brief data type to indicate split condition */
            typedef TNodeStat  NodeStat;
            /*! \brief auxiliary statistics of node to help tree building */
            typedef TSplitCond SplitCond;
        public:
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
                /*! \brief reserved part */
                int reserved[ 32 ];
                /*! \brief constructor */
                Param( void ){
                    max_depth = 0;
                    memset( reserved, 0, sizeof( reserved ) );
                }
                /*! 
                 * \brief set parameters from outside 
                 * \param name name of the parameter
                 * \param val  value of the parameter
                 */
                inline void SetParam( const char *name, const char *val ){
                    if( !strcmp("num_roots", name ) )    num_roots = atoi( val );
                    if( !strcmp("num_feature", name ) )  num_feature = atoi( val );
                }
            };
            /*! \brief tree node */
            class Node{
            private:
                friend class TreeModel<TSplitCond,TNodeStat>;
                /*! 
                 * \brief in leaf node, we have weights, in non-leaf nodes, 
                 *        we have split condition 
                 */
                union Info{
                    float leaf_value;
                    TSplitCond split_cond;
                };
            private:
                // pointer to parent, highest bit is used to indicate whether it's a left child or not 
                int parent_;
                // pointer to left, right
                int cleft_, cright_;
                // split feature index, left split or right split depends on the highest bit
                unsigned sindex_;            
                // extra info
                Info info_;
            private:
                inline void set_parent( int pidx, bool is_left_child = true ){
                    if( is_left_child ) pidx |= (1U << 31);
                    this->parent_ = pidx;
                }
            public:
                /*! \brief index of left child */
                inline int cleft( void ) const{
                    return this->cleft_;
                }
                /*! \brief index of right child */
                inline int cright( void ) const{
                    return this->cright_;
                }
                /*! \brief index of default child when feature is missing */
                inline int cdefault( void ) const{
                    return this->default_left() ? this->cleft() : this->cright();
                }
                /*! \brief feature index of split condition */
                inline unsigned split_index( void ) const{
                    return sindex_ & ( (1U<<31) - 1U );
                }
                /*! \brief when feature is unknown, whether goes to left child */
                inline bool default_left( void ) const{
                    return (sindex_ >> 31) != 0;
                } 
                /*! \brief whether current node is leaf node */
                inline bool is_leaf( void ) const{
                    return cleft_ == -1;
                }
                /*! \brief get leaf value of leaf node */
                inline float leaf_value( void ) const{
                    return (this->info_).leaf_value;
                }
                /*! \brief get split condition of the node */
                inline TSplitCond split_cond( void ) const{
                    return (this->info_).split_cond;
                }
                /*! \brief get parent of the node */
                inline int parent( void ) const{
                    return parent_ & ( (1U << 31) - 1 );
                } 
                /*! \brief whether current node is left child */
                inline bool is_left_child( void ) const{
                    return ( parent_ & (1U << 31)) != 0;
                }
                /*! \brief whether current node is root */
                inline bool is_root( void ) const{
                    return parent_ == -1;
                }
                /*! 
                 * \brief set the right child 
                 * \param nide node id to right child
                 */
                inline void set_right_child( int nid ){
                    this->cright_ = nid;
                }
                /*! 
                 * \brief set split condition of current node 
                 * \param split_index feature index to split
                 * \param split_cond  split condition
                 * \param default_left the default direction when feature is unknown
                 */
                inline void set_split( unsigned split_index, TSplitCond split_cond, bool default_left = false ){
                    if( default_left ) split_index |= (1U << 31);
                    this->sindex_ = split_index;
                    (this->info_).split_cond = split_cond;
                }
                /*! 
                 * \brief set the leaf value of the node
                 * \param value leaf value
                 * \param right right index, could be used to store 
                 *        additional information
                 */
                inline void set_leaf( float value, int right = -1 ){
                    (this->info_).leaf_value = value;
                    this->cleft_ = -1;
                    this->cright_ = right;
                }
            };
        protected:
            // vector of nodes
            std::vector<Node> nodes;
            // stats of nodes
            std::vector<TNodeStat> stats;
        protected:
            // free node space, used during training process
            std::vector<int>  deleted_nodes;
            // allocate a new node, 
            // !!!!!! NOTE: may cause BUG here, nodes.resize
            inline int AllocNode( void ){
                if( param.num_deleted != 0 ){
                    int nd = deleted_nodes.back();
                    deleted_nodes.pop_back();
                    param.num_deleted --;
                    return nd;
                }
                int nd = param.num_nodes ++;
                nodes.resize( param.num_nodes );
                stats.resize( param.num_nodes );
                return nd;
            }
            // delete a tree node
            inline void DeleteNode( int nid ){
                utils::Assert( nid >= param.num_roots, "can not delete root");
                deleted_nodes.push_back( nid );
                nodes[ nid ].set_parent( -1 );
                param.num_deleted ++;
            }
        public:
            /*! 
             * \brief change a non leaf node to a leaf node, delete its children
             * \param rid node id of the node
             * \param new leaf value
             */
            inline void ChangeToLeaf( int rid, float value ){
                utils::Assert( nodes[ nodes[rid].cleft()  ].is_leaf(), "can not delete a non termial child");
                utils::Assert( nodes[ nodes[rid].cright() ].is_leaf(), "can not delete a non termial child");
                this->DeleteNode( nodes[ rid ].cleft() ); 
                this->DeleteNode( nodes[ rid ].cright() );
                nodes[ rid ].set_leaf( value );
            }
            /*! 
             * \brief collapse a non leaf node to a leaf node, delete its children
             * \param rid node id of the node
             * \param new leaf value
             */
            inline void CollapseToLeaf( int rid, float value ){
                if( nodes[rid].is_leaf() ) return;
                if( !nodes[ nodes[rid].cleft()  ].is_leaf() ){
                    CollapseToLeaf( nodes[rid].cleft(), 0.0f );
                }
                if( !nodes[ nodes[rid].cright()  ].is_leaf() ){
                    CollapseToLeaf( nodes[rid].cright(), 0.0f );
                }
                this->ChangeToLeaf( rid, value );
            }
        public:
            /*! \brief model parameter */
            Param param;
        public:
            /*! \brief constructor */
            TreeModel( void ){
                param.num_nodes = 1;
                param.num_roots = 1;
                param.num_deleted = 0;
                nodes.resize( 1 );
            }
            /*! \brief get node given nid */
            inline Node &operator[]( int nid ){
                return nodes[ nid ];
            }
            /*! \brief get node statistics given nid */
            inline NodeStat &stat( int nid ){
                return stats[ nid ];
            }
            /*! \brief initialize the model */
            inline void InitModel( void ){
                param.num_nodes = param.num_roots;
                nodes.resize( param.num_nodes );
                stats.resize( param.num_nodes );
                for( int i = 0; i < param.num_nodes; i ++ ){
                    nodes[i].set_leaf( 0.0f );
                    nodes[i].set_parent( -1 );
                }
            }
            /*! 
             * \brief load model from stream
             * \param fi input stream
             */
            inline void LoadModel( utils::IStream &fi ){
                utils::Assert( fi.Read( &param, sizeof(Param) ) > 0, "TreeModel" );
                nodes.resize( param.num_nodes ); stats.resize( param.num_nodes );
                utils::Assert( fi.Read( &nodes[0], sizeof(Node) * nodes.size() ) > 0, "TreeModel::Node" );
                utils::Assert( fi.Read( &stats[0], sizeof(NodeStat) * stats.size() ) > 0, "TreeModel::Node" );

                deleted_nodes.resize( 0 );
                for( int i = param.num_roots; i < param.num_nodes; i ++ ){
                    if( nodes[i].is_root() ) deleted_nodes.push_back( i );
                }
                utils::Assert( (int)deleted_nodes.size() == param.num_deleted, "number of deleted nodes do not match" );
            }
            /*! 
             * \brief save model to stream
             * \param fo output stream
             */
            inline void SaveModel( utils::IStream &fo ) const{
                utils::Assert( param.num_nodes == (int)nodes.size() );
                utils::Assert( param.num_nodes == (int)stats.size() );
                fo.Write( &param, sizeof(Param) );
                fo.Write( &nodes[0], sizeof(Node) * nodes.size() );
                fo.Write( &stats[0], sizeof(NodeStat) * nodes.size() );
            }
            /*! 
             * \brief add child nodes to node
             * \param nid node id to add childs
             */
            inline void AddChilds( int nid ){
                int pleft  = this->AllocNode();
                int pright = this->AllocNode();
                nodes[ nid ].cleft_  = pleft;
                nodes[ nid ].cright_ = pright;
                nodes[ nodes[ nid ].cleft()  ].set_parent( nid, true );
                nodes[ nodes[ nid ].cright() ].set_parent( nid, false );
            }
            /*! 
             * \brief only add a right child to a leaf node 
             * \param node id to add right child
             */
            inline void AddRightChild( int nid ){
                int pright = this->AllocNode();
                nodes[ nid ].right  = pright;
                nodes[ nodes[ nid ].right  ].set_parent( nid, false );
            }
            /*!
             * \brief get current depth
             * \param nid node id
             * \param pass_rchild whether right child is not counted in depth
             */
            inline int GetDepth( int nid, bool pass_rchild = false ) const{
                int depth = 0;
                while( !nodes[ nid ].is_root() ){
                    if( !pass_rchild || nodes[ nid ].is_left_child() ) depth ++;
                    nid = nodes[ nid ].parent();
                }
                return depth;
            }
            /*!
             * \brief get maximum depth
             * \param nid node id
             */
            inline int MaxDepth( int nid ) const{
                if( nodes[nid].is_leaf() ) return 0;
                return std::max( MaxDepth( nodes[nid].cleft() )+1, 
                                 MaxDepth( nodes[nid].cright() )+1 );
            }
            /*!
             * \brief get maximum depth
             */
            inline int MaxDepth( void ){
                int maxd = 0;
                for( int i = 0; i < param.num_roots; ++ i ){
                    maxd = std::max( maxd, MaxDepth( i ) );
                }
                return maxd;
            }
            /*! \brief number of extra nodes besides the root */
            inline int num_extra_nodes( void ) const {
                return param.num_nodes - param.num_roots - param.num_deleted;
            }
            /*! \brief dump model to text file  */
            inline void DumpModel( FILE *fo, const utils::FeatMap& fmap, bool with_stats ){
                this->Dump( 0, fo, fmap, 0, with_stats );
            }
        private:
            void Dump( int nid, FILE *fo, const utils::FeatMap& fmap, int depth, bool with_stats ){
                for( int  i = 0;  i < depth; ++ i ){
                    fprintf( fo, "\t" );
                }
                if( nodes[ nid ].is_leaf() ){
                    fprintf( fo, "%d:leaf=%f ", nid, nodes[ nid ].leaf_value() );
                    if( with_stats ){
                        stat( nid ).Print( fo, true );
                    }
                    fprintf( fo, "\n" );
                }else{
                    // right then left,
                    TSplitCond cond = nodes[ nid ].split_cond();
                    const unsigned split_index = nodes[ nid ].split_index();

                    if( split_index < fmap.size() ){
                        switch( fmap.type(split_index) ){
                        case utils::FeatMap::kIndicator:{
                            int nyes = nodes[ nid ].default_left()?nodes[nid].cright():nodes[nid].cleft();
                            fprintf( fo, "%d:[%s] yes=%d,no=%d", 
                                     nid, fmap.name( split_index ),
                                     nyes, nodes[nid].cdefault() );
                            break;                            
                        }
                        case utils::FeatMap::kInteger:{
                            fprintf( fo, "%d:[%s<%d] yes=%d,no=%d,missing=%d", 
                                     nid, fmap.name(split_index), int( float(cond)+1.0f), 
                                     nodes[ nid ].cleft(), nodes[ nid ].cright(),
                                     nodes[ nid ].cdefault() );
                            break;
                        }
                        case utils::FeatMap::kFloat:
                        case utils::FeatMap::kQuantitive:{
                            fprintf( fo, "%d:[%s<%f] yes=%d,no=%d,missing=%d", 
                                     nid, fmap.name(split_index), float(cond), 
                                     nodes[ nid ].cleft(), nodes[ nid ].cright(),
                                     nodes[ nid ].cdefault() );
                            break;
                        }
                        default: utils::Error("unknown fmap type");
                        }
                    }else{
                        fprintf( fo, "%d:[f%u<%f] yes=%d,no=%d,missing=%d", 
                                 nid, split_index, float(cond), 
                                 nodes[ nid ].cleft(), nodes[ nid ].cright(),
                                 nodes[ nid ].cdefault() );
                    }
                    if( with_stats ){
                        fprintf( fo, " ");
                        stat( nid ).Print( fo, false );
                    }
                    fprintf( fo, "\n" );
                    this->Dump( nodes[ nid ].cleft(), fo, fmap, depth+1, with_stats );
                    this->Dump( nodes[ nid ].cright(), fo, fmap, depth+1, with_stats );
                }                
            } 
        };
    };
    
    namespace booster{
        /*! \brief training parameters for regression tree */
        struct TreeParamTrain{
            // learning step size for a time
            float learning_rate;
            // minimum loss change required for a split
            float min_split_loss;
            // maximum depth of a tree
            int   max_depth;
            //----- the rest parameters are less important ----
            // minimum amount of hessian(weight) allowed in a child
            float min_child_weight;
            // weight decay parameter used to control leaf fitting
            float reg_lambda;
            // reg method
            int   reg_method;
            // default direction choice
            int   default_direction;
            // whether we want to do subsample
            float subsample;
            // whether to use layerwise aware regularization
            int   use_layerwise;
            // number of threads to be used for tree construction, if OpenMP is enabled, if equals 0, use system default
            int nthread;
            /*! \brief constructor */
            TreeParamTrain( void ){
                learning_rate = 0.3f;
                min_child_weight = 1.0f;
                max_depth = 6;
                reg_lambda = 1.0f;
                reg_method = 2;
                default_direction = 0;
                subsample = 1.0f;
                use_layerwise = 0;
                nthread = 0;
            }
            /*! 
             * \brief set parameters from outside 
             * \param name name of the parameter
             * \param val  value of the parameter
             */            
            inline void SetParam( const char *name, const char *val ){
                // sync-names 
                if( !strcmp( name, "gamma") )  min_split_loss = (float)atof( val );
                if( !strcmp( name, "eta") )    learning_rate  = (float)atof( val );
                if( !strcmp( name, "lambda") ) reg_lambda = (float)atof( val );
                // normal tree prameters
                if( !strcmp( name, "learning_rate") )     learning_rate = (float)atof( val );
                if( !strcmp( name, "min_child_weight") )  min_child_weight = (float)atof( val );
                if( !strcmp( name, "min_split_loss") )    min_split_loss = (float)atof( val );
                if( !strcmp( name, "max_depth") )         max_depth = atoi( val );           
                if( !strcmp( name, "reg_lambda") )        reg_lambda = (float)atof( val );
                if( !strcmp( name, "reg_method") )        reg_method = (float)atof( val );
                if( !strcmp( name, "subsample") )         subsample  = (float)atof( val );
                if( !strcmp( name, "use_layerwise") )     use_layerwise = atoi( val );
                if( !strcmp( name, "nthread") )           nthread = atoi( val );
                if( !strcmp( name, "default_direction") ) {
                    if( !strcmp( val, "learn") )  default_direction = 0;
                    if( !strcmp( val, "left") )   default_direction = 1;
                    if( !strcmp( val, "right") )  default_direction = 2;
                }
            }
        protected:
            // functions for L1 cost
            static inline double ThresholdL1( double w, double lambda ){
                if( w > +lambda ) return w - lambda;
                if( w < -lambda ) return w + lambda;
                return 0.0;
            }
            inline double CalcWeight( double sum_grad, double sum_hess )const{
                if( sum_hess < min_child_weight ){
                    return 0.0;
                }else{
                    switch( reg_method ){
                    case 1: return - ThresholdL1( sum_grad, reg_lambda ) / sum_hess;
                    case 2: return - sum_grad / ( sum_hess + reg_lambda );
                        // elstic net
                    case 3: return - ThresholdL1( sum_grad, 0.5 * reg_lambda ) / ( sum_hess + 0.5 * reg_lambda );
                    default: return - sum_grad / sum_hess;
                    }
                }
            }
        private:
            inline static double Sqr( double a ){
                return a * a;
            }
        public:
            // calculate the cost of loss function
            inline double CalcGain( double sum_grad, double sum_hess ) const{
                if( sum_hess < min_child_weight ){
                    return 0.0;
                }
                switch( reg_method ){
                case 1 : return Sqr( ThresholdL1( sum_grad, reg_lambda ) ) / sum_hess;
                case 2 : return Sqr( sum_grad ) / ( sum_hess + reg_lambda );
                    // elstic net
                case 3 : return Sqr( ThresholdL1( sum_grad, 0.5 * reg_lambda ) ) / ( sum_hess + 0.5 * reg_lambda );
                default: return Sqr( sum_grad ) / sum_hess;
                }        
            }
            // KEY:layerwise
            // calculate cost of root
            inline double CalcRootGain( double sum_grad, double sum_hess ) const{
                if( use_layerwise == 0 ) return this->CalcGain( sum_grad, sum_hess );
                else return 0.0;
            }
            // KEY:layerwise
            // calculate the cost after split
            // base_weight: the base_weight of parent           
            inline double CalcGain( double sum_grad, double sum_hess, double base_weight ) const{
                if( use_layerwise == 0 ) return this->CalcGain( sum_grad, sum_hess );
                else return this->CalcGain( sum_grad + sum_hess * base_weight, sum_hess );
            }
            // calculate the weight of leaf
            inline double CalcWeight( double sum_grad, double sum_hess, double parent_base_weight )const{
                if( use_layerwise == 0 ) return CalcWeight( sum_grad, sum_hess );
                else return parent_base_weight + CalcWeight( sum_grad + parent_base_weight * sum_hess, sum_hess );
            }           
            /*! \brief whether need forward small to big search: default right */
            inline bool need_forward_search( void ) const{
                return this->default_direction != 1;
            }
            /*! \brief whether need forward big to small search: default left */
            inline bool need_backward_search( void ) const{
                return this->default_direction != 2;
            }
            /*! \brief given the loss change, whether we need to invode prunning */
            inline bool need_prune( double loss_chg, int depth ) const{
                return loss_chg < this->min_split_loss;
            }
            /*! \brief whether we can split with current hessian */
            inline bool cannot_split( double sum_hess, int depth ) const{
                return sum_hess < this->min_child_weight * 2.0; 
            }
        };
    };
    
    namespace booster{
        /*! \brief node statistics used in regression tree */
        struct RTreeNodeStat{
            /*! \brief loss chg caused by current split */
            float loss_chg;
            /*! \brief sum of hessian values, used to measure coverage of data */
            float sum_hess;
            /*! \brief weight of current node */
            float base_weight;
            /*! \brief number of child that is leaf node known up to now */
            int   leaf_child_cnt;
            /*! \brief print information of current stats to fo */
            inline void Print( FILE *fo, bool is_leaf ) const{
                if( !is_leaf ){
                    fprintf( fo, "gain=%f,cover=%f", loss_chg, sum_hess );
                }else{
                    fprintf( fo, "cover=%f", sum_hess );
                }
            }
        };
        /*! \brief most comment structure of regression tree */
        class RegTree: public TreeModel<bst_float,RTreeNodeStat>{
        };
    };
};
#endif
