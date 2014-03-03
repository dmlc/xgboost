#ifndef XGBOOST_GBMBASE_H
#define XGBOOST_GBMBASE_H

#include <omp.h>
#include <cstring>
#include "xgboost.h"
#include "../utils/xgboost_config.h"
/*!
 * \file xgboost_gbmbase.h
 * \brief a base model class, 
 *        that assembles the ensembles of booster together and do model update
 *        this class can be used as base code to create booster variants 
 *
 *        The detailed implementation of boosters should start by using the class
 *        provided by this file
 *        
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
namespace xgboost{
    namespace booster{
        /*!
         * \brief a base model class, 
         *        that assembles the ensembles of booster together and provide single routines to do prediction buffer and update
         *        this class can be used as base code to create booster variants 
         *         *
         *  relation to xgboost.h:
         *    (1) xgboost.h provides a interface to a single booster(e.g. a single regression tree )
         *        while GBMBaseModel builds upon IBooster to build a class that 
         *        ensembls the boosters together;
         *    (2) GBMBaseModel provides prediction buffering scheme to speedup training;
         *    (3) Summary: GBMBaseModel is a standard wrapper for boosting ensembles;
         *
         *  Usage of this class, the number index gives calling dependencies:
         *    (1) model.SetParam to set the parameters
         *    (2) model.LoadModel to load old models or model.InitModel to create a new model
         *    (3) model.InitTrainer before calling model.Predict and model.DoBoost
         *    (4) model.Predict to get predictions given a instance
         *    (4) model.DoBoost to update the ensembles, add new booster to the model
         *    (4) model.SaveModel to save learned results 
         *
         *  Bufferring: each instance comes with a buffer_index in Predict. 
         *              when param.num_pbuffer != 0, a unique buffer index can be 
         *              assigned to each instance to buffer previous results of boosters,
         *              this helps to speedup training, so consider assign buffer_index 
         *              for each training instances, if buffer_index = -1, the code
         *              recalculate things from scratch and will still works correctly
         */
        class GBMBaseModel{
        public:
            /*! \brief model parameters */
            struct Param{
                /*! \brief number of boosters */
                int num_boosters;
                /*! \brief type of tree used */
                int booster_type;
                /*! \brief number of root: default 0, means single tree */
                int num_roots;
                /*! \brief number of features to be used by boosters */
                int num_feature;
                /*! \brief size of predicton buffer allocated for buffering boosting computation */
                int num_pbuffer;
                /*! 
                 * \brief whether we repeatly update a single booster each round: default 0
                 *        set to 1 for linear booster, so that regularization term can be considered
                 */
                int do_reboost;
                /*! \brief reserved parameters */
                int reserved[ 32 ];
                /*! \brief constructor */
                Param( void ){
                    num_boosters = 0; 
                    booster_type = 0;
                    num_roots = num_feature = 0;                    
                    do_reboost = 0;
                    num_pbuffer = 0;
                    memset( reserved, 0, sizeof( reserved ) );     
                }
                /*! 
                 * \brief set parameters from outside 
                 * \param name name of the parameter
                 * \param val  value of the parameter
                 */
                inline void SetParam( const char *name, const char *val ){
                    if( !strcmp("booster_type", name ) ){
                        booster_type = atoi( val );
                        // linear boost automatically set do reboost
                        if( booster_type == 1 ) do_reboost = 1;
                    }
                    if( !strcmp("num_pbuffer", name ) )      num_pbuffer = atoi( val );
                    if( !strcmp("do_reboost", name ) )       do_reboost  = atoi( val );
                    if( !strcmp("bst:num_roots", name ) )    num_roots = atoi( val );
                    if( !strcmp("bst:num_feature", name ) )  num_feature = atoi( val );
                }
            };
        public:
            /*! \brief model parameters */ 
            Param param;
        public:
            /*! \brief number of thread used */
            GBMBaseModel( void ){
                this->nthread = 1;
            }
            /*! \brief destructor */
            virtual ~GBMBaseModel( void ){
                this->FreeSpace();
            }
            /*! 
             * \brief set parameters from outside 
             * \param name name of the parameter
             * \param val  value of the parameter
             */
            inline void SetParam( const char *name, const char *val ){
                if( !strncmp( name, "bst:", 4 ) ){
                    cfg.PushBack( name + 4, val );
                }
                if( !strcmp( name, "silent") ){
                    cfg.PushBack( name, val );
                }
                if( !strcmp( name, "nthread") ) nthread = atoi( val );
                if( boosters.size() == 0 ) param.SetParam( name, val );
            }
            /*! 
             * \brief load model from stream
             * \param fi input stream
             */
            inline void LoadModel( utils::IStream &fi ){
                if( boosters.size() != 0 ) this->FreeSpace();
                utils::Assert( fi.Read( &param, sizeof(Param) ) != 0 );
                boosters.resize( param.num_boosters );
                for( size_t i = 0; i < boosters.size(); i ++ ){
                    boosters[ i ] = booster::CreateBooster( param.booster_type );
                    boosters[ i ]->LoadModel( fi );
                }
                {// load info 
                    booster_info.resize( param.num_boosters );
                    if( param.num_boosters != 0 ){
                        utils::Assert( fi.Read( &booster_info[0], sizeof(int)*param.num_boosters ) != 0 );
                    }
                }
                if( param.num_pbuffer != 0 ){
                    pred_buffer.resize ( param.num_pbuffer );
                    pred_counter.resize( param.num_pbuffer );
                    utils::Assert( fi.Read( &pred_buffer[0] , pred_buffer.size()*sizeof(float) ) != 0 );
                    utils::Assert( fi.Read( &pred_counter[0], pred_counter.size()*sizeof(unsigned) ) != 0 );
                }
            }
            /*! 
             * \brief save model to stream
             * \param fo output stream
             */
            inline void SaveModel( utils::IStream &fo ) const {
                utils::Assert( param.num_boosters == (int)boosters.size() );
                fo.Write( &param, sizeof(Param) );
                for( size_t i = 0; i < boosters.size(); i ++ ){
                    boosters[ i ]->SaveModel( fo ); 
                }
                if( booster_info.size() != 0 ){
                    fo.Write( &booster_info[0], sizeof(int) * booster_info.size() );
                }
                if( param.num_pbuffer != 0 ){
                    fo.Write( &pred_buffer[0] , pred_buffer.size()*sizeof(float) );
                    fo.Write( &pred_counter[0], pred_counter.size()*sizeof(unsigned) );
                }
            }
            /*!
             * \brief initialize the current data storage for model, if the model is used first time, call this function
             */
            inline void InitModel( void ){
                pred_buffer.clear(); pred_counter.clear();
                pred_buffer.resize ( param.num_pbuffer, 0.0 );
                pred_counter.resize( param.num_pbuffer, 0 );
                utils::Assert( param.num_boosters == 0 );
                utils::Assert( boosters.size() == 0 );
            }
            /*!
             * \brief initialize solver before training, called before training
             * this function is reserved for solver to allocate necessary space and do other preparation 
             */            
            inline void InitTrainer( void ){
                if( nthread != 0 ){
                    omp_set_num_threads( nthread );
                }
                // make sure all the boosters get the latest parameters
                for( size_t i = 0; i < this->boosters.size(); i ++ ){
                    this->ConfigBooster( this->boosters[i] );
                }
            }
            /*! 
             * \brief DumpModel
             * \param fo text file 
             * \param fmap feature map that may help give interpretations of feature
             * \param with_stats whether print statistics
             */            
            inline void DumpModel( FILE *fo, const utils::FeatMap& fmap, bool with_stats ){
                for( size_t i = 0; i < boosters.size(); i ++ ){
                    fprintf( fo, "booster[%d]\n", (int)i );
                    boosters[i]->DumpModel( fo, fmap, with_stats );
                }
            }
            /*! 
             * \brief Dump path of all trees
             * \param fo text file 
             * \param data input data
             */
            inline void DumpPath( FILE *fo, const FMatrixS &data ){
                for( size_t i = 0; i < data.NumRow(); ++ i ){
                    for( size_t j = 0; j < boosters.size(); ++ j ){
                        if( j != 0 ) fprintf( fo, "\t" );
                        std::vector<int> path;
                        boosters[j]->PredPath( path, data[i] );
                        fprintf( fo, "%d", path[0] );
                        for( size_t k = 1; k < path.size(); ++ k ){
                            fprintf( fo, ",%d", path[k] );
                        }
                    }
                    fprintf( fo, "\n" );
                }
            }
        public:
            /*! 
             * \brief do gradient boost training for one step, using the information given
             *        Note: content of grad and hess can change after DoBoost
             * \param grad first order gradient of each instance
             * \param hess second order gradient of each instance
             * \param feats features of each instance
             * \param root_index pre-partitioned root index of each instance, 
             *          root_index.size() can be 0 which indicates that no pre-partition involved
             */
            inline void DoBoost( std::vector<float> &grad,
                                 std::vector<float> &hess,
                                 const booster::FMatrixS &feats,
                                 const std::vector<unsigned> &root_index ) {
                booster::IBooster *bst = this->GetUpdateBooster();
                bst->DoBoost( grad, hess, feats, root_index );
            }            
            /*! 
             * \brief predict values for given sparse feature vector
             *   NOTE: in tree implementation, this is not threadsafe
             * \param feat vector in sparse format
             * \param buffer_index the buffer index of the current feature line, default -1 means no buffer assigned
             * \param rid root id of current instance, default = 0
             * \return prediction 
             */        
            virtual float Predict( const booster::FMatrixS::Line &feat, int buffer_index = -1, unsigned rid = 0 ){
                size_t istart = 0;
                float  psum = 0.0f;

                // load buffered results if any
                if( param.do_reboost == 0 && buffer_index >= 0 ){
                    utils::Assert( buffer_index < param.num_pbuffer, "buffer index exceed num_pbuffer" );
                    istart = this->pred_counter[ buffer_index ];
                    psum   = this->pred_buffer [ buffer_index ];
                }
            
                for( size_t i = istart; i < this->boosters.size(); i ++ ){
                    psum += this->boosters[ i ]->Predict( feat, rid );
                }
                
                // updated the buffered results
                if( param.do_reboost == 0 && buffer_index >= 0 ){
                    this->pred_counter[ buffer_index ] = static_cast<unsigned>( boosters.size() );
                    this->pred_buffer [ buffer_index ] = psum;
                }
                return psum;
            }
            /*! 
             * \brief predict values for given dense feature vector
             * \param feat feature vector in dense format
             * \param funknown indicator that the feature is missing
             * \param buffer_index the buffer index of the current feature line, default -1 means no buffer assigned
             * \param rid root id of current instance, default = 0
             * \return prediction 
             */                
            virtual float Predict( const std::vector<float> &feat, 
                                   const std::vector<bool>  &funknown,
                                   int buffer_index = -1,
                                   unsigned rid = 0 ){
                size_t istart = 0;
                float  psum = 0.0f;

                // load buffered results if any
                if( param.do_reboost == 0 && buffer_index >= 0 ){
                    utils::Assert( buffer_index < param.num_pbuffer, 
                                   "buffer index exceed num_pbuffer" );
                    istart = this->pred_counter[ buffer_index ];
                    psum   = this->pred_buffer [ buffer_index ];
                }
            
                for( size_t i = istart; i < this->boosters.size(); i ++ ){
                    psum += this->boosters[ i ]->Predict( feat, funknown, rid );
                }
                
                // updated the buffered results
                if( param.do_reboost == 0 && buffer_index >= 0 ){
                    this->pred_counter[ buffer_index ] = static_cast<unsigned>( boosters.size() );
                    this->pred_buffer [ buffer_index ] = psum;
                }
                return psum;
            }
            //-----------non public fields afterwards-------------
        protected:
            /*! \brief free space of the model */
            inline void FreeSpace( void ){
                for( size_t i = 0; i < boosters.size(); i ++ ){
                    delete boosters[i];
                }
                boosters.clear(); booster_info.clear(); param.num_boosters = 0; 
            }
            /*! \brief configure a booster */
            inline void ConfigBooster( booster::IBooster *bst ){
                cfg.BeforeFirst();
                while( cfg.Next() ){
                    bst->SetParam( cfg.name(), cfg.val() );
                }
            }
            /*! 
             * \brief get a booster to update 
             * \return the booster created
             */
            inline booster::IBooster *GetUpdateBooster( void ){
                if( param.do_reboost == 0 || boosters.size() == 0 ){
                    param.num_boosters += 1;
                    boosters.push_back( booster::CreateBooster( param.booster_type ) );
                    booster_info.push_back( 0 );
                    this->ConfigBooster( boosters.back() );
                    boosters.back()->InitModel();                    
                }else{
                    this->ConfigBooster( boosters.back() );
                }
                return boosters.back();
            }
        protected:
            /*! \brief number of OpenMP threads */
            int nthread;
            /*! \brief component boosters */ 
            std::vector<booster::IBooster*> boosters;
            /*! \brief some information indicator of the booster, reserved */ 
            std::vector<int> booster_info;
            /*! \brief prediction buffer */ 
            std::vector<float>    pred_buffer;
            /*! \brief prediction buffer counter, record the progress so fart of the buffer */ 
            std::vector<unsigned> pred_counter;
            /*! \brief configurations saved for each booster */
            utils::ConfigSaver cfg;
        };
    };
};
#endif
