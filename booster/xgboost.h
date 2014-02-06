#ifndef _XGBOOST_H_
#define _XGBOOST_H_
/*!
 * \file xgboost.h
 * \brief the general gradient boosting interface
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <vector>
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"
#include "xgboost_data.h"

/*! \brief namespace for xboost package */
namespace xgboost{
    namespace booster{
        /*! \brief interface of a gradient boosting learner */
        class IBooster{
        public:
            // interface for model setting and loading
            // calling procedure:
            //  (1) booster->SetParam to setting necessary parameters
            //  (2) if it is first time usage of the model: call booster->
            //   if new model to be trained, trainer->init_trainer
            //   elseif just to load from file, trainer->load_model
            //   trainer->do_boost
            //   trainer->save_model
            /*! 
             * \brief set parameters from outside 
             * \param name name of the parameter
             * \param val  value of the parameter
             */
            virtual void SetParam( const char *name, const char *val ) = 0;
            /*! 
             * \brief load model from stream
             * \param fi input stream
             */
            virtual void LoadModel( utils::IStream &fi ) = 0;
            /*! 
             * \brief save model to stream
             * \param fo output stream
             */
            virtual void SaveModel( utils::IStream &fo ) const = 0;
            /*!
             * \brief initialize solver before training, called before training
             * this function is reserved for solver to allocate necessary space and do other preparations 
             */        
            virtual void InitModel( void ) = 0;
        public:
            /*! 
             * \brief do gradient boost training for one step, using the information given
             * \param grad first order gradient of each instance
             * \param hess second order gradient of each instance
             * \param feats features of each instance
             * \param root_index pre-partitioned root index of each instance, 
             *          root_index.size() can be 0 which indicates that no pre-partition involved
             */
            virtual void DoBoost( std::vector<float> &grad,
                                  std::vector<float> &hess,
                                  const FMatrixS::Image &feats,
                                  const std::vector<unsigned> &root_index ) = 0;
            /*! 
             * \brief predict values for given sparse feature
             *   NOTE: in tree implementation, this is not threadsafe
             * \param feat vector in sparse format
             * \param rid root id of current instance, default = 0
             * \return prediction 
             */        
            virtual float Predict( const FMatrixS::Line &feat, unsigned rid = 0 ){
                utils::Error( "not implemented" );
                return 0.0f;
            }
            /*! 
             * \brief predict values for given dense feature
             * \param feat feature vector in dense format
             * \param funknown indicator that the feature is missing
             * \param rid root id of current instance, default = 0
             * \return prediction 
             */                
            virtual float Predict( const std::vector<float> &feat, 
                                   const std::vector<bool>  &funknown,
                                   unsigned rid = 0 ){
                utils::Error( "not implemented" );            
                return 0.0f;
            }
            /*! 
             * \brief print information
             * \param fo output stream 
             */        
            virtual void PrintInfo( FILE *fo ){}
        public:
            virtual ~IBooster( void ){}
        };    
    };
};

#endif
