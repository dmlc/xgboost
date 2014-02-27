#ifndef _XGBOOST_H_
#define _XGBOOST_H_
/*!
 * \file xgboost.h
 * \brief the general gradient boosting interface
 * 
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <vector>
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"
#include "../utils/xgboost_config.h"
#include "xgboost_data.h"

/*! \brief namespace for xboost package */
namespace xgboost{
    /*! \brief namespace for boosters */
    namespace booster{
        /*! 
         * \brief interface of a gradient boosting learner 
         */
        class IBooster{
        public:
            // interface for model setting and loading
            // calling procedure:
            //  (1) booster->SetParam to setting necessary parameters
            //  (2) if it is first time usage of the model: 
            //          call booster->InitModel
            //      else: 
            //          call booster->LoadModel
            //  (3) booster->DoBoost to update the model
            //  (4) booster->Predict to get new prediction
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
             * this function is reserved for solver to allocate necessary space and do other preparation 
             */
            virtual void InitModel( void ) = 0;
        public:
            /*! 
             * \brief do gradient boost training for one step, using the information given, 
             *        Note: content of grad and hess can change after DoBoost
             * \param grad first order gradient of each instance
             * \param hess second order gradient of each instance
             * \param feats features of each instance
             * \param root_index pre-partitioned root index of each instance, 
             *          root_index.size() can be 0 which indicates that no pre-partition involved
             */
            virtual void DoBoost( std::vector<float> &grad,
                                  std::vector<float> &hess,
                                  const FMatrixS &feats,
                                  const std::vector<unsigned> &root_index ) = 0;
            /*! 
             * \brief predict the path ids along a trees, for given sparse feature vector. When booster is a tree
             * \param path the result of path
             * \param rid root id of current instance, default = 0
             */
            virtual void PredPath( std::vector<int> &path, const FMatrixS::Line &feat, unsigned rid = 0 ){
                utils::Error( "not implemented" );
            }
            /*! 
             * \brief predict values for given sparse feature vector
             *   NOTE: in tree implementation, this is not threadsafe, used dense version to ensure threadsafety
             * \param feat vector in sparse format
             * \param rid root id of current instance, default = 0
             * \return prediction 
             */        
            virtual float Predict( const FMatrixS::Line &feat, unsigned rid = 0 ){
                utils::Error( "not implemented" );
                return 0.0f;
            }
            /*! 
             * \brief predict values for given dense feature vector
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
            /*! 
             * \brief dump model into text file
             * \param fo output stream 
             */        
            virtual void DumpModel( FILE *fo ){
                utils::Error( "not implemented" );                
            }
        public:
            /*! \brief virtual destructor */
            virtual ~IBooster( void ){}
        };    
    };
};

namespace xgboost{
    namespace booster{
        /*! 
         * \brief create a gradient booster, given type of booster
         * \param booster_type type of gradient booster, can be used to specify implements
         * \return the pointer to the gradient booster created
         */
        IBooster *CreateBooster( int booster_type );
    };
};
#endif
