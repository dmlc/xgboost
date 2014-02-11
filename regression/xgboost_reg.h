#ifndef _XGBOOST_REG_H_
#define _XGBOOST_REG_H_
/*!
 * \file xgboost_reg.h
 * \brief class for gradient boosted regression
 * \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <cmath>
#include "xgboost_regdata.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost{
    namespace regression{
        /*! \brief class for gradient boosted regression */
        class RegBoostLearner{            
        public:
            /*! 
             * \brief a regression booter associated with training and evaluating data 
             * \param train pointer to the training data
             * \param evals array of evaluating data
             * \param evname name of evaluation data, used print statistics
             */
            RegBoostLearner( const DMatrix *train,
                             std::vector<const DMatrix *> evals,
                             std::vector<std::string> evname ){
                this->train_ = train;
                this->evals_ = evals;
                this->evname_ = evname;                
                //TODO: assign buffer index
            }
            /*! 
             * \brief set parameters from outside 
             * \param name name of the parameter
             * \param val  value of the parameter
             */
            inline void SetParam( const char *name, const char *val ){
                mparam.SetParam( name, val );
                base_model.SetParam( name, val );
            }
            /*!
             * \brief initialize solver before training, called before training
             * this function is reserved for solver to allocate necessary space and do other preparation 
             */
            inline void InitTrainer( void ){
                base_model.InitTrainer();
                mparam.AdjustBase();
            } 
            /*! 
             * \brief load model from stream
             * \param fi input stream
             */          
            inline void LoadModel( utils::IStream &fi ){
                utils::Assert( fi.Read( &mparam, sizeof(ModelParam) ) != 0 );
                base_model.LoadModel( fi );
            }
            /*! 
             * \brief save model to stream
             * \param fo output stream
             */
            inline void SaveModel( utils::IStream &fo ) const{
                fo.Write( &mparam, sizeof(ModelParam) );
                base_model.SaveModel( fo );	
            } 
            /*! 
             * \brief update the model for one iteration
             */           
            inline void UpdateOneIter( void ){
                //TODO
            }
            /*! \brief predict the results, given data */
            inline void Predict( std::vector<float> &preds, const DMatrix &data ){
                //TODO
            }
        private:
            /*! \brief training parameter for regression */
            struct ModelParam{
                /* \brief global bias */
                float base_score;
                /* \brief type of loss function */
                int   loss_type;
                ModelParam( void ){
                    base_score = 0.5f;
                    loss_type  = 0;
                }
                /*! 
                 * \brief set parameters from outside 
                 * \param name name of the parameter
                 * \param val  value of the parameter
                 */
                inline void SetParam( const char *name, const char *val ){
                    if( !strcmp("base_score", name ) )  base_score = (float)atof( val );
                    if( !strcmp("loss_type", name ) )   loss_type = atoi( val );
                }
                /*! 
                 * \brief adjust base_score
                 */                
                inline void AdjustBase( void ){
                    if( loss_type == 1 ){
                        utils::Assert( base_score > 0.0f && base_score < 1.0f, "sigmoid range constrain" );
                        base_score = - logf( 1.0f / base_score - 1.0f );
                    }
                }
                /*! 
                 * \brief calculate first order gradient of loss, given transformed prediction
                 * \param predt transformed prediction
                 * \param label true label
                 * \return first order gradient
                 */
                inline float FirstOrderGradient( float predt, float label ) const{
                    switch( loss_type ){                        
                    case 0: return predt - label;
                    case 1: return predt - label;
                    default: utils::Error("unknown loss_type"); return 0.0f;
                    }
                }
                /*! 
                 * \brief calculate second order gradient of loss, given transformed prediction
                 * \param predt transformed prediction
                 * \param label true label
                 * \return second order gradient
                 */
                inline float SecondOrderGradient( float predt, float label ) const{
                    switch( loss_type ){                        
                    case 0: return 1.0f;
                    case 1: return predt * ( 1 - predt );
                    default: utils::Error("unknown loss_type"); return 0.0f;
                    }
                }
                /*! 
                 * \brief transform the linear sum to prediction 
                 * \param x linear sum of boosting ensemble
                 * \return transformed prediction
                 */
                inline float PredTransform( float x ){
                    switch( loss_type ){                        
                    case 0: return x;
                    case 1: return 1.0f/(1.0f + expf(-x));
                    default: utils::Error("unknown loss_type"); return 0.0f;
                    }
                }
            };            
        private:            
            booster::GBMBaseModel base_model;
            ModelParam   mparam;
            const DMatrix *train_;
            std::vector<const DMatrix *> evals_;
            std::vector<std::string> evname_;
        };
    };
};

#endif
