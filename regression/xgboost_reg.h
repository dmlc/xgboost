#ifndef _XGBOOST_REG_H_
#define _XGBOOST_REG_H_
/*!
* \file xgboost_reg.h
* \brief class for gradient boosted regression
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "xgboost_regdata.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost{
    namespace regression{
        /*! \brief class for gradient boosted regression */
        class RegBoostLearner{            
        public:
            /*! \brief constructor */
            RegBoostLearner( void ){
                silent = 0;            
            }
            /*! 
            * \brief a regression booter associated with training and evaluating data 
            * \param train pointer to the training data
            * \param evals array of evaluating data
            * \param evname name of evaluation data, used print statistics
            */
            RegBoostLearner( const DMatrix *train,
                             const std::vector<DMatrix *> &evals,
                             const std::vector<std::string> &evname ){
                silent = 0;
                this->SetData(train,evals,evname);
            }

            /*! 
            * \brief associate regression booster with training and evaluating data 
            * \param train pointer to the training data
            * \param evals array of evaluating data
            * \param evname name of evaluation data, used print statistics
            */
            inline void SetData( const DMatrix *train,
                                 const std::vector<DMatrix *> &evals,
                                 const std::vector<std::string> &evname ){
                this->train_ = train;
                this->evals_ = evals;
                this->evname_ = evname; 
                //assign buffer index
                unsigned buffer_size = static_cast<unsigned>( train->Size() );
                
                for( size_t i = 0; i < evals.size(); ++ i ){
                    buffer_size += static_cast<unsigned>( evals[i]->Size() );
                }
                char snum_pbuffer[25];
                sprintf( snum_pbuffer, "%u", buffer_size );
                if( !silent ){
                    printf( "buffer_size=%u\n", buffer_size );
                }
                base_model.SetParam( "num_pbuffer",snum_pbuffer );
            }
            /*! 
            * \brief set parameters from outside 
            * \param name name of the parameter
            * \param val  value of the parameter
            */
            inline void SetParam( const char *name, const char *val ){
                if( !strcmp( name, "silent") ) silent = atoi( val );
                mparam.SetParam( name, val );
                base_model.SetParam( name, val );
            }
            /*!
            * \brief initialize solver before training, called before training
            * this function is reserved for solver to allocate necessary space and do other preparation 
            */
            inline void InitTrainer( void ){
                base_model.InitTrainer();
            } 
            /*!
            * \brief initialize the current data storage for model, if the model is used first time, call this function
            */
            inline void InitModel( void ){
                base_model.InitModel();
                mparam.AdjustBase();
            }
            /*! 
            * \brief load model from stream
            * \param fi input stream
            */          
            inline void LoadModel( utils::IStream &fi ){
                base_model.LoadModel( fi );
                utils::Assert( fi.Read( &mparam, sizeof(ModelParam) ) != 0 );
            }
            /*! 
             * \brief DumpModel
             * \param fo text file 
             */            
            inline void DumpModel( FILE *fo ){
                base_model.DumpModel( fo );
            }
            /*! 
            * \brief save model to stream
            * \param fo output stream
            */
            inline void SaveModel( utils::IStream &fo ) const{
                base_model.SaveModel( fo );	
                fo.Write( &mparam, sizeof(ModelParam) );
            } 
            /*! 
             * \brief update the model for one iteration
             * \param iteration iteration number
             */
            inline void UpdateOneIter( int iter ){
                std::vector<float> grad, hess, preds;
                this->Predict( preds, *train_, 0 );
                this->GetGradient( preds, train_->labels, grad, hess );

                std::vector<unsigned> root_index;
                booster::FMatrixS::Image train_image( train_->data );                
                base_model.DoBoost(grad,hess,train_image,root_index);                
            }
            /*! 
             * \brief evaluate the model for specific iteration
             * \param iter iteration number
             * \param fo file to output log
             */            
            inline void EvalOneIter( int iter, FILE *fo = stderr ){
                std::vector<float> preds;
                fprintf( fo, "[%d]", iter );
                int buffer_offset = static_cast<int>( train_->Size() );

                for(size_t i = 0; i < evals_.size();i++){
                    this->Predict(preds, *evals_[i], buffer_offset);
                    this->Eval( fo, evname_[i].c_str(), preds, (*evals_[i]).labels );
                    buffer_offset += static_cast<int>( evals_[i]->Size() );
                }
                fprintf( fo,"\n" );
            }

            /*! \brief get prediction, without buffering */
            inline void Predict( std::vector<float> &preds, const DMatrix &data ){
                preds.resize( data.Size() );
                for( size_t j = 0; j < data.Size(); j++ ){
                    preds[j] = mparam.PredTransform
                        ( mparam.base_score + base_model.Predict( data.data[j], -1 ) );
                }
            }
        private:
            /*! \brief print evaluation results */
            inline void Eval( FILE *fo, const char *evname,
                              const std::vector<float> &preds, 
                              const std::vector<float> &labels ){
                const float loss = mparam.Loss( preds, labels );
                fprintf( fo, "\t%s:%f", evname, loss );
            }
            /*! \brief get the transformed predictions, given data */
            inline void Predict( std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset ){
                preds.resize( data.Size() );
                for( size_t j = 0; j < data.Size(); j++ ){
                    preds[j] = mparam.PredTransform
                        ( mparam.base_score + base_model.Predict( data.data[j], buffer_offset + j ) );
                }
            }

            /*! \brief get the first order and second order gradient, given the transformed predictions and labels */
            inline void GetGradient( const std::vector<float> &preds, 
                                     const std::vector<float> &labels, 
                                     std::vector<float> &grad,
                                     std::vector<float> &hess ){
                grad.clear(); hess.clear();
                for( size_t j = 0; j < preds.size(); j++ ){
                    grad.push_back( mparam.FirstOrderGradient (preds[j],labels[j]) );
                    hess.push_back( mparam.SecondOrderGradient(preds[j],labels[j]) );
                }
            }

        private:
            enum LossType{
                kLinearSquare = 0,
                kLogisticNeglik = 1,
                kLogisticClassify = 2
            };

            /*! \brief training parameter for regression */
            struct ModelParam{
                /* \brief global bias */
                float base_score;
                /* \brief type of loss function */
                int loss_type;
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
                * \brief transform the linear sum to prediction 
                * \param x linear sum of boosting ensemble
                * \return transformed prediction
                */
                inline float PredTransform( float x ){
                    switch( loss_type ){                        
                    case kLinearSquare: return x;
                    case kLogisticClassify:
                    case kLogisticNeglik: return 1.0f/(1.0f + expf(-x));
                    default: utils::Error("unknown loss_type"); return 0.0f;
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
                    case kLinearSquare: return predt - label;
                    case kLogisticClassify:
                    case kLogisticNeglik: return predt - label;
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
                    case kLinearSquare: return 1.0f;
                    case kLogisticClassify:
                    case kLogisticNeglik: return predt * ( 1 - predt );
                    default: utils::Error("unknown loss_type"); return 0.0f;
                    }
                }

                /*!
                * \brief calculating the loss, given the predictions, labels and the loss type
                * \param preds the given predictions
                * \param labels the given labels
                * \return the specified loss
                */
                inline float Loss(const std::vector<float> &preds, const std::vector<float> &labels) const{
                    switch( loss_type ){
                    case kLinearSquare: return SquareLoss(preds,labels);
                    case kLogisticNeglik: return NegLoglikelihoodLoss(preds,labels);
                    case kLogisticClassify: return ClassificationError(preds, labels);
                    default: utils::Error("unknown loss_type"); return 0.0f;
                    }
                }

                /*!
                * \brief calculating the square loss, given the predictions and labels
                * \param preds the given predictions
                * \param labels the given labels
                * \return the summation of square loss
                */
                inline float SquareLoss(const std::vector<float> &preds, const std::vector<float> &labels) const{
                    float ans = 0.0;
                    for(size_t i = 0; i < preds.size(); i++){
                        float dif = preds[i] - labels[i];
                        ans += dif * dif;
                    }
                    return ans;
                }

                /*!
                * \brief calculating the square loss, given the predictions and labels
                * \param preds the given predictions
                * \param labels the given labels
                * \return the summation of square loss
                */
                inline float NegLoglikelihoodLoss(const std::vector<float> &preds, const std::vector<float> &labels) const{
                    float ans = 0.0;
                    for(size_t i = 0; i < preds.size(); i++)
                        ans -= labels[i] * logf(preds[i]) + ( 1 - labels[i] ) * logf(1 - preds[i]);
                    return ans;
                }

                /*!
                * \brief calculating the ClassificationError  loss, given the predictions and labels
                * \param preds the given predictions
                * \param labels the given labels
                * \return the summation of square loss
                */
                inline float ClassificationError(const std::vector<float> &preds, const std::vector<float> &labels) const{
                    int nerr = 0;
                    for(size_t i = 0; i < preds.size(); i++){
                        if( preds[i] > 0.5f ){
                            if( labels[i] < 0.5f ) nerr ++;
                        }else{
                            if( labels[i] > 0.5f ) nerr ++;
                        }
                    }
                    return (float)nerr/preds.size();
                }                
            };
        private:
            int silent;
            booster::GBMBaseModel base_model;
            ModelParam   mparam;
            const DMatrix *train_;
            std::vector<DMatrix *> evals_;
            std::vector<std::string> evname_;
            std::vector<unsigned> buffer_index_;
        };
    }
};

#endif
