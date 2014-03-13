#ifndef XGBOOST_REG_H
#define XGBOOST_REG_H
/*!
* \file xgboost_reg.h
* \brief class for gradient boosted regression
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "xgboost_reg_data.h"
#include "xgboost_reg_eval.h"
#include "../utils/xgboost_omp.h"
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
                // estimate feature bound
                int num_feature = (int)(train->data.NumCol());
                // assign buffer index
                unsigned buffer_size = static_cast<unsigned>( train->Size() );
                
                for( size_t i = 0; i < evals.size(); ++ i ){
                    buffer_size += static_cast<unsigned>( evals[i]->Size() );
                    num_feature = std::max( num_feature, (int)(evals[i]->data.NumCol()) );
                }

                char str_temp[25];
                if( num_feature > mparam.num_feature ){
                    mparam.num_feature = num_feature;
                    sprintf( str_temp, "%d", num_feature );
                    base_gbm.SetParam( "bst:num_feature", str_temp );
                }
                
                sprintf( str_temp, "%u", buffer_size );
                base_gbm.SetParam( "num_pbuffer", str_temp );
                if( !silent ){
                    printf( "buffer_size=%u\n", buffer_size );
                }
                
                // set eval_preds tmp sapce
                this->eval_preds_.resize( evals.size(), std::vector<float>() );
            }
            /*! 
            * \brief set parameters from outside 
            * \param name name of the parameter
            * \param val  value of the parameter
            */
            inline void SetParam( const char *name, const char *val ){
                if( !strcmp( name, "silent") )  silent = atoi( val );
                if( !strcmp( name, "eval_metric") )  evaluator_.AddEval( val );                
                mparam.SetParam( name, val );
                base_gbm.SetParam( name, val );
            }
            /*!
            * \brief initialize solver before training, called before training
            * this function is reserved for solver to allocate necessary space and do other preparation 
            */
            inline void InitTrainer( void ){
                base_gbm.InitTrainer();
                if( mparam.loss_type == kLogisticClassify ){
                    evaluator_.AddEval( "error" );
                }else{
                    evaluator_.AddEval( "rmse" );
                }
                evaluator_.Init();
            } 
            /*!
            * \brief initialize the current data storage for model, if the model is used first time, call this function
            */
            inline void InitModel( void ){
                base_gbm.InitModel();
                mparam.AdjustBase();
            }
            /*! 
            * \brief load model from stream
            * \param fi input stream
            */          
            inline void LoadModel( utils::IStream &fi ){
                base_gbm.LoadModel( fi );
                utils::Assert( fi.Read( &mparam, sizeof(ModelParam) ) != 0 );
            }
            /*! 
             * \brief DumpModel
             * \param fo text file 
             * \param fmap feature map that may help give interpretations of feature            
              * \param with_stats whether print statistics as well
             */            
            inline void DumpModel( FILE *fo, const utils::FeatMap& fmap, bool with_stats ){
                base_gbm.DumpModel( fo, fmap, with_stats );
            }
            /*! 
             * \brief Dump path of all trees
             * \param fo text file 
             * \param data input data
             */
            inline void DumpPath( FILE *fo, const DMatrix &data ){
                base_gbm.DumpPath( fo, data.data );
            }
            /*! 
            * \brief save model to stream
            * \param fo output stream
            */
            inline void SaveModel( utils::IStream &fo ) const{
                base_gbm.SaveModel( fo );	
                fo.Write( &mparam, sizeof(ModelParam) );
            } 
            /*! 
             * \brief update the model for one iteration
             * \param iteration iteration number
             */
            inline void UpdateOneIter( int iter ){
                this->PredictBuffer( preds_, *train_, 0 );
                this->GetGradient( preds_, train_->labels, grad_, hess_ );
                std::vector<unsigned> root_index;
                base_gbm.DoBoost( grad_, hess_, train_->data, root_index );                
            }
            /*! 
             * \brief evaluate the model for specific iteration
             * \param iter iteration number
             * \param fo file to output log
             */            
            inline void EvalOneIter( int iter, FILE *fo = stderr ){
                fprintf( fo, "[%d]", iter );
                int buffer_offset = static_cast<int>( train_->Size() );
                
                for( size_t i = 0; i < evals_.size(); ++i ){
                    std::vector<float> &preds = this->eval_preds_[ i ];
                    this->PredictBuffer( preds, *evals_[i], buffer_offset);
                    evaluator_.Eval( fo, evname_[i].c_str(), preds, (*evals_[i]).labels );
                    buffer_offset += static_cast<int>( evals_[i]->Size() );
                }
                fprintf( fo,"\n" );
            }
            /*! \brief get prediction, without buffering */
            inline void Predict( std::vector<float> &preds, const DMatrix &data ){
                preds.resize( data.Size() );

                const unsigned ndata = static_cast<unsigned>( data.Size() );
                #pragma omp parallel for schedule( static )
                for( unsigned j = 0; j < ndata; ++ j ){
                    preds[j] = mparam.PredTransform
                        ( mparam.base_score + base_gbm.Predict( data.data, j, -1 ) );
                }
            }
        public:
            /*! 
             * \brief update the model for one iteration
             * \param iteration iteration number
             */
            inline void UpdateInteract( std::string action ){
                this->InteractPredict( preds_, *train_, 0 ); 

                if( action == "remove" ){
                    base_gbm.DelteBooster(); return;
                }

                int buffer_offset = static_cast<int>( train_->Size() );
                for( size_t i = 0; i < evals_.size(); ++i ){
                    std::vector<float> &preds = this->eval_preds_[ i ];                
                    this->InteractPredict( preds, *evals_[i], buffer_offset );
                    buffer_offset += static_cast<int>( evals_[i]->Size() );
                }
                
                this->GetGradient( preds_, train_->labels, grad_, hess_ );
                std::vector<unsigned> root_index;
                base_gbm.DoBoost( grad_, hess_, train_->data, root_index );

                this->InteractRePredict( *train_, 0 );
                buffer_offset = static_cast<int>( train_->Size() );
                for( size_t i = 0; i < evals_.size(); ++i ){
                    this->InteractRePredict( *evals_[i], buffer_offset );
                    buffer_offset += static_cast<int>( evals_[i]->Size() );
                }
            }
        private:
            /*! \brief get the transformed predictions, given data */
            inline void InteractPredict( std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset ){
                preds.resize( data.Size() );
                const unsigned ndata = static_cast<unsigned>( data.Size() );
                #pragma omp parallel for schedule( static )
                for( unsigned j = 0; j < ndata; ++ j ){                
                    preds[j] = mparam.PredTransform
                        ( mparam.base_score + base_gbm.InteractPredict( data.data, j, buffer_offset + j ) );
                }
            }
            /*! \brief repredict trial */
            inline void InteractRePredict( const DMatrix &data, unsigned buffer_offset ){
                const unsigned ndata = static_cast<unsigned>( data.Size() );
                #pragma omp parallel for schedule( static )
                for( unsigned j = 0; j < ndata; ++ j ){
                    base_gbm.InteractRePredict( data.data, j, buffer_offset + j );
                }
            }
        private:
            /*! \brief get the transformed predictions, given data */
            inline void PredictBuffer( std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset ){
                preds.resize( data.Size() );

                const unsigned ndata = static_cast<unsigned>( data.Size() );
                #pragma omp parallel for schedule( static )
                for( unsigned j = 0; j < ndata; ++ j ){                
                    preds[j] = mparam.PredTransform
                        ( mparam.base_score + base_gbm.Predict( data.data, j, buffer_offset + j ) );
                }
            }

            /*! \brief get the first order and second order gradient, given the transformed predictions and labels */
            inline void GetGradient( const std::vector<float> &preds, 
                                     const std::vector<float> &labels, 
                                     std::vector<float> &grad,
                                     std::vector<float> &hess ){
                grad.resize( preds.size() ); hess.resize( preds.size() );

                const unsigned ndata = static_cast<unsigned>( preds.size() );
                #pragma omp parallel for schedule( static )
                for( unsigned j = 0; j < ndata; ++ j ){
                    grad[j] = mparam.FirstOrderGradient( preds[j], labels[j] );
                    hess[j] = mparam.SecondOrderGradient( preds[j], labels[j] );
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
                /* \brief number of features  */
                int num_feature;
                /*! \brief reserved field */
                int reserved[ 16 ];
                /*! \brief constructor */
                ModelParam( void ){
                    base_score = 0.5f;
                    loss_type  = 0;
                    num_feature = 0;
                    memset( reserved, 0, sizeof( reserved ) );
                }
                /*! 
                * \brief set parameters from outside 
                * \param name name of the parameter
                * \param val  value of the parameter
                */
                inline void SetParam( const char *name, const char *val ){
                    if( !strcmp("base_score", name ) )  base_score = (float)atof( val );
                    if( !strcmp("loss_type", name ) )   loss_type = atoi( val );
                    if( !strcmp("bst:num_feature", name ) ) num_feature = atoi( val );
                }
                /*! 
                * \brief adjust base_score
                */                
                inline void AdjustBase( void ){
                    if( loss_type == 1 || loss_type == 2 ){
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
                    case kLogisticNeglik: 
                    case kLogisticClassify: return NegLoglikelihoodLoss(preds,labels);
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
            };
        private:
            int silent;
            EvalSet evaluator_;
            booster::GBMBase base_gbm;
            ModelParam   mparam;
            const DMatrix *train_;
            std::vector<DMatrix *> evals_;
            std::vector<std::string> evname_;
            std::vector<unsigned> buffer_index_;
        private:
            std::vector<float> grad_, hess_, preds_;
            std::vector< std::vector<float> > eval_preds_;
        };
    }
};

#endif
