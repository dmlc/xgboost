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

			RegBoostLearner(bool silent = false){
				this->silent = silent;
			}

			/*! 
			* \brief a regression booter associated with training and evaluating data 
			* \param train pointer to the training data
			* \param evals array of evaluating data
			* \param evname name of evaluation data, used print statistics
			*/
			RegBoostLearner( const DMatrix *train,
				std::vector<const DMatrix *> evals,
				std::vector<std::string> evname, bool silent = false ){
					this->silent = silent;
					SetData(train,evals,evname);
			}

			/*! 
			* \brief associate regression booster with training and evaluating data 
			* \param train pointer to the training data
			* \param evals array of evaluating data
			* \param evname name of evaluation data, used print statistics
			*/
			inline void SetData(const DMatrix *train,
				std::vector<const DMatrix *> evals,
				std::vector<std::string> evname){
					this->train_ = train;
					this->evals_ = evals;
					this->evname_ = evname; 
					//assign buffer index
					int buffer_size = (*train).size();
					for(int i = 0; i < evals.size(); i++){
						buffer_size += (*evals[i]).size();
					}
					char str[25];
					itoa(buffer_size,str,10);
					base_model.SetParam("num_pbuffer",str);
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
             * \brief initialize the current data storage for model, if the model is used first time, call this function
             */
			inline void InitModel( void ){
				base_model.InitModel();
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
			* \param iteration the number of updating iteration 
			*/           
			inline void UpdateOneIter( int iteration ){
				std::vector<float> grad,hess,preds;
				std::vector<unsigned> root_index;
				booster::FMatrixS::Image train_image((*train_).data);
				Predict(preds,*train_,0);
				Gradient(preds,(*train_).labels,grad,hess);
				base_model.DoBoost(grad,hess,train_image,root_index);
				int buffer_index_offset = (*train_).size();
				float loss = 0.0;
				for(int i = 0; i < evals_.size();i++){
					Predict(preds, *evals_[i], buffer_index_offset);
					loss = mparam.Loss(preds,(*evals_[i]).labels);
					if(!silent){
						printf("The loss of %s data set in %d the \
							iteration is %f",evname_[i].c_str(),&iteration,&loss);
					}
					buffer_index_offset += (*evals_[i]).size();
				}
				
			}

			/*! \brief get the transformed predictions, given data */
			inline void Predict( std::vector<float> &preds, const DMatrix &data,int buffer_index_offset = 0 ){
				int data_size = data.size();
				preds.resize(data_size);
				for(int j = 0; j < data_size; j++){
					preds[j] = mparam.PredTransform(mparam.base_score + 
						base_model.Predict(data.data[j],buffer_index_offset + j));
				}
			}

		private:
			/*! \brief get the first order and second order gradient, given the transformed predictions and labels*/
			inline void Gradient(const std::vector<float> &preds, const std::vector<float> &labels, std::vector<float> &grad,
				std::vector<float> &hess){
					grad.clear(); 
					hess.clear();
					for(int j = 0; j < preds.size(); j++){
						grad.push_back(mparam.FirstOrderGradient(preds[j],labels[j]));
						hess.push_back(mparam.SecondOrderGradient(preds[j],labels[j]));
					}
			}

			enum LOSS_TYPE_LIST{
				LINEAR_SQUARE,
				LOGISTIC_NEGLOGLIKELIHOOD,
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
				* \brief calculate first order gradient of loss, given transformed prediction
				* \param predt transformed prediction
				* \param label true label
				* \return first order gradient
				*/
				inline float FirstOrderGradient( float predt, float label ) const{
					switch( loss_type ){                        
					case LINEAR_SQUARE: return predt - label;
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
					case LINEAR_SQUARE: return 1.0f;
					case LOGISTIC_NEGLOGLIKELIHOOD: return predt * ( 1 - predt );
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
					case LINEAR_SQUARE: return SquareLoss(preds,labels);
					case LOGISTIC_NEGLOGLIKELIHOOD: return NegLoglikelihoodLoss(preds,labels);
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
					for(int i = 0; i < preds.size(); i++)
						ans += pow(preds[i] - labels[i], 2);
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
					for(int i = 0; i < preds.size(); i++)
						ans -= labels[i] * log(preds[i]) + ( 1 - labels[i] ) * log(1 - preds[i]);
					return ans;
				}

				
				/*! 
				* \brief transform the linear sum to prediction 
				* \param x linear sum of boosting ensemble
				* \return transformed prediction
				*/
				inline float PredTransform( float x ){
					switch( loss_type ){                        
					case LINEAR_SQUARE: return x;
					case LOGISTIC_NEGLOGLIKELIHOOD: return 1.0f/(1.0f + expf(-x));
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
			bool silent;
		};
	}
};

#endif
