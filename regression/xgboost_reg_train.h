#ifndef _XGBOOST_REG_TRAIN_H_
#define _XGBOOST_REG_TRAIN_H_

#include<iostream>
#include<string>
#include<fstream>
#include"../utils/xgboost_config.h"
#include"xgboost_reg.h"
#include"xgboost_regdata.h"
#include"../utils/xgboost_string.h"

using namespace xgboost::utils;
namespace xgboost{
	namespace regression{
		class RegBoostTrain{
		public:
			void train(char* config_path,bool silent = false){
				reg_boost_learner = new xgboost::regression::RegBoostLearner(silent);
				ConfigIterator config_itr(config_path);
				//Get the training data and validation data paths, config the Learner
				while (config_itr.Next()){
					reg_boost_learner->SetParam(config_itr.name(),config_itr.val());
					train_param.SetParam(config_itr.name(),config_itr.val());
				}

				Assert(train_param.validation_data_paths.size() == train_param.validation_data_names.size(),
					"The number of validation paths is not the same as the number of validation data set names");

				//Load Data
				xgboost::regression::DMatrix train;
				train.LoadText(train_param.train_path);
				std::vector<const xgboost::regression::DMatrix*> evals;
				for(int i = 0; i < train_param.validation_data_paths.size(); i++){
					xgboost::regression::DMatrix eval;
					eval.LoadText(train_param.validation_data_paths[i].c_str());
					evals.push_back(&eval);
				}
				reg_boost_learner->SetData(&train,evals,train_param.validation_data_names);

				//begin training
				reg_boost_learner->InitTrainer();
				char model_path[256];
				for(int i = 1; i <= train_param.boost_iterations; i++){
					reg_boost_learner->UpdateOneIter(i);
					//save the models during the iterations
					if(train_param.save_period != 0 && i % train_param.save_period == 0){
						sscanf(model_path,"%s/%d.model",train_param.model_dir_path,i);
						FILE* file = fopen(model_path,"w");
						FileStream fin(file);
						reg_boost_learner->SaveModel(fin);
						fin.Close();
					}
				}

				//save the final model
				sscanf(model_path,"%s/final.model",train_param.model_dir_path);
				FILE* file = fopen(model_path,"w");
				FileStream fin(file);
				reg_boost_learner->SaveModel(fin);
				fin.Close();

			}
		private:
			struct TrainParam{
				/* \brief upperbound of the number of boosters */
				int boost_iterations;

				/* \brief the period to save the model, -1 means only save the final round model */
				int save_period;

				/* \brief the path of training data set */
				const char* train_path;

				/* \brief the path of directory containing the saved models */
				const char* model_dir_path;

				/* \brief the paths of validation data sets */
				std::vector<std::string> validation_data_paths;

				/* \brief the names of the validation data sets */
				std::vector<std::string> validation_data_names;

				/*! 
				* \brief set parameters from outside 
				* \param name name of the parameter
				* \param val  value of the parameter
				*/
				inline void SetParam(const char *name,const char *val ){
					if( !strcmp("boost_iterations", name ) )  boost_iterations = (float)atof( val );
					if( !strcmp("save_period", name ) )   save_period = atoi( val );
					if( !strcmp("train_path",  name ) ) train_path = val;
					if( !strcmp("model_dir_path", name ) ) model_dir_path = val;
					if( !strcmp("validation_paths",  name) ) {
						validation_data_paths = StringProcessing::split(val,';');
					}
					if( !strcmp("validation_names",  name) ) {
						validation_data_names = StringProcessing::split(val,';');
					}
				}
			};

			TrainParam train_param;
			xgboost::regression::RegBoostLearner* reg_boost_learner;
		};
	}
}

#endif
