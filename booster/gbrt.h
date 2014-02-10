#ifndef _GBRT_H_
#define _GBRT_H_

#include "../utils/xgboost_config.h"
#include "xgboost_regression_data_reader.h"
#include "xgboost_gbmbase.h"
#include <math.h>
using namespace xgboost::utils;
using namespace xgboost::booster;

class gbrt{

public:
	gbrt(const char* config_path){
		ConfigIterator config_itr(config_path);
		while(config_itr.Next()){
			SetParam(config_itr.name,config_itr.val);
			base_model.SetParam(config_itr.name,config_itr.val);
		}
	}

	void SetParam( const char *name, const char *val ){
		param.SetParam(name, val);
	}

	void train(){
		xgboost_regression_data_reader data_reader(param.train_file_path);
		base_model.InitModel();
		base_model.InitTrainer();
		std::vector<float> grad,hess;
		std::vector<unsigned> root_index;
		int instance_num = data_reader.InsNum();
		float label = 0,pred_transform = 0;
		grad.resize(instance_num); hess.resize(instance_num);
		for(int i = 0; i < 100; i++){
			grad.clear();hess.clear();
			for(int j = 0; j < instance_num; j++){
				label = data_reader.GetLabel(j);
				pred_transform = Logistic(base_model.Predict(data_reader.GetLine(j)));
				grad.push_back(FirstOrderGradient(pred_transform,label));
				hess.push_back(SecondOrderGradient(pred_transform));
			}
			base_model.DoBoost(grad,hess,data_reader.GetImage(),root_index );
		}
	}

	struct GBRTParam{

		/*! \brief path of input training data */
		const char* train_file_path;

		GBRTParam( void ){
		}
		/*! 
		* \brief set parameters from outside 
		* \param name name of the parameter
		* \param val  value of the parameter
		*/
		inline void SetParam( const char *name, const char *val ){
			if( !strcmp("train_file_path", name ) )  train_file_path = val;
		}
	};

private:
	inline float FirstOrderGradient(float pred_transform,float label){
		return label - pred_transform;
	}

	inline float SecondOrderGradient(float pred_transform){
		return pred_transform * ( 1 - pred_transform );
	}

	inline float Logistic(float x){
		return 1.0/(1.0 + exp(-x));
	}

	GBMBaseModel base_model;	
	GBRTParam param;

};

#endif