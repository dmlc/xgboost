#ifndef _XGBOOST_REG_TEST_H_
#define _XGBOOST_REG_TEST_H_

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
        /*!
        * \brief wrapping the testing process of the gradient 
        boosting regression model,given the configuation
        * \author Kailong Chen: chenkl198812@gmail.com
        */
        class RegBoostTest{
        public:
            /*!
            * \brief to start the testing process of gradient boosting regression
            *        model given the configuation, and finally save the prediction
            *        results to the specified paths.
            * \param config_path the location of the configuration
            * \param silent whether to print feedback messages
            */
            void test(char* config_path,bool silent = false){
                reg_boost_learner = new xgboost::regression::RegBoostLearner();
                ConfigIterator config_itr(config_path);
                //Get the training data and validation data paths, config the Learner
                while (config_itr.Next()){
                    reg_boost_learner->SetParam(config_itr.name(),config_itr.val());
                    test_param.SetParam(config_itr.name(),config_itr.val());
                }

                Assert(test_param.test_paths.size() == test_param.test_names.size(),
                    "The number of test data set paths is not the same as the number of test data set data set names");

                //begin testing
                reg_boost_learner->InitModel();
                char model_path[256];
                std::vector<float> preds;
                for(size_t i = 0; i < test_param.test_paths.size(); i++){
                    xgboost::regression::DMatrix test_data;
                    test_data.LoadText(test_param.test_paths[i].c_str());
                    sprintf(model_path,"%s/final.model",test_param.model_dir_path);
                    // BUG: model need to be rb
                    FileStream fin(fopen(model_path,"r"));
                    reg_boost_learner->LoadModel(fin);
                    fin.Close();
                    reg_boost_learner->Predict(preds,test_data);
                }
            }

        private:
            struct TestParam{
                /* \brief upperbound of the number of boosters */
                int boost_iterations;

                /* \brief the period to save the model, -1 means only save the final round model */
                int save_period;

                /* \brief the path of directory containing the saved models */
                char model_dir_path[256];

                /* \brief the path of directory containing the output prediction results */
                char pred_dir_path[256];

                /* \brief the paths of test data sets */
                std::vector<std::string> test_paths;

                /* \brief the names of the test data sets */
                std::vector<std::string> test_names;

                /*! 
                * \brief set parameters from outside 
                * \param name name of the parameter
                * \param val  value of the parameter
                */
                inline void SetParam(const char *name,const char *val ){
                    if( !strcmp("model_dir_path", name ) ) strcpy(model_dir_path,val);
                    if( !strcmp("pred_dir_path", name ) ) strcpy(pred_dir_path,val);
                    if( !strcmp("test_paths",  name) ) {
                        test_paths = StringProcessing::split(val,';');
                    }
                    if( !strcmp("test_names",  name) ) {
                        test_names = StringProcessing::split(val,';');
                    }
                }
            };

            TestParam test_param;
            xgboost::regression::RegBoostLearner* reg_boost_learner;
        };
    }
}

#endif
