#ifndef XGBOOST_REGRANK_H
#define XGBOOST_REGRANK_H
/*!
* \file xgboost_regrank.h
* \brief class for gradient boosted regression and ranking
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "xgboost_regrank_data.h"
#include "xgboost_regrank_eval.h"
#include "xgboost_regrank_obj.h"
#include "../utils/xgboost_omp.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost{
    namespace regrank{
        /*! \brief class for gradient boosted regression and ranking */
        class RegRankBoostLearner{
        public:
            /*! \brief constructor */
            RegRankBoostLearner(void){
                silent = 0;
                obj_ = NULL;
                name_obj_ = "reg";
            }
            /*!
            * \brief a regression booter associated with training and evaluating data
            * \param train pointer to the training data
            * \param evals array of evaluating data
            * \param evname name of evaluation data, used print statistics
            */
            RegRankBoostLearner(const DMatrix *train,
                                const std::vector<DMatrix *> &evals,
                                const std::vector<std::string> &evname){
                silent = 0;
                this->SetData(train, evals, evname);
            }

            /*!
            * \brief associate regression booster with training and evaluating data
            * \param train pointer to the training data
            * \param evals array of evaluating data
            * \param evname name of evaluation data, used print statistics
            */
            inline void SetData(const DMatrix *train,
                                const std::vector<DMatrix *> &evals,
                                const std::vector<std::string> &evname){
                this->train_ = train;
                this->evals_ = evals;
                this->evname_ = evname;
                // estimate feature bound
                int num_feature = (int)(train->data.NumCol());
                // assign buffer index
                unsigned buffer_size = static_cast<unsigned>(train->Size());

                for (size_t i = 0; i < evals.size(); ++i){
                    buffer_size += static_cast<unsigned>(evals[i]->Size());
                    num_feature = std::max(num_feature, (int)(evals[i]->data.NumCol()));
                }

                char str_temp[25];
                if (num_feature > mparam.num_feature){
                    mparam.num_feature = num_feature;
                    sprintf(str_temp, "%d", num_feature);
                    base_gbm.SetParam("bst:num_feature", str_temp);
                }

                sprintf(str_temp, "%u", buffer_size);
                base_gbm.SetParam("num_pbuffer", str_temp);
                if (!silent){
                    printf("buffer_size=%u\n", buffer_size);
                }

                // set eval_preds tmp sapce
                this->eval_preds_.resize(evals.size(), std::vector<float>());
            }
            /*!
            * \brief set parameters from outside
            * \param name name of the parameter
            * \param val  value of the parameter
            */
            inline void SetParam(const char *name, const char *val){
                if (!strcmp(name, "silent"))  silent = atoi(val);
                if (!strcmp(name, "eval_metric"))  evaluator_.AddEval(val);
                if (!strcmp(name, "objective") )   name_obj_ = val;
                mparam.SetParam(name, val);
                base_gbm.SetParam(name, val);
                cfg_.push_back( std::make_pair( std::string(name), std::string(val) ) );
            }
            /*!
            * \brief initialize solver before training, called before training
            * this function is reserved for solver to allocate necessary space and do other preparation
            */
            inline void InitTrainer(void){
                base_gbm.InitTrainer();
                obj_ = CreateObjFunction( name_obj_.c_str() );
                for( size_t i = 0; i < cfg_.size(); ++ i ){
                    obj_->SetParam( cfg_[i].first.c_str(), cfg_[i].second.c_str() );
                }
                evaluator_.AddEval( obj_->DefaultEvalMetric() );
            }
            /*!
            * \brief initialize the current data storage for model, if the model is used first time, call this function
            */
            inline void InitModel(void){
                base_gbm.InitModel();
                mparam.AdjustBase();
            }
            /*!
            * \brief load model from stream
            * \param fi input stream
            */
            inline void LoadModel(utils::IStream &fi){
                base_gbm.LoadModel(fi);
                utils::Assert(fi.Read(&mparam, sizeof(ModelParam)) != 0);
            }
            /*!
             * \brief DumpModel
             * \param fo text file
             * \param fmap feature map that may help give interpretations of feature
             * \param with_stats whether print statistics as well
             */
            inline void DumpModel(FILE *fo, const utils::FeatMap& fmap, bool with_stats){
                base_gbm.DumpModel(fo, fmap, with_stats);
            }
            /*!
             * \brief Dump path of all trees
             * \param fo text file
             * \param data input data
             */
            inline void DumpPath(FILE *fo, const DMatrix &data){
                base_gbm.DumpPath(fo, data.data);
            }
            /*!
            * \brief save model to stream
            * \param fo output stream
            */
            inline void SaveModel(utils::IStream &fo) const{
                base_gbm.SaveModel(fo);
                fo.Write(&mparam, sizeof(ModelParam));
            }
            /*!
             * \brief update the model for one iteration
             * \param iteration iteration number
             */
            inline void UpdateOneIter(int iter){
                this->PredictBuffer(preds_, *train_, 0);
                obj_->GetGradient(preds_, train_->info, base_gbm.NumBoosters(), grad_, hess_);
                std::vector<unsigned> root_index;
                base_gbm.DoBoost(grad_, hess_, train_->data, root_index);
            }
            /*!
             * \brief evaluate the model for specific iteration
             * \param iter iteration number
             * \param fo file to output log
             */
            inline void EvalOneIter(int iter, FILE *fo = stderr){
                fprintf(fo, "[%d]", iter);
                int buffer_offset = static_cast<int>(train_->Size());

                for (size_t i = 0; i < evals_.size(); ++i){
                    std::vector<float> &preds = this->eval_preds_[i];
                    this->PredictBuffer(preds, *evals_[i], buffer_offset);
                    obj_->PredTransform(preds);
                    evaluator_.Eval(fo, evname_[i].c_str(), preds, evals_[i]->info);
                    buffer_offset += static_cast<int>(evals_[i]->Size());
                }
                fprintf(fo, "\n");
                fflush(fo);
            }
            /*! \brief get prediction, without buffering */
            inline void Predict(std::vector<float> &preds, const DMatrix &data){
                preds.resize(data.Size());
                const unsigned ndata = static_cast<unsigned>(data.Size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = mparam.base_score + base_gbm.Predict(data.data, j, -1);
                }
                obj_->PredTransform( preds );
            }            
        public:
            /*!
             * \brief interactive update 
             * \param action action type 
             */
            inline void UpdateInteract(std::string action){
                this->InteractPredict(preds_, *train_, 0);

                int buffer_offset = static_cast<int>(train_->Size());
                for (size_t i = 0; i < evals_.size(); ++i){
                    std::vector<float> &preds = this->eval_preds_[i];
                    this->InteractPredict(preds, *evals_[i], buffer_offset);
                    buffer_offset += static_cast<int>(evals_[i]->Size());
                }

                if (action == "remove"){
                    base_gbm.DelteBooster(); return;
                }

                obj_->GetGradient(preds_, train_->info, base_gbm.NumBoosters(), grad_, hess_);
                std::vector<unsigned> root_index;
                base_gbm.DoBoost(grad_, hess_, train_->data, root_index);

                this->InteractRePredict(*train_, 0);
                buffer_offset = static_cast<int>(train_->Size());
                for (size_t i = 0; i < evals_.size(); ++i){
                    this->InteractRePredict(*evals_[i], buffer_offset);
                    buffer_offset += static_cast<int>(evals_[i]->Size());
                }
            }
        private:
            /*! \brief get the transformed predictions, given data */
            inline void InteractPredict(std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset){
                preds.resize(data.Size());
                const unsigned ndata = static_cast<unsigned>(data.Size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = mparam.base_score + base_gbm.InteractPredict(data.data, j, buffer_offset + j);                    
                }
                obj_->PredTransform( preds );
            }
            /*! \brief repredict trial */
            inline void InteractRePredict(const DMatrix &data, unsigned buffer_offset){
                const unsigned ndata = static_cast<unsigned>(data.Size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    base_gbm.InteractRePredict(data.data, j, buffer_offset + j);
                }
            }
        private:
            /*! \brief get the transformed predictions, given data */
            inline void PredictBuffer(std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset){
                preds.resize(data.Size());
                const unsigned ndata = static_cast<unsigned>(data.Size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = mparam.base_score + base_gbm.Predict(data.data, j, buffer_offset + j);
                }
            }
        private:
            /*! \brief training parameter for regression */
            struct ModelParam{
                /* \brief global bias */
                float base_score;
                /* \brief type of loss function */
                int loss_type;
                /* \brief number of features  */
                int num_feature;                
                /*! \brief reserved field */
                int reserved[16];
                /*! \brief constructor */
                ModelParam(void){
                    base_score = 0.5f;
                    loss_type = 0;
                    num_feature = 0;
                    memset(reserved, 0, sizeof(reserved));
                }
                /*!
                * \brief set parameters from outside
                * \param name name of the parameter
                * \param val  value of the parameter
                */
                inline void SetParam(const char *name, const char *val){
                    if (!strcmp("base_score", name))  base_score = (float)atof(val);
                    if (!strcmp("loss_type", name))   loss_type = atoi(val);
                    if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
                }
                /*!
                * \brief adjust base_score
                */
                inline void AdjustBase(void){
                    if (loss_type == 1 || loss_type == 2){
                        utils::Assert(base_score > 0.0f && base_score < 1.0f, "sigmoid range constrain");
                        base_score = -logf(1.0f / base_score - 1.0f);
                    }
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
            // objective fnction
            IObjFunction *obj_;
            // name of objective function
            std::string name_obj_;
            std::vector< std::pair<std::string, std::string> > cfg_;
        private:
            std::vector<float> grad_, hess_, preds_;
            std::vector< std::vector<float> > eval_preds_;
        };
    }
};

#endif
