#ifndef XGBOOST_LEARNER_H
#define XGBOOST_LEARNER_H
/*!
* \file xgboost_learner.h
* \brief class for gradient boosting learner
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "xgboost_data_instance.h"
#include "../utils/xgboost_omp.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost {
    namespace base {
        /*! \brief class for gradient boosting learner */
        class BoostLearner {
        public:
            /*! \brief constructor */
            BoostLearner(void) {
                silent = 0;
            }
            /*!
            * \brief booster associated with training and evaluating data
            * \param train pointer to the training data
            * \param evals array of evaluating data
            * \param evname name of evaluation data, used print statistics
            */
            BoostLearner(const DMatrix *train,
                const std::vector<DMatrix *> &evals,
                const std::vector<std::string> &evname) {
                silent = 0;
                this->SetData(train, evals, evname);
            }

            /*!
            * \brief associate booster with training and evaluating data
            * \param train pointer to the training data
            * \param evals array of evaluating data
            * \param evname name of evaluation data, used print statistics
            */
            inline void SetData(const DMatrix *train,
                const std::vector<DMatrix *> &evals,
                const std::vector<std::string> &evname) {
                this->train_ = train;
                this->evals_ = evals;
                this->evname_ = evname;
                // estimate feature bound
                int num_feature = (int)(train->data.NumCol());
                // assign buffer index
                unsigned buffer_size = static_cast<unsigned>(train->Size());

                for (size_t i = 0; i < evals.size(); ++i) {
                    buffer_size += static_cast<unsigned>(evals[i]->Size());
                    num_feature = std::max(num_feature, (int)(evals[i]->data.NumCol()));
                }

                char str_temp[25];
                if (num_feature > mparam.num_feature) {
                    mparam.num_feature = num_feature;
                    sprintf(str_temp, "%d", num_feature);
                    base_gbm.SetParam("bst:num_feature", str_temp);
                }

                sprintf(str_temp, "%u", buffer_size);
                base_gbm.SetParam("num_pbuffer", str_temp);
                if (!silent) {
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
            virtual inline void SetParam(const char *name, const char *val) {
                if (!strcmp(name, "silent"))  silent = atoi(val);
                mparam.SetParam(name, val);
                base_gbm.SetParam(name, val);
            }
            /*!
            * \brief initialize solver before training, called before training
            * this function is reserved for solver to allocate necessary space and do other preparation
            */
            inline void InitTrainer(void) {
                base_gbm.InitTrainer();
            }
            /*!
            * \brief initialize the current data storage for model, if the model is used first time, call this function
            */
            inline void InitModel(void) {
                base_gbm.InitModel();
		if(!silent) printf("BoostLearner:InitModel Done!\n");
            }
            /*!
            * \brief load model from stream
            * \param fi input stream
            */
            inline void LoadModel(utils::IStream &fi) {
                base_gbm.LoadModel(fi);
                utils::Assert(fi.Read(&mparam, sizeof(ModelParam)) != 0);
            }
            /*!
             * \brief DumpModel
             * \param fo text file
             * \param fmap feature map that may help give interpretations of feature
             * \param with_stats whether print statistics as well
             */
            inline void DumpModel(FILE *fo, const utils::FeatMap& fmap, bool with_stats) {
                base_gbm.DumpModel(fo, fmap, with_stats);
            }
            /*!
             * \brief Dump path of all trees
             * \param fo text file
             * \param data input data
             */
            inline void DumpPath(FILE *fo, const DMatrix &data) {
                base_gbm.DumpPath(fo, data.data);
            }

            /*!
            * \brief save model to stream
            * \param fo output stream
            */
            inline void SaveModel(utils::IStream &fo) const {
                base_gbm.SaveModel(fo);
                fo.Write(&mparam, sizeof(ModelParam));
            }

            virtual void EvalOneIter(int iter, FILE *fo = stderr) {}

            /*!
             * \brief update the model for one iteration
             * \param iteration iteration number
             */
            inline void UpdateOneIter(int iter) {
                this->PredictBuffer(preds_, *train_, 0);
        	this->GetGradient(preds_, train_->labels, train_->group_index, grad_, hess_);
                std::vector<unsigned> root_index;
                base_gbm.DoBoost(grad_, hess_, train_->data, root_index);
	    }

            /*! \brief get intransformed prediction, without buffering */
            inline void Predict(std::vector<float> &preds, const DMatrix &data) {
                preds.resize(data.Size());

                const unsigned ndata = static_cast<unsigned>(data.Size());
#pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j) {
                    preds[j] = base_gbm.Predict(data.data, j, -1);
                }
            }

        public:
            /*!
             * \brief update the model for one iteration
             * \param iteration iteration number
             */
            virtual inline void UpdateInteract(std::string action){
                this->InteractPredict(preds_, *train_, 0);

                int buffer_offset = static_cast<int>(train_->Size());
                for (size_t i = 0; i < evals_.size(); ++i) {
                    std::vector<float> &preds = this->eval_preds_[i];
                    this->InteractPredict(preds, *evals_[i], buffer_offset);
                    buffer_offset += static_cast<int>(evals_[i]->Size());
                }

                if (action == "remove") {
                    base_gbm.DelteBooster();
                    return;
                }

                this->GetGradient(preds_, train_->labels, train_->group_index, grad_, hess_);
                std::vector<unsigned> root_index;
                base_gbm.DoBoost(grad_, hess_, train_->data, root_index);

                this->InteractRePredict(*train_, 0);
                buffer_offset = static_cast<int>(train_->Size());
                for (size_t i = 0; i < evals_.size(); ++i) {
                    this->InteractRePredict(*evals_[i], buffer_offset);
                    buffer_offset += static_cast<int>(evals_[i]->Size());
                }
            };

        protected:
            /*! \brief get the intransformed predictions, given data */
            inline void InteractPredict(std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset) {
                preds.resize(data.Size());
                const unsigned ndata = static_cast<unsigned>(data.Size());
#pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j) {
                    preds[j] = base_gbm.InteractPredict(data.data, j, buffer_offset + j);
                }
            }
            /*! \brief repredict trial */
            inline void InteractRePredict(const xgboost::base::DMatrix &data, unsigned buffer_offset) {
                const unsigned ndata = static_cast<unsigned>(data.Size());
#pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j) {
                    base_gbm.InteractRePredict(data.data, j, buffer_offset + j);
                }
            }

            /*! \brief get intransformed predictions, given data */
            virtual inline void PredictBuffer(std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset) {
                preds.resize(data.Size());
		const unsigned ndata = static_cast<unsigned>(data.Size());
#pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j) {
		    preds[j] = base_gbm.Predict(data.data, j, buffer_offset + j);
                }
            }

            /*! \brief get the first order and second order gradient, given the transformed predictions and labels */
            virtual inline void GetGradient(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index,
                std::vector<float> &grad,
                std::vector<float> &hess) {};


        protected:

            /*! \brief training parameter for regression */
            struct ModelParam {
                /* \brief type of loss function */
                int loss_type;
                /* \brief number of features  */
                int num_feature;
                /*! \brief reserved field */
                int reserved[16];
                /*! \brief constructor */
                ModelParam(void) {
                    loss_type = 0;
                    num_feature = 0;
                    memset(reserved, 0, sizeof(reserved));
                }
                /*!
                * \brief set parameters from outside
                * \param name name of the parameter
                * \param val  value of the parameter
                */
                inline void SetParam(const char *name, const char *val) {
                    if (!strcmp("loss_type", name))   loss_type = atoi(val);
                    if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
                }

            };

            int silent;
            booster::GBMBase base_gbm;
            ModelParam   mparam;
            const DMatrix *train_;
            std::vector<DMatrix *> evals_;
            std::vector<std::string> evname_;
            std::vector<unsigned> buffer_index_;
            std::vector<float> grad_, hess_, preds_;
            std::vector< std::vector<float> > eval_preds_;
        };
    }
};

#endif





