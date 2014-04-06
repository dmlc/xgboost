#ifndef XGBOOST_RANK_H
#define XGBOOST_RANK_H
/*!
* \file xgboost_rank.h
* \brief class for gradient boosting ranking
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "xgboost_sample.h"
#include "xgboost_rank_eval.h"
#include "../base/xgboost_data_instance.h"
#include "../utils/xgboost_omp.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"
#include "../base/xgboost_learner.h"

namespace xgboost {
	namespace rank {
		/*! \brief class for gradient boosted regression */
		class RankBoostLearner :public base::BoostLearner{
		public:
			/*! \brief constructor */
			RankBoostLearner(void) {
				BoostLearner();
			}
			/*!
			* \brief a rank booster associated with training and evaluating data
			* \param train pointer to the training data
			* \param evals array of evaluating data
			* \param evname name of evaluation data, used print statistics
			*/
			RankBoostLearner(const base::DMatrix *train,
				const std::vector<base::DMatrix *> &evals,
				const std::vector<std::string> &evname) {

				BoostLearner(train, evals, evname);
			}

			/*!
			* \brief initialize solver before training, called before training
			* this function is reserved for solver to allocate necessary space
			* and do other preparation
			*/
			inline void InitTrainer(void) {
				BoostLearner::InitTrainer();
				if (mparam.loss_type == PAIRWISE) {
					evaluator_.AddEval("PAIR");
				}
				else if (mparam.loss_type == MAP) {
					evaluator_.AddEval("MAP");
				}
				else {
					evaluator_.AddEval("NDCG");
				}
				evaluator_.Init();
			}

			void EvalOneIter(int iter, FILE *fo = stderr) {
				fprintf(fo, "[%d]", iter);
				int buffer_offset = static_cast<int>(train_->Size());

				for (size_t i = 0; i < evals_.size(); ++i) {
					std::vector<float> &preds = this->eval_preds_[i];
					this->PredictBuffer(preds, *evals_[i], buffer_offset);
					evaluator_.Eval(fo, evname_[i].c_str(), preds, (*evals_[i]).labels, (*evals_[i]).group_index);
					buffer_offset += static_cast<int>(evals_[i]->Size());
				}
				fprintf(fo, "\n");
			}

			inline void SetParam(const char *name, const char *val){
				if (!strcmp(name, "eval_metric"))  evaluator_.AddEval(val);
				if (!strcmp(name, "rank:sampler"))  sampler.AssignSampler(atoi(val));
			}
			/*! \brief get the first order and second order gradient, given the transformed predictions and labels */
			inline void GetGradient(const std::vector<float> &preds,
				const std::vector<float> &labels,
				const std::vector<int> &group_index,
				std::vector<float> &grad,
				std::vector<float> &hess) {
				grad.resize(preds.size());
				hess.resize(preds.size());
				bool j_better;
				float pred_diff, pred_diff_exp, first_order_gradient, second_order_gradient;
				for (int i = 0; i < group_index.size() - 1; i++){
					sample::Pairs pairs = sampler.GenPairs(preds, labels, group_index[i], group_index[i + 1]);
					for (int j = group_index[i]; j < group_index[i + 1]; j++){
						std::vector<int> pair_instance = pairs.GetPairs(j);
						for (int k = 0; k < pair_instance.size(); k++){
							j_better = labels[j] > labels[pair_instance[k]];
							if (j_better){
								pred_diff = preds[preds[j] - pair_instance[k]];
								pred_diff_exp = j_better ? expf(-pred_diff) : expf(pred_diff);
								first_order_gradient = FirstOrderGradient(pred_diff_exp);
								second_order_gradient = 2 * SecondOrderGradient(pred_diff_exp);
								hess[j] += second_order_gradient;
								grad[j] += first_order_gradient;
								hess[pair_instance[k]] += second_order_gradient;
								grad[pair_instance[k]] += -first_order_gradient;
							}
						}
					}
				}
			}

			inline void UpdateInteract(std::string action) {
				
			}
		private:
			enum LossType {
				PAIRWISE = 0,
				MAP = 1,
				NDCG = 2
			};



			/*!
			* \brief calculate first order gradient of pairwise loss function(f(x) = ln(1+exp(-x)),
			* given the exponential of the difference of intransformed pair predictions
			* \param the intransformed prediction of positive instance
			* \param the intransformed prediction of negative instance
			* \return first order gradient
			*/
			inline float FirstOrderGradient(float pred_diff_exp) const {
				return -pred_diff_exp / (1 + pred_diff_exp);
			}

			/*!
			* \brief calculate second order gradient of pairwise loss function(f(x) = ln(1+exp(-x)),
			* given the exponential of the difference of intransformed pair predictions
			* \param the intransformed prediction of positive instance
			* \param the intransformed prediction of negative instance
			* \return second order gradient
			*/
			inline float SecondOrderGradient(float pred_diff_exp) const {
				return pred_diff_exp / pow(1 + pred_diff_exp, 2);
			}

		private:
			RankEvalSet evaluator_;
			sample::PairSamplerWrapper sampler;
		};
	};
};

#endif





