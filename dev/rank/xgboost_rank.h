#ifndef XGBOOST_RANK_H
#define XGBOOST_RANK_H
/*!
* \file xgboost_rank.h
* \brief class for gradient boosting ranking
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <vector>
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

            virtual inline void SetParam(const char *name, const char *val){
                BoostLearner::SetParam(name,val);
	        if (!strcmp(name, "eval_metric"))  evaluator_.AddEval(val);
                if (!strcmp(name, "rank:sampler"))  sampler.AssignSampler(atoi(val));
            }
            
	private:
	    inline std::vector< Triple<float,float,int> > GetSortedTuple(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index,
		int group){
		std::vector< Triple<float,float,int> > sorted_triple;
		for(int j = group_index[group]; j < group_index[group+1]; j++){
		      sorted_triple.push_back(Triple<float,float,int>(preds[j],labels[j],j));
		}
                std::sort(sorted_triple.begin(),sorted_triple.end(),Triplef1Comparer);
		return sorted_triple;
	    }
	    
	    inline std::vector<int> GetIndexMap(std::vector< Triple<float,float,int> > sorted_triple,int start){
		std::vector<int> index_remap;
		index_remap.resize(sorted_triple.size());
		for(int i = 0; i < sorted_triple.size(); i++){
		  index_remap[sorted_triple[i].f3_-start] = i;
		}
		return index_remap;
	    }
	    
	    inline float GetLambdaMAP(const std::vector< Triple<float,float,int> > sorted_triple,
		int index1,int index2,
		std::vector< Quadruple<float,float,float,float> > map_acc){
	        if(index1 > index2) std::swap(index1,index2);
		float original = map_acc[index2].f1_;
		if(index1 != 0) original -= map_acc[index1 - 1].f1_;
		float changed = 0;
		if(sorted_triple[index1].f2_ < sorted_triple[index2].f2_){
		  changed += map_acc[index2 - 1].f3_ - map_acc[index1].f3_;
		  changed += (map_acc[index1].f4_ + 1.0f)/(index1 + 1);
		}else{
		  changed += map_acc[index2 - 1].f2_ - map_acc[index1].f2_;
		  changed += map_acc[index2].f4_/(index2 + 1);
		}
		float ans = (changed - original)/(map_acc[map_acc.size() - 1].f4_);
		if(ans < 0) ans = -ans;
		return ans;
	    }
	    
	    inline float GetLambdaNDCG(const std::vector< Triple<float,float,int> > sorted_triple,
		int index1,
		int index2,float IDCG){
	        float original = pow(2,sorted_triple[index1].f2_)/log(index1+2)
		                + pow(2,sorted_triple[index2].f2_)/log(index2+2);
	        float changed = pow(2,sorted_triple[index2].f2_)/log(index1+2)
		                + pow(2,sorted_triple[index1].f2_)/log(index2+2);
		float ans = (original - changed)/IDCG;
		if(ans < 0) ans = -ans;
		return ans;
	    }
	    
	    
	    inline float GetIDCG(const std::vector< Triple<float,float,int> > sorted_triple){
		std::vector<float> labels;
		for(int i = 0; i < sorted_triple.size(); i++){
		  labels.push_back(sorted_triple[i].f2_);
		}
		
		std::sort(labels.begin(),labels.end(),std::greater<float>());
		return EvalNDCG::DCG(labels);
	    }
	    
	    inline std::vector< Quadruple<float,float,float,float> > GetMAPAcc(const std::vector< Triple<float,float,int> > sorted_triple){
		std::vector< Quadruple<float,float,float,float> > map_acc;
		float hit = 0,acc1 = 0,acc2 = 0,acc3 = 0;
		for(int i = 0; i < sorted_triple.size(); i++){
		    if(sorted_triple[i].f2_ == 1) {
		      hit++;
		      acc1 += hit /( i + 1 );
		      acc2 += (hit - 1)/(i+1);
		      acc3 += (hit + 1)/(i+1);
		    }
		    map_acc.push_back(Quadruple<float,float,float,float>(acc1,acc2,acc3,hit));
		}
		return map_acc;
	      
	    }
	    
	    inline void GetGroupGradient(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index,
                std::vector<float> &grad,
                std::vector<float> &hess,
		const std::vector< Triple<float,float,int> > sorted_triple,
		const std::vector<int> index_remap,
		const sample::Pairs& pairs,
	        int group){
	        bool j_better;
                float IDCG, pred_diff, pred_diff_exp, delta;
		float first_order_gradient, second_order_gradient;
                std::vector< Quadruple<float,float,float,float> > map_acc;
		
		if(mparam.loss_type == NDCG){
		  IDCG = GetIDCG(sorted_triple);
		}else if(mparam.loss_type == MAP){
		  map_acc = GetMAPAcc(sorted_triple);
		}
		
		for (int j = group_index[group]; j < group_index[group + 1]; j++){
		    std::vector<int> pair_instance = pairs.GetPairs(j);
                    for (int k = 0; k < pair_instance.size(); k++){
			j_better = labels[j] > labels[pair_instance[k]];
                        if (j_better){
			    switch(mparam.loss_type){
			      case PAIRWISE: delta = 1.0;break;
			      case MAP: delta = GetLambdaMAP(sorted_triple,index_remap[j - group_index[group]],index_remap[pair_instance[k]-group_index[group]],map_acc);break;
			      case NDCG: delta = GetLambdaNDCG(sorted_triple,index_remap[j - group_index[group]],index_remap[pair_instance[k]-group_index[group]],IDCG);break;
			      default: utils::Error("Cannot find the specified loss type");
			    }
		        
			    pred_diff = preds[preds[j] - pair_instance[k]];
			    pred_diff_exp = j_better ? expf(-pred_diff) : expf(pred_diff);
			    first_order_gradient = delta * FirstOrderGradient(pred_diff_exp);
			    second_order_gradient = 2 * delta * SecondOrderGradient(pred_diff_exp);
			    hess[j] += second_order_gradient;
			    grad[j] += first_order_gradient;
			    hess[pair_instance[k]] += second_order_gradient;
			    grad[pair_instance[k]] += -first_order_gradient;
			}
		    }
                }
	    }
	public:
            /*! \brief get the first order and second order gradient, given the 
	     *         intransformed predictions and labels */
            inline void GetGradient(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index,
                std::vector<float> &grad,
                std::vector<float> &hess) {
                grad.resize(preds.size());
                hess.resize(preds.size());
                for (int i = 0; i < group_index.size() - 1; i++){
		    sample::Pairs pairs = sampler.GenPairs(preds, labels, group_index[i], group_index[i + 1]);
                    //pairs.GetPairs()
		    std::vector< Triple<float,float,int> > sorted_triple = GetSortedTuple(preds,labels,group_index,i);
		    std::vector<int> index_remap = GetIndexMap(sorted_triple,group_index[i]);
		    GetGroupGradient(preds,labels,group_index,
			grad,hess,sorted_triple,index_remap,pairs,i);
                }
            }

            inline void UpdateInteract(std::string action) {
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

                this->GetGradient(preds_, train_->labels,train_->group_index, grad_, hess_);
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





