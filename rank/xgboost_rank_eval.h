#ifndef XGBOOST_RANK_EVAL_H
#define XGBOOST_RANK_EVAL_H
/*!
* \file xgboost_rank_eval.h
* \brief evaluation metrics for ranking
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_omp.h"

namespace xgboost {
    namespace rank {
        /*! \brief evaluator that evaluates the loss metrics */
        class IRankEvaluator {
        public:
            /*!
             * \brief evaluate a specific metric
             * \param preds prediction
             * \param labels label
             */
            virtual float Eval(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index) const = 0;
            /*! \return name of metric */
            virtual const char *Name(void) const = 0;
        };

        class Pair{
        public:
            float key_;
            float value_;

            Pair(float key, float value):key_(key),value_(value){
            }
        };

        bool PairKeyComparer(const Pair &a, const Pair &b){  
	  return a.key_ < b.key_;
        }

        bool PairValueComparer(const Pair &a, const Pair &b){
            return a.value_ < b.value_;
        }

        template<typename T1,typename T2,typename T3>
        class Triple{
	public:
	  T1 f1_;
	  T2 f2_;
	  T3 f3_;
	  Triple(T1 f1,T2 f2,T3 f3):f1_(f1),f2_(f2),f3_(f3){
	    
	  }
	};
	
	template<typename T1,typename T2,typename T3,typename T4>
        class Quadruple{
	public:
	  T1 f1_;
	  T2 f2_;
	  T3 f3_;
	  T4 f4_;
	  Quadruple(T1 f1,T2 f2,T3 f3,T4 f4):f1_(f1),f2_(f2),f3_(f3),f4_(f4){
	    
	  }
	};
	
	bool Triplef1Comparer(const Triple<float,float,int> &a, const Triple<float,float,int> &b){  
	  return a.f1_< b.f1_;
        }
        
        /*! \brief Mean Average Precision */
        class EvalMAP : public IRankEvaluator {
        public:
            float Eval(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index) const {
		if (group_index.size() <= 1) return 0;
                float acc = 0;
                std::vector<Pair> pairs_sort;
                for (int i = 0; i < group_index.size() - 1; i++){
                    for (int j = group_index[i]; j < group_index[i + 1]; j++){
                        Pair pair(preds[j], labels[j]);
                        pairs_sort.push_back(pair);
                    }
                    acc += average_precision(pairs_sort);
                }
                return acc / (group_index.size() - 1);
            }
            
	    

            virtual const char *Name(void) const {
                return "MAP";
            }
	private:
            float average_precision(std::vector<Pair> pairs_sort) const{

                std::sort(pairs_sort.begin(), pairs_sort.end(), PairKeyComparer);
                float hits = 0;
                float average_precision = 0;
                for (int j = 0; j < pairs_sort.size(); j++){
                    if (pairs_sort[j].value_ == 1){
                        hits++;
                        average_precision += hits / (j + 1);
                    }
                }
                if (hits != 0) average_precision /= hits;
                return average_precision;
            }
        };


        class EvalPair : public IRankEvaluator{
        public:
            float Eval(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index) const {
		if (group_index.size() <= 1) return 0;
                float acc = 0;
                for (int i = 0; i < group_index.size() - 1; i++){
                    acc += Count_Inversion(preds,labels,
			group_index[i],group_index[i+1]);
                }
                return acc / (group_index.size() - 1);	  
	    }

            const char *Name(void) const {
                return "PAIR";
            }
	private:
	    float Count_Inversion(const std::vector<float> &preds,
	      const std::vector<float> &labels,int begin,int end
	    ) const{
	      float ans = 0;
	      for(int i = begin; i < end; i++){
		for(int j = i + 1; j < end; j++){
		  if(preds[i] > preds[j] && labels[i] < labels[j])
		    ans++;
		}
	      }
	      return ans;
	    }
        };

        /*! \brief Normalized DCG */
        class EvalNDCG : public IRankEvaluator {
        public:
            float Eval(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index) const {
                if (group_index.size() <= 1) return 0;
                float acc = 0;
                std::vector<Pair> pairs_sort;
                for (int i = 0; i < group_index.size() - 1; i++){
                    for (int j = group_index[i]; j < group_index[i + 1]; j++){
                        Pair pair(preds[j], labels[j]);
                        pairs_sort.push_back(pair);
                    }
                    acc += NDCG(pairs_sort);
                }
                return acc / (group_index.size() - 1);
            }
            
            static float DCG(const std::vector<float> &labels){
		float ans = 0.0;
                for (int i = 0; i < labels.size(); i++){
                    ans += (pow(2,labels[i]) - 1 ) / log(i + 2);
                }
                return ans;
	    }
	    
            virtual const char *Name(void) const {
                return "NDCG";
            }
            
	  private:
            float NDCG(std::vector<Pair> pairs_sort) const{
                std::sort(pairs_sort.begin(), pairs_sort.end(), PairKeyComparer);
                float dcg = DCG(pairs_sort);
                std::sort(pairs_sort.begin(), pairs_sort.end(), PairValueComparer);
                float IDCG = DCG(pairs_sort);
                if (IDCG == 0) return 0;
                return dcg / IDCG;
            }

            float DCG(std::vector<Pair> pairs_sort) const{
                std::vector<float> labels;
	        for (int i = 1; i < pairs_sort.size(); i++){
		  labels.push_back(pairs_sort[i].value_);
		}
                return DCG(labels);
            }

            
        };

    };

    namespace rank {
        /*! \brief a set of evaluators */
        class RankEvalSet {
        public:
            inline void AddEval(const char *name) {
                if (!strcmp(name, "PAIR")) evals_.push_back(&pair_);
                if (!strcmp(name, "MAP")) evals_.push_back(&map_);
                if (!strcmp(name, "NDCG")) evals_.push_back(&ndcg_);
            }

            inline void Init(void) {
                std::sort(evals_.begin(), evals_.end());
                evals_.resize(std::unique(evals_.begin(), evals_.end()) - evals_.begin());
            }

            inline void Eval(FILE *fo, const char *evname,
                const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<int> &group_index) const {
                for (size_t i = 0; i < evals_.size(); ++i) {
                    float res = evals_[i]->Eval(preds, labels, group_index);
                    fprintf(fo, "\t%s-%s:%f", evname, evals_[i]->Name(), res);
                }
            }

        private:
            EvalPair pair_;
            EvalMAP map_;
            EvalNDCG ndcg_;
            std::vector<const IRankEvaluator*> evals_;
        };
    };
};
#endif
