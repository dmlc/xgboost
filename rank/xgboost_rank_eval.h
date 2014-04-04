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
struct IRankEvaluator {
    /*!
     * \brief evaluate a specific metric
     * \param preds prediction
     * \param labels label
     */
    virtual float Eval( const std::vector<float> &preds,
                        const std::vector<float> &labels,
                        const std::vector<int> &group_index) const= 0;
    /*! \return name of metric */
    virtual const char *Name( void ) const= 0;
};

struct Pair{
    float key_;
    float value_;
    
    Pair(float key,float value){
      key_ = key;
      value_ = value_;
    }
};

bool PairKeyComparer(const Pair &a, const Pair &b){
    return a.key_ < b.key_;  
} 

bool PairValueComparer(const Pair &a, const Pair &b){
    return a.value_ < b.value_;
}

struct EvalPair : public IRankEvaluator{
    virtual float Eval( const std::vector<float> &preds,
                        const std::vector<float> &labels,
                        const std::vector<int> &group_index  ) const {
	return 0;
    }  
};

/*! \brief Mean Average Precision */
struct EvalMAP : public IRankEvaluator {
    virtual float Eval( const std::vector<float> &preds,
                        const std::vector<float> &labels,
                        const std::vector<int> &group_index  ) const {
	float acc = 0;
	std::vector<Pair> pairs_sort;
	for(int i = 0; i < group_index.size() - 1; i++){
	   for(int j = group_index[i]; j < group_index[i+1];j++){
	      Pair pair(preds[j],labels[j]);
	      pairs_sort.push_back(pair);
	   }
	   acc += average_precision(pairs_sort);
	}
	return acc / (group_index.size() - 1);
    }

    float float average_precision(std::vector<Pair> pairs_sort){
	  std::sort<Pair>(pairs_sort.begin(),pairs_sort.end(),PairKeyComparer);
	  float hits = 0;
	  float average_precision = 0;
	  for(int j = 0; j < pairs_sort.size(); j++){
	    if(pairs_sort[j].value_ == 1){
	      hits++;
	      average_precision += hits/(j+1);
	    }
	  }
	  if(hits != 0) average_precision /= hits;
          return average_precision;      
    }
    
    virtual const char *Name( void ) const {
        return "MAP";
    }

};


/*! \brief Normalized DCG */
struct EvalNDCG : public IRankEvaluator {
    virtual float Eval( const std::vector<float> &preds,
                        const std::vector<float> &labels,
                        const std::vector<int> &group_index ) const {
	float acc = 0;
	std::vector<Pair> pairs_sort;
	for(int i = 0; i < group_index.size() - 1; i++){
	   for(int j = group_index[i]; j < group_index[i+1];j++){
	      Pair pair(preds[j],labels[j]);
	      pairs_sort.push_back(pair);
	   }
	   acc += NDCG(pairs_sort);
	}
    }
    
    float NDCG(std::vector<Pair> pairs_sort){
	std::sort<Pair>(pairs_sort.begin(),pairs_sort.end(),PairKeyComparer);
	float DCG = DCG(pairs_sort);
	std::sort<Pair>(pairs_sort.begin(),pairs_sort.end(),PairValueComparer);
	float IDCG = DCG(pairs_sort);
	if(IDCG == 0) return 0;
	return DCG/IDCG;
    }
    
    float DCG(std::vector<Pair> pairs_sort){
        float ans = 0.0;
	ans += pairs_sort[0].value_;
	for(int i = 1; i < pairs_sort.size(); i++){
	  ans += pairs_sort[i].value_/log(i + 1);
	}
	return ans;
    }
    
    virtual const char *Name( void ) const {
        return "NDCG";
    }
};

};

namespace rank {
/*! \brief a set of evaluators */
struct RankEvalSet {
public:
    inline void AddEval( const char *name ) {
        if( !strcmp( name, "PAIR" )) evals_.push_back( &pair_);
        if( !strcmp( name, "MAP") ) evals_.push_back( &map_ );
        if( !strcmp( name, "NDCG") ) evals_.push_back( &ndcg_ );
    }
    
    inline void Init( void ) {
        std::sort( evals_.begin(), evals_.end() );
        evals_.resize( std::unique( evals_.begin(), evals_.end() ) - evals_.begin() );
    }
    
    inline void Eval( FILE *fo, const char *evname,
                      const std::vector<float> &preds,
                      const std::vector<float> &labels,
                      const std::vector<int> &group_index ) const {
        for( size_t i = 0; i < evals_.size(); ++ i ) {
            float res = evals_[i]->Eval( preds, labels,group_index );
            fprintf( fo, "\t%s-%s:%f", evname, evals_[i]->Name(), res );
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
