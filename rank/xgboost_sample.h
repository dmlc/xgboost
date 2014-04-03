#ifndef _XGBOOST_SAMPLE_H_
#define _XGBOOST_SAMPLE_H_

#include"../utils/xgboost_utils.h"

namespace xgboost {
  namespace rank {
    namespace sample {

      struct Pairs {

      /*
       * \brief retrieve the related pair information of an data instances
       * \param index, the index of retrieved instance
       * \return the index of instances paired
       */
      std::vector<int> GetPairs(int index) {
        utils::assert(index >= start_ && index < end_, "The query index out of sampling bound");
      }

      std::vector<std::vector<int>> pairs_;
      int start_;
      int end_;
    };

      struct IPairSampler {
	  /*
	   * \brief Generate sample pairs given the predcions, labels, the start and the end index 
	   *        of a specified group
	   * \param preds, the predictions of all data instances
	   * \param labels, the labels of all data instances
	   * \param start, the start index of a specified group
	   * \param end, the end index of a specified group
	   * \return the generated pairs
	   */
	  virtual Pairs GenPairs(const std::vector<float> &preds,
				const std::vector<float> &labels,
			      int start,int end) = 0;
      };
      
      /*! \brief a set of evaluators */
        struct PairSamplerSet{
        public:
            inline void AssignSampler( const char *name ){                
                if( !strcmp( name, "rmse") ) evals_.push_back( &rmse_ );
                if( !strcmp( name, "error") ) evals_.push_back( &error_ );
                if( !strcmp( name, "logloss") ) evals_.push_back( &logloss_ );
            }
            
            
            Pairs GenPairs(const std::vector<float> &preds,
			   const std::vector<float> &labels,
			   int start,int end){
			
	      
	    }
        private:
            EvalRMSE  rmse_;
            EvalError error_;
            EvalLogLoss logloss_;
            std::vector<const IEvaluator*> evals_;  
        };
    }
  }
}

