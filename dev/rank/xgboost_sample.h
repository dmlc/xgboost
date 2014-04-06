#ifndef _XGBOOST_SAMPLE_H_
#define _XGBOOST_SAMPLE_H_

#include <vector>
#include"../utils/xgboost_utils.h"

namespace xgboost {
  namespace rank {
    namespace sample {

      /*
       * \brief the data structure to maintain the sample pairs
       */
      struct Pairs {

      /*
       * \brief constructor given the start and end offset of the sampling group
       *        in overall instances
       * \param start the begin index of the group
       * \param end the end index of the group
       */
      Pairs(int start,int end):start_(start),end_(end_){
	for(int i = start; i < end; i++){
	  std::vector<int> v;
	  pairs_.push_back(v);
	}
      }
      /*
       * \brief retrieve the related pair information of an data instances
       * \param index, the index of retrieved instance
       * \return the index of instances paired
       */
      std::vector<int> GetPairs(int index) {
	utils::Assert(index >= start_ && index < end_,"The query index out of sampling bound");
        return pairs_[index-start_];
      }

      /*
       * \brief add in a sampled pair
       * \param index the index of the instance to sample a friend
       * \param paired_index the index of the instance sampled as a friend
       */
      void push(int index,int paired_index){
	pairs_[index - start_].push_back(paired_index);
      }
      
      std::vector< std::vector<int> > pairs_;
      int start_;
      int end_;
    };

      /*
       * \brief the interface of pair sampler
       */
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
      
      enum{
	  BINARY_LINEAR_SAMPLER
      };
      
      /*! \brief A simple pair sampler when the rank relevence scale is binary
       *         for each positive instance, we will pick a negative
       *         instance and add in a pair. When using binary linear sampler, 
       *         we should guarantee the labels are 0 or 1
       */
      struct BinaryLinearSampler:public IPairSampler{
	virtual Pairs GenPairs(const std::vector<float> &preds,
			       const std::vector<float> &labels,
			       int start,int end) {
  	    Pairs pairs(start,end);
	    int pointer = 0, last_pointer = 0,index = start, interval = end - start;
	    for(int i = start; i < end; i++){
	      if(labels[i] == 1){
		while(true){
		  index = (++pointer) % interval + start;
		  if(labels[index] == 0) break;
		  if(pointer - last_pointer > interval) return pairs;
		} 
		pairs.push(i,index);
		pairs.push(index,i);
		last_pointer = pointer;
	      }
	    }
	    return pairs; 
	}	
      };
      
      
      /*! \brief Pair Sampler Wrapper*/
        struct PairSamplerWrapper{
        public:
            inline void AssignSampler( int sampler_index ){                
            
	      switch(sampler_index){
		case BINARY_LINEAR_SAMPLER:sampler_ = &binary_linear_sampler;break;
	
		default:utils::Error("Cannot find the specified sampler");
	      }
	    }
            
            Pairs GenPairs(const std::vector<float> &preds,
			   const std::vector<float> &labels,
			   int start,int end){
	      return sampler_->GenPairs(preds,labels,start,end);
	    }
        private:
	    BinaryLinearSampler binary_linear_sampler;
	    IPairSampler *sampler_; 
        };
    }
  }
}
#endif