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
#include "xgboost_rank_data.h"
#include "xgboost_rank_eval.h"
#include "../utils/xgboost_omp.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost {
namespace rank {
/*! \brief class for gradient boosted regression */
class RankBoostLearner {
public:
    /*! \brief constructor */
    RegBoostLearner( void ) {
        silent = 0;
    }
    /*!
    * \brief a rank booster associated with training and evaluating data
    * \param train pointer to the training data
    * \param evals array of evaluating data
    * \param evname name of evaluation data, used print statistics
    */
    RankBoostLearner( const RMatrix *train,
                      const std::vector<RMatrix *> &evals,
                      const std::vector<std::string> &evname ) {
        silent = 0;
        this->SetData(train,evals,evname);
    }

    /*!
    * \brief associate rank booster with training and evaluating data
    * \param train pointer to the training data
    * \param evals array of evaluating data
    * \param evname name of evaluation data, used print statistics
    */
    inline void SetData( const RMatrix *train,
                         const std::vector<RMatrix *> &evals,
                         const std::vector<std::string> &evname ) {
        this->train_ = train;
        this->evals_ = evals;
        this->evname_ = evname;
        // estimate feature bound
        int num_feature = (int)(train->data.NumCol());
        // assign buffer index
        unsigned buffer_size = static_cast<unsigned>( train->Size() );

        for( size_t i = 0; i < evals.size(); ++ i ) {
            buffer_size += static_cast<unsigned>( evals[i]->Size() );
            num_feature = std::max( num_feature, (int)(evals[i]->data.NumCol()) );
        }

        char str_temp[25];
        if( num_feature > mparam.num_feature ) {
            mparam.num_feature = num_feature;
            sprintf( str_temp, "%d", num_feature );
            base_gbm.SetParam( "bst:num_feature", str_temp );
        }

        sprintf( str_temp, "%u", buffer_size );
        base_gbm.SetParam( "num_pbuffer", str_temp );
        if( !silent ) {
            printf( "buffer_size=%u\n", buffer_size );
        }

        // set eval_preds tmp sapce
        this->eval_preds_.resize( evals.size(), std::vector<float>() );
    }
    /*!
    * \brief set parameters from outside
    * \param name name of the parameter
    * \param val  value of the parameter
    */
    inline void SetParam( const char *name, const char *val ) {
        if( !strcmp( name, "silent") )  silent = atoi( val );
        if( !strcmp( name, "eval_metric") )  evaluator_.AddEval( val );
        mparam.SetParam( name, val );
        base_gbm.SetParam( name, val );
    }
    /*!
    * \brief initialize solver before training, called before training
    * this function is reserved for solver to allocate necessary space and do other preparation
    */
    inline void InitTrainer( void ) {
        base_gbm.InitTrainer();
        if( mparam.loss_type == PAIRWISE) {
            evaluator_.AddEval( "PAIR" );
        } else if( mparam.loss_type == MAP) {
            evaluator_.AddEval( "MAP" );
        } else {
            evaluator_.AddEval( "NDCG" );
        }
        evaluator_.Init();
	sampler.AssignSampler(mparam.sampler_type);
    }
    /*!
    * \brief initialize the current data storage for model, if the model is used first time, call this function
    */
    inline void InitModel( void ) {
        base_gbm.InitModel();
    }
    /*!
    * \brief load model from stream
    * \param fi input stream
    */
    inline void LoadModel( utils::IStream &fi ) {
        base_gbm.LoadModel( fi );
        utils::Assert( fi.Read( &mparam, sizeof(ModelParam) ) != 0 );
    }
    /*!
     * \brief DumpModel
     * \param fo text file
     * \param fmap feature map that may help give interpretations of feature
      * \param with_stats whether print statistics as well
     */
    inline void DumpModel( FILE *fo, const utils::FeatMap& fmap, bool with_stats ) {
        base_gbm.DumpModel( fo, fmap, with_stats );
    }
    /*!
     * \brief Dump path of all trees
     * \param fo text file
     * \param data input data
     */
    inline void DumpPath( FILE *fo, const RMatrix &data ) {
        base_gbm.DumpPath( fo, data.data );
    }
    
    /*!
    * \brief save model to stream
    * \param fo output stream
    */
    inline void SaveModel( utils::IStream &fo ) const {
        base_gbm.SaveModel( fo );
        fo.Write( &mparam, sizeof(ModelParam) );
    }
    
    /*!
     * \brief update the model for one iteration
     * \param iteration iteration number
     */
    inline void UpdateOneIter( int iter ) {
        this->PredictBuffer( preds_, *train_, 0 );
        this->GetGradient( preds_, train_->labels,train_->group_index, grad_, hess_ );
        std::vector<unsigned> root_index;
        base_gbm.DoBoost( grad_, hess_, train_->data, root_index );
    }
    /*!
     * \brief evaluate the model for specific iteration
     * \param iter iteration number
     * \param fo file to output log
     */
    inline void EvalOneIter( int iter, FILE *fo = stderr ) {
        fprintf( fo, "[%d]", iter );
        int buffer_offset = static_cast<int>( train_->Size() );

        for( size_t i = 0; i < evals_.size(); ++i ) {
            std::vector<float> &preds = this->eval_preds_[ i ];
            this->PredictBuffer( preds, *evals_[i], buffer_offset);
            evaluator_.Eval( fo, evname_[i].c_str(), preds, (*evals_[i]).labels );
            buffer_offset += static_cast<int>( evals_[i]->Size() );
        }
        fprintf( fo,"\n" );
    }
    
    /*! \brief get intransformed prediction, without buffering */
    inline void Predict( std::vector<float> &preds, const DMatrix &data ) {
        preds.resize( data.Size() );

        const unsigned ndata = static_cast<unsigned>( data.Size() );
        #pragma omp parallel for schedule( static )
        for( unsigned j = 0; j < ndata; ++ j ) {
            preds[j] = base_gbm.Predict( data.data, j, -1 );
        }
    }
    
public:
    /*!
     * \brief update the model for one iteration
     * \param iteration iteration number
     */
    inline void UpdateInteract( std::string action ) {
        this->InteractPredict( preds_, *train_, 0 );

        int buffer_offset = static_cast<int>( train_->Size() );
        for( size_t i = 0; i < evals_.size(); ++i ) {
            std::vector<float> &preds = this->eval_preds_[ i ];
            this->InteractPredict( preds, *evals_[i], buffer_offset );
            buffer_offset += static_cast<int>( evals_[i]->Size() );
        }

        if( action == "remove" ) {
            base_gbm.DelteBooster();
            return;
        }

        this->GetGradient( preds_, train_->labels, grad_, hess_ );
        std::vector<unsigned> root_index;
        base_gbm.DoBoost( grad_, hess_, train_->data, root_index );

        this->InteractRePredict( *train_, 0 );
        buffer_offset = static_cast<int>( train_->Size() );
        for( size_t i = 0; i < evals_.size(); ++i ) {
            this->InteractRePredict( *evals_[i], buffer_offset );
            buffer_offset += static_cast<int>( evals_[i]->Size() );
        }
    }
private:
    /*! \brief get the transformed predictions, given data */
    inline void InteractPredict( std::vector<float> &preds, const DMatrix &data, unsigned buffer_offset ) {
        preds.resize( data.Size() );
        const unsigned ndata = static_cast<unsigned>( data.Size() );
        #pragma omp parallel for schedule( static )
        for( unsigned j = 0; j < ndata; ++ j ) {
            preds[j] = base_gbm.InteractPredict( data.data, j, buffer_offset + j );
        }
    }
    /*! \brief repredict trial */
    inline void InteractRePredict( const DMatrix &data, unsigned buffer_offset ) {
        const unsigned ndata = static_cast<unsigned>( data.Size() );
        #pragma omp parallel for schedule( static )
        for( unsigned j = 0; j < ndata; ++ j ) {
            base_gbm.InteractRePredict( data.data, j, buffer_offset + j );
        }
    }
private:
    /*! \brief get intransformed predictions, given data */
    inline void PredictBuffer( std::vector<float> &preds, const RMatrix &data, unsigned buffer_offset ) {
        preds.resize( data.Size() );

        const unsigned ndata = static_cast<unsigned>( data.Size() );
        #pragma omp parallel for schedule( static )
        for( unsigned j = 0; j < ndata; ++ j ) {
            preds[j] = base_gbm.Predict( data.data, j, buffer_offset + j );
      
	}
    }

    /*! \brief get the first order and second order gradient, given the transformed predictions and labels */
    inline void GetGradient( const std::vector<float> &preds,
                             const std::vector<float> &labels,
                             const std::vector<int> &group_index,
                             std::vector<float> &grad,
                             std::vector<float> &hess ) {
        grad.resize( preds.size() );
        hess.resize( preds.size() );
	bool j_better;
	float pred_diff,pred_diff_exp,first_order_gradient,second_order_gradient;
	for(int i = 0; i < group_index.size() - 1; i++){
	  
	  sample::Pairs pairs = sampler.GenPairs(preds,labels,group_index[i],group_index[i+1]);
	  for(int j = group_index[i]; j < group_index[i + 1]; j++){
	      std::vector<int> pair_instance = pairs.GetPairs(j);
	      for(int k = 0; k < pair_instance.size(); k++){
		 j_better =  labels[j] > labels[pair_instance[k]];
	         if(j_better){
	   	     pred_diff = preds[preds[j] - pair_instance[k]];
		     pred_diff_exp =  j_better? expf(-pred_diff):expf(pred_diff);
                     first_order_gradient = mparam.FirstOrderGradient(pred_diff_exp);	    
	             second_order_gradient = 2 * mparam.SecondOrderGradient(pred_diff_exp);	    
	             hess[j] += second_order_gradient;
	             grad[j] += first_order_gradient;
	             hess[pair_instance[k]] += second_order_gradient;
		     grad[pair_instance[k]] += -first_order_gradient;
	         }
	      }
	  }
	}
      
    }

private:
    enum LossType {
        PAIRWISE = 0,
        MAP = 1,
        NDCG = 2
    };
    
    /*! \brief training parameter for regression */
    struct ModelParam {
        /* \brief type of loss function */
        int loss_type;
        /* \brief number of features  */
        int num_feature;
        /*! \brief reserved field */
        int reserved[ 16 ];
	/*! \brief sampler type */
	int sampler_type;
        /*! \brief constructor */
        ModelParam( void ) {
            loss_type  = 0;
            num_feature = 0;
            memset( reserved, 0, sizeof( reserved ) );
        }
        /*!
        * \brief set parameters from outside
        * \param name name of the parameter
        * \param val  value of the parameter
        */
        inline void SetParam( const char *name, const char *val ) {
            if( !strcmp("loss_type", name ) )   loss_type = atoi( val );
            if( !strcmp("bst:num_feature", name ) ) num_feature = atoi( val );
	    if( !strcmp("rank:sampler",name)) sampler = atoi( val );
        }


        /*!
        * \brief calculate first order gradient of pairwise loss function(f(x) = ln(1+exp(-x)), 
	* given the exponential of the difference of intransformed pair predictions
        * \param the intransformed prediction of positive instance
        * \param the intransformed prediction of negative instance
        * \return first order gradient
        */
        inline float FirstOrderGradient( float pred_diff_exp) const {
	   return -pred_diff_exp/(1 + pred_diff_exp);
        }
        
        /*!
        * \brief calculate second order gradient of pairwise loss function(f(x) = ln(1+exp(-x)), 
	* given the exponential of the difference of intransformed pair predictions
        * \param the intransformed prediction of positive instance
        * \param the intransformed prediction of negative instance
        * \return second order gradient
        */
        inline float SecondOrderGradient( float pred_diff_exp ) const {
            return pred_diff_exp/pow(1 + pred_diff_exp,2);
        }
    };
private:
    int silent;
    RankEvalSet evaluator_;
    sample::PairSamplerWrapper sampler;
    booster::GBMBase base_gbm;
    ModelParam   mparam;
    const RMatrix *train_;
    std::vector<RMatrix *> evals_;
    std::vector<std::string> evname_;
    std::vector<unsigned> buffer_index_;
private:
    std::vector<float> grad_, hess_, preds_;
    std::vector< std::vector<float> > eval_preds_;
};
}
};

#endif





