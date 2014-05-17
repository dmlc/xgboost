#ifndef XGBOOST_REGRANK_OBJ_H
#define XGBOOST_REGRANK_OBJ_H
/*!
 * \file xgboost_regrank_obj.h
 * \brief defines objective function interface used in xgboost for regression and rank
 * \author Tianqi Chen, Kailong Chen
 */
#include "xgboost_regrank_data.h"

namespace xgboost{
    namespace regrank{
        /*! \brief interface of objective function */
        class IObjFunction{
        public:
            /*! \brief virtual destructor */
            virtual ~IObjFunction(void){}
            /*!
             * \brief set parameters from outside
             * \param name name of the parameter
             * \param val  value of the parameter
             */
            virtual void SetParam(const char *name, const char *val) = 0;
            
            /*! 
             * \brief get gradient over each of predictions, given existing information
             * \param preds prediction of current round             
             * \param info information about labels, weights, groups in rank
             * \param iter current iteration number 
             * \param grad gradient over each preds
             * \param hess second order gradient over each preds
             */
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) = 0;
            /*! \return the default evaluation metric for the problem */
            virtual const char* DefaultEvalMetric(void) = 0;
            /*! 
             * \brief transform prediction values, this is only called when Prediction is called
             * \param preds prediction values, saves to this vector as well
             */
            virtual void PredTransform(std::vector<float> &preds){}
        };
    };
    
    namespace regrank{
        /*! \brief defines functions to calculate some commonly used functions */
        struct LossType{
        public:
            const static int kLinearSquare = 0;
            const static int kLogisticNeglik = 1;
            const static int kLogisticClassify = 2;
            const static int kLogisticRaw = 3;
        public:
            /*! \brief indicate which type we are using */
            int loss_type;
        public:
            /*!
             * \brief transform the linear sum to prediction
             * \param x linear sum of boosting ensemble
             * \return transformed prediction
             */
            inline float PredTransform(float x){
                switch (loss_type){
                case kLogisticRaw: 
                case kLinearSquare: return x;
                case kLogisticClassify:
                case kLogisticNeglik: return 1.0f / (1.0f + expf(-x));
                default: utils::Error("unknown loss_type"); return 0.0f;
                }
            }
            
            /*!
             * \brief calculate first order gradient of loss, given transformed prediction
             * \param predt transformed prediction
             * \param label true label
             * \return first order gradient
             */
            inline float FirstOrderGradient(float predt, float label) const{
                switch (loss_type){
                case kLinearSquare: return predt - label;
                case kLogisticRaw: predt = 1.0f / (1.0f + expf(-predt));
                case kLogisticClassify:
                case kLogisticNeglik: return predt - label;
                default: utils::Error("unknown loss_type"); return 0.0f;
                }
            }
            /*!
             * \brief calculate second order gradient of loss, given transformed prediction
             * \param predt transformed prediction
             * \param label true label
             * \return second order gradient
             */
            inline float SecondOrderGradient(float predt, float label) const{
                switch (loss_type){
                case kLinearSquare: return 1.0f;
                case kLogisticRaw: predt = 1.0f / (1.0f + expf(-predt));
                case kLogisticClassify:
                case kLogisticNeglik: return predt * (1 - predt);
                default: utils::Error("unknown loss_type"); return 0.0f;
                }
            }
        };
    };
};

#include "xgboost_regrank_obj.hpp"

namespace xgboost{
    namespace regrank{        
       inline IObjFunction* CreateObjFunction( const char *name ){
           if( !strcmp("reg:linear", name ) )     return new RegressionObj( LossType::kLinearSquare );
           if( !strcmp("reg:logistic", name ) )    return new RegressionObj( LossType::kLogisticNeglik );
           if( !strcmp("binary:logistic", name ) ) return new RegressionObj( LossType::kLogisticClassify );
           if( !strcmp("binary:logitraw", name ) ) return new RegressionObj( LossType::kLogisticRaw );
           if( !strcmp("multi:softmax", name ) )   return new SoftmaxMultiClassObj();
           if( !strcmp("rank:pairwise", name ) ) return new PairwiseRankObj();
           if( !strcmp("rank:pairwise", name ) ) return new PairwiseRankObj();
           if( !strcmp("rank:softmax", name ) )  return new SoftmaxRankObj();
           utils::Error("unknown objective function type");
           return NULL;
       }
    };
};
#endif
