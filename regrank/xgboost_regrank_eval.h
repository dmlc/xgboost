#ifndef XGBOOST_REGRANK_EVAL_H
#define XGBOOST_REGRANK_EVAL_H
/*!
* \file xgboost_regrank_eval.h
* \brief evaluation metrics for regression and classification and rank
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_omp.h"
#include "../utils/xgboost_random.h"
#include "xgboost_regrank_data.h"

namespace xgboost{
    namespace regrank{
        /*! \brief evaluator that evaluates the loss metrics */
        struct IEvaluator{
            /*!
             * \brief evaluate a specific metric
             * \param preds prediction
             * \param info information, including label etc.
             */
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info ) const = 0;
            /*! \return name of metric */
            virtual const char *Name(void) const = 0;
        };

        /*! \brief RMSE */
        struct EvalRMSE : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info ) const {
                const unsigned ndata = static_cast<unsigned>(preds.size());
                float sum = 0.0, wsum = 0.0;
                #pragma omp parallel for reduction(+:sum,wsum) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    const float wt = info.GetWeight(i);
                    const float diff = info.labels[i] - preds[i];
                    sum += diff*diff * wt;
                    wsum += wt;
                }
                return sqrtf(sum / wsum);
            }
            virtual const char *Name(void) const{
                return "rmse";
            }
        };

        /*! \brief Error */
        struct EvalLogLoss : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info ) const {
                const unsigned ndata = static_cast<unsigned>(preds.size());
                float sum = 0.0f, wsum = 0.0f;                
                #pragma omp parallel for reduction(+:sum,wsum) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    const float y = info.labels[i];
                    const float py = preds[i];
                    const float wt = info.GetWeight(i);
                    sum -= wt * ( y * std::log(py) + (1.0f - y)*std::log(1 - py) );
                    wsum+= wt;
                }
                return sum / wsum;
            }
            virtual const char *Name(void) const{
                return "negllik";
            }
        };

        /*! \brief Error */
        struct EvalError : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info ) const {
                const unsigned ndata = static_cast<unsigned>(preds.size());
                float sum = 0.0f, wsum = 0.0f;                
                #pragma omp parallel for reduction(+:sum,wsum) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    const float wt = info.GetWeight(i);
                    if (preds[i] > 0.5f){
                        if (info.labels[i] < 0.5f) sum += wt; 
                     }
                    else{
                        if (info.labels[i] >= 0.5f) sum += wt;
                    }
                    wsum += wt;
                }
                return sum / wsum;
            }
            virtual const char *Name(void) const{
                return "error";
            }
        };

        /*! \brief Area under curve */
        struct EvalAuc : public IEvaluator{         
            inline static bool CmpFirst( const std::pair<float,float> &a, const std::pair<float,float> &b ){
                return a.first > b.first;
            }
            virtual float Eval( const std::vector<float> &preds, 
                                const DMatrix::Info &info ) const {
                const std::vector<float> &labels  = info.labels;
                const unsigned ndata = static_cast<unsigned>( preds.size() );
                std::vector< std::pair<float, float> > rec;
                for( unsigned i = 0; i < ndata; ++ i ){
                    rec.push_back( std::make_pair( preds[i], labels[i]) );
                }
                random::Shuffle( rec );
                std::sort( rec.begin(), rec.end(), CmpFirst );

                long npos = 0, nhit = 0;
                for( unsigned i = 0; i < ndata; ++ i ){
                    if( rec[i].second > 0.5f ) {
                        ++ npos; 
                    }else{
                        // this is the number of correct pairs
                        nhit += npos;
                    }
                } 
                long nneg = ndata - npos;
                utils::Assert( nneg > 0, "the dataset only contains pos samples" );
                return static_cast<float>(nhit) / nneg / npos;
            }
            virtual const char *Name( void ) const{
                return "auc";
            }
        };
    };

    namespace regrank{
        /*! \brief a set of evaluators */
        struct EvalSet{
        public:
            inline void AddEval(const char *name){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    if(!strcmp(name, evals_[i]->Name())) return;
                }
                if (!strcmp(name, "rmse"))    evals_.push_back( new EvalRMSE() );
                if (!strcmp(name, "error"))   evals_.push_back( new EvalError() );
                if (!strcmp(name, "logloss")) evals_.push_back( new EvalLogLoss() );
                if (!strcmp( name, "auc"))    evals_.push_back( new EvalAuc() );
            }
            ~EvalSet(){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    delete evals_[i];
                }
            }
            inline void Eval(FILE *fo, const char *evname,
                             const std::vector<float> &preds,
                             const DMatrix::Info &info ) const{
                for (size_t i = 0; i < evals_.size(); ++i){
                    float res = evals_[i]->Eval(preds, info);
                    fprintf(fo, "\t%s-%s:%f", evname, evals_[i]->Name(), res);
                }
            }
        private:
            std::vector<const IEvaluator*> evals_;
        };
    };
};
#endif
