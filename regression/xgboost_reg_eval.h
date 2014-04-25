#ifndef XGBOOST_REG_EVAL_H
#define XGBOOST_REG_EVAL_H
/*!
* \file xgboost_reg_eval.h
* \brief evaluation metrics for regression and classification
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/

#include <cmath>
#include <vector>
#include <algorithm>
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_omp.h"
#include "../utils/xgboost_random.h"

namespace xgboost{
    namespace regression{
        /*! \brief evaluator that evaluates the loss metrics */
        struct IEvaluator{
            /*!
             * \brief evaluate a specific metric
             * \param preds prediction
             * \param labels label
             */
            virtual float Eval(const std::vector<float> &preds,
            const std::vector<float> &labels) const = 0;
            /*! \return name of metric */
            virtual const char *Name(void) const = 0;
        };

        /*! \brief RMSE */
        struct EvalRMSE : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const std::vector<float> &labels) const{
                const unsigned ndata = static_cast<unsigned>(preds.size());
                float sum = 0.0;
                #pragma omp parallel for reduction(+:sum) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    float diff = preds[i] - labels[i];
                    sum += diff * diff;
                }
                return sqrtf(sum / ndata);
            }
            virtual const char *Name(void) const{
                return "rmse";
            }
        };

        /*! \brief Error */
        struct EvalError : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const std::vector<float> &labels) const{
                const unsigned ndata = static_cast<unsigned>(preds.size());
                unsigned nerr = 0;
                #pragma omp parallel for reduction(+:nerr) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    if (preds[i] > 0.5f){
                        if (labels[i] < 0.5f) nerr += 1;
                    }
                    else{
                        if (labels[i] > 0.5f) nerr += 1;
                    }
                }
                return static_cast<float>(nerr) / ndata;
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
                                const std::vector<float> &labels ) const{
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

        /*! \brief Error */
        struct EvalLogLoss : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const std::vector<float> &labels) const{
                const unsigned ndata = static_cast<unsigned>(preds.size());
                unsigned nerr = 0;
                #pragma omp parallel for reduction(+:nerr) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    const float y = labels[i];
                    const float py = preds[i];
                    nerr -= y * std::log(py) + (1.0f - y)*std::log(1 - py);
                }
                return static_cast<float>(nerr) / ndata;
            }
            virtual const char *Name(void) const{
                return "negllik";
            }
        };
    };

    namespace regression{
        /*! \brief a set of evaluators */
        struct EvalSet{
        public:
            inline void AddEval(const char *name){
                if (!strcmp(name, "rmse")) evals_.push_back(&rmse_);
                if (!strcmp(name, "error")) evals_.push_back(&error_);
                if (!strcmp(name, "logloss")) evals_.push_back(&logloss_);
                if (!strcmp( name, "auc"))   evals_.push_back( &auc_ );
            }
            inline void Init(void){
                std::sort(evals_.begin(), evals_.end());
                evals_.resize(std::unique(evals_.begin(), evals_.end()) - evals_.begin());
            }
            inline void Eval(FILE *fo, const char *evname,
                             const std::vector<float> &preds,
                             const std::vector<float> &labels) const{
                for (size_t i = 0; i < evals_.size(); ++i){
                    float res = evals_[i]->Eval(preds, labels);
                    fprintf(fo, "\t%s-%s:%f", evname, evals_[i]->Name(), res);
                }
            }
        private:
            EvalRMSE  rmse_;
            EvalError error_;
            EvalAuc   auc_;
            EvalLogLoss logloss_;
            std::vector<const IEvaluator*> evals_;
        };
    };
};
#endif
