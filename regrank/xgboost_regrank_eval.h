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
#include "xgboost_regrank_utils.h"

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
            const DMatrix::Info &info) const = 0;
            /*! \return name of metric */
            virtual const char *Name(void) const = 0;
            /*! \brief virtual destructor */
            virtual ~IEvaluator(void){}
        };

        /*! \brief RMSE */
        struct EvalRMSE : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info) const {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
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
                               const DMatrix::Info &info) const {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                const unsigned ndata = static_cast<unsigned>(preds.size());
                float sum = 0.0f, wsum = 0.0f;
                #pragma omp parallel for reduction(+:sum,wsum) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    const float y = info.labels[i];
                    const float py = preds[i];
                    const float wt = info.GetWeight(i);
                    sum -= wt * (y * std::log(py) + (1.0f - y)*std::log(1 - py));
                    wsum += wt;
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
                               const DMatrix::Info &info) const {
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

        /*! \brief AMS: also records best threshold */
        struct EvalAMS : public IEvaluator{
        public:
            EvalAMS(const char *name){
                name_ = name;
                // note: ams@0 will automatically select which ratio to go
                utils::Assert( sscanf(name, "ams@%f", &ratio_ ) == 1, "invalid ams format" );
            }            
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info) const {
                const unsigned ndata = static_cast<unsigned>(preds.size());
                utils::Assert( info.weights.size() == ndata, "we need weight to evaluate ams");
                std::vector< std::pair<float, unsigned> > rec(ndata);
                
                #pragma omp parallel for schedule( static )                
                for (unsigned i = 0; i < ndata; ++i){
                    rec[i] = std::make_pair( preds[i], i );
                }
                std::sort( rec.begin(), rec.end(), CmpFirst );
                unsigned ntop = static_cast<unsigned>( ratio_ * ndata );
                if( ntop == 0 ) ntop = ndata;
                const double br = 10.0;
                unsigned thresindex = 0;
                double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
                for (unsigned i = 0; i < ndata-1 && i < ntop; ++i){
                    const unsigned ridx = rec[i].second;
                    const float wt = info.weights[ridx];
                    if( info.labels[ridx] > 0.5f ){
                        s_tp += wt;
                    }else{
                        b_fp += wt;
                    }
                    if( rec[i].first != rec[i+1].first ){
                        double ams = sqrtf( 2*((s_tp+b_fp+br) * log( 1.0 + s_tp/(b_fp+br) ) - s_tp) );
                        if( tams < ams ){
                            thresindex = i;
                            tams = ams;
                        }
                    }
                }
                if( ntop == ndata ){
                    fprintf( stderr, "\tams-ratio=%g", float(thresindex)/ndata );
                    return tams;
                }else{
                    return sqrtf( 2*((s_tp+b_fp+br) * log( 1.0 + s_tp/(b_fp+br) ) - s_tp) );
                }
            }
            virtual const char *Name(void) const{
                return name_.c_str();
            }
        private:
            std::string name_;
            float ratio_;
        };

        /*! \brief Error for multi-class classification, need exact match */
        struct EvalMatchError : public IEvaluator{
        public:
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info) const {
                const unsigned ndata = static_cast<unsigned>(preds.size());
                float sum = 0.0f, wsum = 0.0f;
                #pragma omp parallel for reduction(+:sum,wsum) schedule( static )
                for (unsigned i = 0; i < ndata; ++i){
                    const float wt = info.GetWeight(i);
                    int label = static_cast<int>(info.labels[i]);
                    if (static_cast<int>(preds[i]) != label ) sum += wt;
                    wsum += wt;
                }
                return sum / wsum;
            }
            virtual const char *Name(void) const{
                return "merror";
            }
        };

        /*! \brief Area under curve, for both classification and rank */
        struct EvalAuc : public IEvaluator{
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info) const {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                std::vector<unsigned> tgptr(2, 0); tgptr[1] = preds.size();
                const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
                utils::Assert(gptr.back() == preds.size(), "EvalAuc: group structure must match number of prediction");
                const unsigned ngroup = static_cast<unsigned>(gptr.size() - 1);

                double sum_auc = 0.0f;
                #pragma omp parallel reduction(+:sum_auc) 
                {
                    // each thread takes a local rec
                    std::vector< std::pair<float, unsigned> > rec;
                    #pragma omp for schedule(static) 
                    for (unsigned k = 0; k < ngroup; ++k){
                        rec.clear();
                        for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j){
                            rec.push_back(std::make_pair(preds[j], j));
                        }
                        std::sort(rec.begin(), rec.end(), CmpFirst);
                        // calculate AUC
                        double sum_pospair = 0.0;
                        double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
                        for (size_t j = 0; j < rec.size(); ++j){
                            const float wt = info.GetWeight(rec[j].second);
                            const float ctr = info.labels[rec[j].second];
                            // keep bucketing predictions in same bucket
                            if (j != 0 && rec[j].first != rec[j - 1].first){
                                sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
                                sum_npos += buf_pos; sum_nneg += buf_neg;
                                buf_neg = buf_pos = 0.0f;
                            }
                            buf_pos += ctr * wt; buf_neg += (1.0f - ctr) * wt;
                        }
                        sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
                        sum_npos += buf_pos; sum_nneg += buf_neg;
                        // 
                        utils::Assert(sum_npos > 0.0 && sum_nneg > 0.0, "the dataset only contains pos or neg samples");
                        // this is the AUC
                        sum_auc += sum_pospair / (sum_npos*sum_nneg);
                    }
                }
                // return average AUC over list
                return static_cast<float>(sum_auc) / ngroup;
            }
            virtual const char *Name(void) const{
                return "auc";
            }
        };

        /*! \brief Evaluate rank list */          
        struct EvalRankList : public IEvaluator{
        public:
            virtual float Eval(const std::vector<float> &preds,
                               const DMatrix::Info &info) const {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                const std::vector<unsigned> &gptr = info.group_ptr;
                utils::Assert(gptr.size() != 0, "must specify group when constructing rank file");
                utils::Assert( gptr.back() == preds.size(), "EvalRanklist: group structure must match number of prediction");
                const unsigned ngroup = static_cast<unsigned>(gptr.size() - 1);

                double sum_metric = 0.0f;
                #pragma omp parallel reduction(+:sum_metric) 
                {
                    // each thread takes a local rec
                    std::vector< std::pair<float, unsigned> > rec;
                    #pragma omp for schedule(static) 
                    for (unsigned k = 0; k < ngroup; ++k){
                        rec.clear();
                        for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j){
                            rec.push_back(std::make_pair(preds[j], (int)info.labels[j]));
                        }
                        sum_metric += this->EvalMetric( rec );                        
                    }
                }
                return static_cast<float>(sum_metric) / ngroup;
            }
            virtual const char *Name(void) const{
                return name_.c_str();
            }
        protected:
            EvalRankList(const char *name){
                name_ = name;
                if( sscanf(name, "%*[^@]@%u", &topn_) != 1 ){
                    topn_ = UINT_MAX;
                }
            }
            /*! \return evaluation metric, given the pair_sort record, (pred,label) */
            virtual float EvalMetric( std::vector< std::pair<float, unsigned> > &pair_sort ) const = 0;
        protected:
            unsigned topn_;
            std::string name_;
        };
        
        /*! \brief Precison at N, for both classification and rank */
        struct EvalPrecision : public EvalRankList{
        public:
            EvalPrecision(const char *name):EvalRankList(name){}
        protected:
            virtual float EvalMetric( std::vector< std::pair<float, unsigned> > &rec ) const {
                // calculate Preicsion
                std::sort(rec.begin(), rec.end(), CmpFirst);
                unsigned nhit = 0;
                for (size_t j = 0; j < rec.size() && j < this->topn_; ++j){
                    nhit += (rec[j].second != 0 );
                }
                return static_cast<float>( nhit ) / topn_;
            }
        };


        /*! \brief NDCG */
        struct EvalNDCG : public EvalRankList{
        public:
            EvalNDCG(const char *name):EvalRankList(name){}
        protected:
            inline float CalcDCG( const std::vector< std::pair<float,unsigned> > &rec ) const {
                double sumdcg = 0.0;
                for( size_t i = 0; i < rec.size() && i < this->topn_; i ++ ){
                    const unsigned rel = rec[i].second;
                    if( rel != 0 ){ 
                        sumdcg += logf(2.0f) * ((1<<rel)-1) / logf( i + 2 );
                    }
                }
                return static_cast<float>(sumdcg);
            }
            virtual float EvalMetric( std::vector< std::pair<float, unsigned> > &rec ) const {
                std::sort(rec.begin(), rec.end(), CmpSecond);
                float idcg = this->CalcDCG(rec);
                std::sort(rec.begin(), rec.end(), CmpFirst);
                float dcg = this->CalcDCG(rec);
                if( idcg == 0.0f ) return 0.0f;
                else return dcg/idcg;
            }
        };

        /*! \brief Precison at N, for both classification and rank */
        struct EvalMAP : public EvalRankList{
        public:
            EvalMAP(const char *name):EvalRankList(name){}
        protected:
            virtual float EvalMetric( std::vector< std::pair<float, unsigned> > &rec ) const {
                std::sort(rec.begin(), rec.end(), CmpFirst);
                unsigned nhits = 0;
                double sumap = 0.0;
                for( size_t i = 0; i < rec.size(); ++i){
                    if( rec[i].second != 0 ){
                        nhits += 1;
                        if( i < this->topn_ ){
                            sumap += static_cast<float>(nhits) / (i+1);
                        }
                    }
                }
                if (nhits != 0) sumap /= nhits;
                return static_cast<float>(sumap);                
            }
        };
    };

    namespace regrank{
        /*! \brief a set of evaluators */
        struct EvalSet{
        public:
            inline void AddEval(const char *name){
                for (size_t i = 0; i < evals_.size(); ++i){
                    if (!strcmp(name, evals_[i]->Name())) return;
                }
                if (!strcmp(name, "rmse"))    evals_.push_back(new EvalRMSE());
                if (!strcmp(name, "error"))   evals_.push_back(new EvalError());
                if (!strcmp(name, "merror"))   evals_.push_back(new EvalMatchError());
                if (!strcmp(name, "logloss")) evals_.push_back(new EvalLogLoss());
                if (!strcmp(name, "auc"))    evals_.push_back(new EvalAuc());
                if (!strncmp(name, "ams@",4))  evals_.push_back(new EvalAMS(name));
                if (!strncmp(name, "pre@", 4)) evals_.push_back(new EvalPrecision(name));
                if (!strncmp(name, "map", 3))   evals_.push_back(new EvalMAP(name));
                if (!strncmp(name, "ndcg", 3))  evals_.push_back(new EvalNDCG(name));
            }
            ~EvalSet(){
                for (size_t i = 0; i < evals_.size(); ++i){
                    delete evals_[i];
                }
            }
            inline void Eval(FILE *fo, const char *evname,
                const std::vector<float> &preds,
                const DMatrix::Info &info) const{
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
