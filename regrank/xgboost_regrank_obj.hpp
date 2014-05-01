#ifndef XGBOOST_REGRANK_OBJ_HPP
#define XGBOOST_REGRANK_OBJ_HPP
/*!
 * \file xgboost_regrank_obj.h
 * \brief implementation of objective functions
 * \author Tianqi Chen, Kailong Chen
 */
//#include "xgboost_regrank_sample.h"
#include <vector>

namespace xgboost{
    namespace regrank{        
        class RegressionObj : public IObjFunction{
        public:
            RegressionObj(void){
                loss.loss_type = LossType::kLinearSquare;
            }
            virtual ~RegressionObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "loss_type", name ) ) loss.loss_type = atoi( val );
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                grad.resize(preds.size()); hess.resize(preds.size());

                const unsigned ndata = static_cast<unsigned>(preds.size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    float p = loss.PredTransform(preds[j]);
                    grad[j] = loss.FirstOrderGradient(p, info.labels[j]) * info.GetWeight(j);
                    hess[j] = loss.SecondOrderGradient(p, info.labels[j]) * info.GetWeight(j);
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                if( loss.loss_type == LossType::kLogisticClassify ) return "error";
                else return "rmse";
            }
            virtual void PredTransform(std::vector<float> &preds){
                const unsigned ndata = static_cast<unsigned>(preds.size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = loss.PredTransform( preds[j] );
                }
            }
        private:
            LossType loss;
        };
    };

    namespace regrank{
        // simple softmax rak
        class SoftmaxObj : public IObjFunction{
        public:
            SoftmaxObj(void){
            }
            virtual ~SoftmaxObj(){}
            virtual void SetParam(const char *name, const char *val){
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                grad.resize(preds.size()); hess.resize(preds.size());
                const std::vector<unsigned> &gptr = info.group_ptr;
                utils::Assert( gptr.size() != 0 && gptr.back() == preds.size(), "rank loss must have group file" );
                const unsigned ngroup = static_cast<unsigned>( gptr.size() - 1 );

                #pragma omp parallel
                {
                    std::vector< float > rec;                    
                    #pragma for schedule(static)
                    for (unsigned k = 0; k < ngroup; ++k){
                        rec.clear();
                        int nhit = 0;
                        for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                            rec.push_back( preds[j] );
                            grad[j] = hess[j] = 0.0f;
                            nhit += info.labels[j];
                        }
                        Softmax( rec );
                        if( nhit == 1 ){
                            for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                                float p = rec[ j - gptr[k] ];
                                grad[j] = p - info.labels[j];
                                hess[j] = 2.0f * p * ( 1.0f - p );
                            }  
                        }else{
                            utils::Assert( nhit == 0, "softmax does not allow multiple labels" );
                        }
                    }
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                return "pre@1";
            }            
        private:
            inline static void Softmax( std::vector<float>& rec ){
                float wmax = rec[0];
                for( size_t i = 1; i < rec.size(); ++ i ){
                    wmax = std::max( rec[i], wmax );
                }
                double wsum = 0.0f;
                for( size_t i = 0; i < rec.size(); ++ i ){
                    rec[i] = expf(rec[i]-wmax);
                    wsum += rec[i];
                }
                for( size_t i = 0; i < rec.size(); ++ i ){
                    rec[i] /= wsum;
                }                
            }
        };
    };

    namespace regrank{
        // simple pairwise rank 
        class PairwiseRankObj : public IObjFunction{
        public:
            PairwiseRankObj(void){
                loss.loss_type = LossType::kLinearSquare;
                fix_list_weight = 0.0f;
            }
            virtual ~PairwiseRankObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "loss_type", name ) ) loss.loss_type = atoi( val );
                if( !strcmp( "fix_list_weight", name ) ) fix_list_weight = (float)atof( val );
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                grad.resize(preds.size()); hess.resize(preds.size());
                const std::vector<unsigned> &gptr = info.group_ptr;
                utils::Assert( gptr.size() != 0 && gptr.back() == preds.size(), "rank loss must have group file" );
                const unsigned ngroup = static_cast<unsigned>( gptr.size() - 1 );

                #pragma omp parallel
                {
                    // parall construct, declare random number generator here, so that each 
                    // thread use its own random number generator, seed by thread id and current iteration
                    random::Random rnd; rnd.Seed( iter * 1111 + omp_get_thread_num() );
                    std::vector< std::pair<float,unsigned> > rec;
                    #pragma for schedule(static)
                    for (unsigned k = 0; k < ngroup; ++k){
                        rec.clear();
                        for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                            rec.push_back( std::make_pair(info.labels[j], j) );
                            grad[j] = hess[j] = 0.0f;
                        }                        
                        std::sort( rec.begin(), rec.end(), CmpFirst );
                        // enumerate buckets with same label, for each item in the list, grab another sample randomly
                        for( unsigned i = 0; i < rec.size(); ){
                            unsigned j = i + 1;
                            while( j < rec.size() && rec[j].first == rec[i].first ) ++ j;
                            // bucket in [i,j), get a sample outside bucket
                            unsigned nleft = i, nright = rec.size() - j;
                            for( unsigned pid = i; pid < j; ++ pid ){
                                unsigned ridx = static_cast<int>( rnd.RandDouble() * (nleft+nright) );
                                if( ridx < nleft ){
                                    // get the samples in left side, ridx is pos sample
                                    this->AddGradient( rec[ridx].second, rec[pid].second, preds, grad, hess );
                                }else{
                                    // get samples in right side, ridx is negsample
                                    this->AddGradient( rec[pid].second, rec[ridx+j-i].second, preds, grad, hess );
                                }
                            }                            
                            i = j;
                        }
                        // rescale each gradient and hessian so that the list have constant weight
                        if( fix_list_weight != 0.0f ){
                            float scale = fix_list_weight / (gptr[k+1] - gptr[k]);
                            for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                                grad[j] *= scale; hess[j] *= scale;
                            }                            
                        }
                    }
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                return "auc";
            }            
        private:
            inline void AddGradient( unsigned pid, unsigned nid, 
                                     const std::vector<float> &pred,
                                     std::vector<float> &grad,
                                     std::vector<float> &hess ){
                float p = loss.PredTransform( pred[pid]-pred[nid] );
                float g = loss.FirstOrderGradient( p, 1.0f );
                float h = loss.SecondOrderGradient( p, 1.0f );
                // accumulate gradient and hessian in both pid, and nid, 
                grad[pid] += g; grad[nid] -= g;
                // take conservative update, scale hessian by 2
                hess[pid] += 2.0f * h; hess[nid] += 2.0f * h;
            }                                     
            inline static bool CmpFirst( const std::pair<float,unsigned> &a, const std::pair<float,unsigned> &b ){
                return a.first > b.first;
            }
        private:
            // fix weight of each list
            float fix_list_weight;
            LossType loss;
        };
    };
};
#endif
