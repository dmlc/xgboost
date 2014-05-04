#ifndef XGBOOST_REGRANK_OBJ_HPP
#define XGBOOST_REGRANK_OBJ_HPP
/*!
 * \file xgboost_regrank_obj.h
 * \brief implementation of objective functions
 * \author Tianqi Chen, Kailong Chen
 */
//#include "xgboost_regrank_sample.h"
#include <vector>
#include <tuple>
#include <functional>
#include "xgboost_regrank_sample.h"
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


    namespace regrank{
        // simple pairwise rank 
        class LambdaRankObj : public IObjFunction{
        public:
            LambdaRankObj(void){}

            virtual ~LambdaRankObj(){}

            virtual void SetParam(const char *name, const char *val){
                if (!strcmp("loss_type", name)) loss_.loss_type = atoi(val);
                if (!strcmp("sampler", name)) sampler_.AssignSampler(atoi(val));
                if (!strcmp("lambda", name)) lambda_ = atoi(val);
            }
        private:
            int lambda_;
            const static int PAIRWISE = 0;
            const static int MAP = 1;
            const static int NDCG = 2;
            sample::PairSamplerWrapper sampler_;
            LossType loss_;
           

            /* \brief Sorted tuples of a group by the predictions, and
            *         the fields in the return tuples successively are predicions,
            *         labels, and the original index of the instance in the group
            */
            inline std::vector< sample::Triple<float, float, int> > GetSortedTuple(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<unsigned> &group_index,
                int group){
                std::vector< sample::Triple<float, float, int> > sorted_triple;
                for (int j = group_index[group]; j < group_index[group + 1]; j++){
                    sorted_triple.push_back(sample::Triple<float, float, int>(preds[j], labels[j], j));
                }
                std::sort(sorted_triple.begin(), sorted_triple.end(), sample::Triplef1Comparer);
                return sorted_triple;
            }

            /*
            * \brief Get the position of instances after sorted
            * \param sorted_triple  the fields successively are predicions,
            *         labels, and the original index of the instance in the group
            * \param start  the offset index of the group
            * \return a vector indicating the new position of each instance after sorted, 
            *         for example,[1,0] means that the second instance is put ahead after sorted
            */
            inline std::vector<int> GetIndexMap(std::vector< sample::Triple<float, float, int> > sorted_triple, int start){
                std::vector<int> index_remap;
                index_remap.resize(sorted_triple.size());
                for (int i = 0; i < sorted_triple.size(); i++){
                    index_remap[sorted_triple[i].f3_ - start] = i;
                }
                return index_remap;
            }

            /*
            * \brief Obtain the delta MAP if trying to switch the positions of instances in index1 or index2
            *        in sorted triples
            * \param sorted_triple the fields are predition,label,original index
            * \param index1,index2 the instances switched 
            * \param map_acc The first field is the accumulated precision, the second field is the 
            *         accumulated precision assuming a positive instance is missing,
            *         the third field is the accumulated precision assuming that one more positive 
            *         instance is inserted, the fourth field is the accumulated positive instance count 
            */
            inline float GetLambdaMAP(const std::vector< sample::Triple<float, float, int> > sorted_triple,
                int index1, int index2,
                std::vector< sample::Quadruple<float, float, float, float> > map_acc){
                if (index1 == index2 || sorted_triple[index1].f2_ == sorted_triple[index2].f2_) return 0.0;
                if (index1 > index2) std::swap(index1, index2);
                float original = map_acc[index2].f1_; // The accumulated precision in the interval [index1,index2]
                if (index1 != 0) original -= map_acc[index1 - 1].f1_;
                float changed = 0;
                if (sorted_triple[index1].f2_ < sorted_triple[index2].f2_){
                    changed += map_acc[index2 - 1].f3_ - map_acc[index1].f3_;
                    changed += (map_acc[index1].f4_ + 1.0f) / (index1 + 1);
                }
                else{
                    changed += map_acc[index2 - 1].f2_ - map_acc[index1].f2_;
                    changed += map_acc[index2].f4_ / (index2 + 1);
                }
                float ans = (changed - original) / (map_acc[map_acc.size() - 1].f4_);
                if (ans < 0) ans = -ans;
                return ans;
            }

            /*
            * \brief Obtain the delta NDCG if trying to switch the positions of instances in index1 or index2
            *        in sorted triples. Here DCG is calculated as sigma_i 2^rel_i/log(i + 1)
            * \param sorted_triple the fields are predition,label,original index
            * \param index1,index2 the instances switched
            * \param the IDCG of the list
            */
            inline float GetLambdaNDCG(const std::vector< sample::Triple<float, float, int> > sorted_triple,
                int index1,
                int index2, float IDCG){
                float original = (1 << (int)sorted_triple[index1].f2_) / log(index1 + 2)
                    + (1 << (int)sorted_triple[index2].f2_) / log(index2 + 2);
                float changed = (1 << (int)sorted_triple[index2].f2_) / log(index1 + 2)
                    + (1 << (int)sorted_triple[index1].f2_) / log(index2 + 2);
                float ans = (original - changed) / IDCG;
                if (ans < 0) ans = -ans;
                return ans;
            }


            inline float GetIDCG(const std::vector< sample::Triple<float, float, int> > sorted_triple){
                std::vector<float> labels;
                for (int i = 0; i < sorted_triple.size(); i++){
                    labels.push_back(sorted_triple[i].f2_);
                }

                std::sort(labels.begin(), labels.end(), std::greater<float>());
                return EvalNDCG::CalcDCG(labels);
            }

            /*
            * \brief preprocessing results for calculating delta MAP
            * \return The first field is the accumulated precision, the second field is the 
            *         accumulated precision assuming a positive instance is missing,
            *         the third field is the accumulated precision assuming that one more positive 
            *         instance is inserted, the fourth field is the accumulated positive instance count
            */
            inline std::vector< sample::Quadruple<float, float, float, float> > GetMAPAcc(const std::vector< sample::Triple<float, float, int> > sorted_triple){
                std::vector< sample::Quadruple<float, float, float, float> > map_acc;
                float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                for (int i = 1; i <= sorted_triple.size(); i++){
                    if (sorted_triple[i-1].f2_ == 1) {
                        hit++;
                        acc1 += hit / i;
                        acc2 += (hit - 1) / i;
                        acc3 += (hit + 1) / i;
                    }
                    map_acc.push_back(sample::Quadruple<float, float, float, float>(acc1, acc2, acc3, hit));
                }
                return map_acc;

            }

            inline float GetLambdaDelta(std::vector< sample::Triple<float, float, int> > sorted_triple,
            int ins1,int ins2,
            std::vector< sample::Quadruple<float, float, float, float> > map_acc,
            float IDCG){
                float delta = 0.0;
                switch (lambda_){
                case PAIRWISE: delta = 1.0; break;
                case MAP: delta = GetLambdaMAP(sorted_triple, ins1, ins2, map_acc); break;
                case NDCG: delta = GetLambdaNDCG(sorted_triple, ins1, ins2, IDCG); break;
                default: utils::Error("Cannot find the specified loss type");
                }
                return delta;
            }

            inline void GetGroupGradient(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<unsigned> &group_index,
                std::vector<float> &grad,
                std::vector<float> &hess,
                const sample::Pairs& pairs,
                int group){
                bool j_better;
                float pred_diff, pred_diff_exp, delta;
                float first_order_gradient, second_order_gradient;
                std::vector< sample::Triple<float, float, int> > sorted_triple;
                std::vector<int> index_remap;
                std::vector< sample::Quadruple<float, float, float, float> > map_acc;
                float IDCG;

                // preparing data for lambda NDCG
                if (lambda_ == NDCG){
                    sorted_triple = GetSortedTuple(preds, labels, group_index, group);
                    IDCG = GetIDCG(sorted_triple);
                    index_remap = GetIndexMap(sorted_triple, group_index[group]);
                }

                // preparing data for lambda MAP
                else if (lambda_ == MAP){
                     sorted_triple = GetSortedTuple(preds, labels, group_index, group);
                     map_acc = GetMAPAcc(sorted_triple);
                     index_remap = GetIndexMap(sorted_triple, group_index[group]);
                }

                for (int j = group_index[group]; j < group_index[group + 1]; j++){
                    std::vector<int> pair_instance = pairs.GetPairs(j);
                    for (int k = 0; k < pair_instance.size(); k++){
                        j_better = labels[j] > labels[pair_instance[k]];
                        if (j_better){
                            delta = GetLambdaDelta(sorted_triple, index_remap[j - group_index[group]],
                                index_remap[pair_instance[k] - group_index[group]],map_acc,IDCG);
                            pred_diff = preds[j] - preds[pair_instance[k]];
                            pred_diff_exp = j_better ? expf(-pred_diff) : expf(pred_diff);
                            first_order_gradient = delta * FirstOrderGradient(pred_diff_exp);
                            second_order_gradient = 2 * delta * SecondOrderGradient(pred_diff_exp);
                            hess[j] += second_order_gradient;
                            grad[j] += first_order_gradient;
                            hess[pair_instance[k]] += second_order_gradient;
                            grad[pair_instance[k]] += -first_order_gradient;
                        }
                    }
                }
            }

            inline float FirstOrderGradient(float pred_diff_exp) const {
                return -pred_diff_exp / (1 + pred_diff_exp);
            }

            inline float SecondOrderGradient(float pred_diff_exp) const {
                return pred_diff_exp / pow(1 + pred_diff_exp, 2);
            }

        public:
            virtual void GetGradient(const std::vector<float>& preds,
                const DMatrix::Info &info,
                int iter,
                std::vector<float> &grad,
                std::vector<float> &hess) {
                grad.resize(preds.size()); hess.resize(preds.size());
                const std::vector<unsigned> &group_index = info.group_ptr;
                utils::Assert(group_index.size() != 0 && group_index.back() == preds.size(), "rank loss must have group file");

                for (int i = 0; i < group_index.size() - 1; i++){
                    sample::Pairs pairs = sampler_.GenPairs(preds, info.labels, group_index[i], group_index[i + 1]);
                    //pairs.GetPairs()
                    GetGroupGradient(preds, info.labels, group_index, grad, hess, pairs, i);
                }
            }

            virtual const char* DefaultEvalMetric(void) {
                return "auc";
            }
        };
    };
};
#endif
