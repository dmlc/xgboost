#ifndef XGBOOST_REGRANK_OBJ_HPP
#define XGBOOST_REGRANK_OBJ_HPP
/*!
 * \file xgboost_regrank_obj.hpp
 * \brief implementation of objective functions
 * \author Tianqi Chen, Kailong Chen
 */
//#include "xgboost_regrank_sample.h"
#include <vector>
#include <functional>
#include "xgboost_regrank_utils.h"

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
                if( !strcmp( "scale_pos_weight", name ) ) scale_pos_weight = (float)atof( val );
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                grad.resize(preds.size()); hess.resize(preds.size());

                const unsigned ndata = static_cast<unsigned>(preds.size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    float p = loss.PredTransform(preds[j]);
                    float w = info.GetWeight(j);
                    if( info.labels[j] == 1.0f ) w *= scale_pos_weight;
                    grad[j] = loss.FirstOrderGradient(p, info.labels[j]) * w;
                    hess[j] = loss.SecondOrderGradient(p, info.labels[j]) * w;
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                if( loss.loss_type == LossType::kLogisticClassify ) return "error";
                if( loss.loss_type == LossType::kLogisticRaw ) return "auc";
                return "rmse";
            }
            virtual void PredTransform(std::vector<float> &preds){
                const unsigned ndata = static_cast<unsigned>(preds.size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = loss.PredTransform( preds[j] );
                }
            }
        private:
            float scale_pos_weight;
            LossType loss;
        };
    };

    namespace regrank{
        // simple softmax rak
        class SoftmaxRankObj : public IObjFunction{
        public:
            SoftmaxRankObj(void){
            }
            virtual ~SoftmaxRankObj(){}
            virtual void SetParam(const char *name, const char *val){
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                grad.resize(preds.size()); hess.resize(preds.size());
                const std::vector<unsigned> &gptr = info.group_ptr;
                utils::Assert( gptr.size() != 0 && gptr.back() == preds.size(), "rank loss must have group file" );
                const unsigned ngroup = static_cast<unsigned>( gptr.size() - 1 );

                #pragma omp parallel
                {
                    std::vector< float > rec;                    
                    #pragma omp for schedule(static)
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
        };

        // simple softmax multi-class classification
        class SoftmaxMultiClassObj : public IObjFunction{
        public:
            SoftmaxMultiClassObj(void){
                nclass = 0;
            }
            virtual ~SoftmaxMultiClassObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "num_class", name ) ) nclass = atoi(val); 
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( nclass != 0, "must set num_class to use softmax" );
                utils::Assert( preds.size() == (size_t)nclass * info.labels.size(), "SoftmaxMultiClassObj: label size and pred size does not match" );
                grad.resize(preds.size()); hess.resize(preds.size());
                
                const unsigned ndata = static_cast<unsigned>(info.labels.size());
                #pragma omp parallel
                {
                    std::vector<float> rec(nclass);
                    #pragma omp for schedule(static)
                    for (unsigned j = 0; j < ndata; ++j){
                        for( int k = 0; k < nclass; ++ k ){
                            rec[k] = preds[j + k * ndata];
                        }
                        Softmax( rec );
                        int label = static_cast<int>(info.labels[j]);
                        if( label < 0 ){
                            label = -label - 1;
                        }
                        utils::Assert( label < nclass, "SoftmaxMultiClassObj: label exceed num_class" );
                        for( int k = 0; k < nclass; ++ k ){
                            float p = rec[ k ];
                            if( label == k ){
                                grad[j+k*ndata] = p - 1.0f;
                            }else{
                                grad[j+k*ndata] = p;
                            }
                            hess[j+k*ndata] = 2.0f * p * ( 1.0f - p );
                        }  
                    }
                }
            }
            virtual void PredTransform(std::vector<float> &preds){
                utils::Assert( nclass != 0, "must set num_class to use softmax" );
                utils::Assert( preds.size() % nclass == 0, "SoftmaxMultiClassObj: label size and pred size does not match" );                
                const unsigned ndata = static_cast<unsigned>(preds.size()/nclass);
                
                #pragma omp parallel
                {
                    std::vector<float> rec(nclass);
                    #pragma omp for schedule(static)
                    for (unsigned j = 0; j < ndata; ++j){
                        for( int k = 0; k < nclass; ++ k ){
                            rec[k] = preds[j + k * ndata];
                        }
                        preds[j] = FindMaxIndex( rec );
                    }
                }
                preds.resize( ndata );
            }
            virtual const char* DefaultEvalMetric(void) {
                return "merror";
            }
        private:
            int nclass;
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
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );              
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
                    #pragma omp for schedule(static)
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
                return "ndcg";
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
        class LambdaRankObj : public IObjFunction{
        public:
            LambdaRankObj(void){}

            virtual ~LambdaRankObj(){}

            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "loss_type", name ) ) loss_.loss_type = atoi( val );
                if( !strcmp( "fix_list_weight", name ) ) fix_list_weight_ = (float)atof( val );

	    }
        private:
            LossType loss_;
            float fix_list_weight_;
        protected:

            class Triple{
            public:
                float pred_;
                float label_;
                int index_;

                Triple(){

                }

                Triple(const Triple& t){
                    pred_ = t.pred_;
                    label_ = t.label_;
                    index_ = t.index_;
                }

                Triple(float pred, float label, int index) :pred_(pred), label_(label), index_(index){

                }
            };

            static inline bool TripleComparer(const Triple &a, const Triple &b){
                return a.pred_ > b.pred_;
            }

            /* \brief Sorted tuples of a group by the predictions, and
            *         the fields in the return tuples successively are predicions,
            *         labels, and the original index of the instance in the group
            */
            inline void GetSortedTuple(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<unsigned> &group_index,
                int group, std::vector< Triple > &sorted_triple){
                sorted_triple.resize(group_index[group + 1] - group_index[group]);
                for (unsigned j = group_index[group]; j < group_index[group + 1]; j++){
                    sorted_triple[j - group_index[group]] = Triple(preds[j], labels[j], j);
                }
                
                std::sort(sorted_triple.begin(), sorted_triple.end(), TripleComparer);
            }

            /*
            * \brief Get the position of instances after sorted
            * \param sorted_triple  the fields successively are predicions,
            *         labels, and the original index of the instance in the group
            * \param start  the offset index of the group
            * \param index_remap a vector indicating the new position of each instance after sorted, 
            *         for example,[1,0] means that the second instance is put ahead after sorted
            */
            inline void GetIndexMap(std::vector< Triple > sorted_triple, int start, std::vector<int> &index_remap){
                index_remap.resize(sorted_triple.size());
                for (size_t i = 0; i < sorted_triple.size(); i++){
                    index_remap[sorted_triple[i].index_ - start] = i;
                }
            }

            
            virtual void GetLambda(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<unsigned> &group_index,
                const std::vector< std::pair<int, int> > &pairs, std::vector<float> &lambda, int group) = 0;

            inline void GetGroupGradient(const std::vector<float> &preds,
                const std::vector<float> &labels,
                const std::vector<unsigned> &group_index,
                std::vector<float> &grad,
                std::vector<float> &hess,
                const std::vector< std::pair<int, int> > pairs,
                int group){

                std::vector<float> lambda;
                GetLambda(preds, labels, group_index, pairs, lambda, group);

                float pred_diff, delta;
                float first_order_gradient, second_order_gradient;
                
                for (size_t i = 0; i < pairs.size(); i++){
                    delta = lambda[i];
                    pred_diff = loss_.PredTransform(preds[pairs[i].first] - preds[pairs[i].second]);
                    first_order_gradient = delta * loss_.FirstOrderGradient(pred_diff, 1.0f);
                    second_order_gradient = 2 * delta *  loss_.SecondOrderGradient(pred_diff, 1.0f);
                    hess[pairs[i].first] += second_order_gradient;
                    grad[pairs[i].first] += first_order_gradient;
                    hess[pairs[i].second] += second_order_gradient;
                    grad[pairs[i].second] -= first_order_gradient;
                    	
		}

                if( fix_list_weight_ != 0.0f ){
                    float scale = fix_list_weight_ / (group_index[group+1] - group_index[group]);
                    for(unsigned j = group_index[group]; j < group_index[group+1]; ++j ){
                        grad[j] *= scale; 
			hess[j] *= scale;
                    }                            
                }
            }

           virtual void GenPairs(const std::vector<float>& preds,
                const std::vector<float>& labels,
                const int &start, const int &end,
		std::vector< std::pair<int,int> > &pairs){
	            
	        random::Random rnd; rnd.Seed(0);
		std::vector< std::pair<float,unsigned> > rec;
                for(int j = start; j < end; ++j ){
                    rec.push_back( std::make_pair(labels[j], j) );
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
			    pairs.push_back(std::make_pair(rec[ridx].second, rec[pid].second));
                        }else{
                            // get samples in right side, ridx is negsample
			    pairs.push_back(std::make_pair(rec[pid].second, rec[ridx+j-i].second));
                        }
                    }                            
                    i = j;
                }
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

                for (size_t i = 0; i < group_index.size() - 1; i++){
                    std::vector< std::pair<int,int> > pairs;
		    GenPairs(preds, info.labels, group_index[i], group_index[i + 1],pairs);
                    GetGroupGradient(preds, info.labels, group_index, grad, hess, pairs, i);
                }
            }

            virtual const char* DefaultEvalMetric(void) {
                return "auc";
            }
        };

        class LambdaRankObj_NDCG : public LambdaRankObj{

            /*
            * \brief Obtain the delta NDCG if trying to switch the positions of instances in index1 or index2
            *        in sorted triples. Here DCG is calculated as sigma_i 2^rel_i/log(i + 1)
            * \param sorted_triple the fields are predition,label,original index
            * \param index1,index2 the instances switched
            * \param the IDCG of the list
            */
            inline float GetLambdaNDCG(const std::vector< Triple > sorted_triple,
            int index1,
            int index2, float IDCG){
                double original = (1 << static_cast<int>(sorted_triple[index1].label_)) / log(index1 + 2)
                    + (1 << static_cast<int>(sorted_triple[index2].label_)) / log(index2 + 2);
                double changed = (1 << static_cast<int>(sorted_triple[index2].label_)) / log(index1 + 2)
                    + (1 << static_cast<int>(sorted_triple[index1].label_)) / log(index2 + 2);
                double ans = (original - changed) / IDCG;
                if (ans < 0) ans = -ans;
                return static_cast<float>(ans);
            }


            inline float GetIDCG(const std::vector< Triple > sorted_triple){
                std::vector<float> labels;
                for (size_t i = 0; i < sorted_triple.size(); i++){
                    labels.push_back(sorted_triple[i].label_);
                }

                std::sort(labels.begin(), labels.end(), std::greater<float>());
                return EvalNDCG::CalcDCG(labels);
            }

            inline void GetLambda(const std::vector<float> &preds,
            const std::vector<float> &labels,
            const std::vector<unsigned> &group_index,
            const std::vector< std::pair<int, int> > &pairs, std::vector<float> &lambda, int group){
                std::vector< Triple > sorted_triple;
                std::vector<int> index_remap;
                float IDCG;
                
                GetSortedTuple(preds, labels, group_index, group, sorted_triple);
                GetIndexMap(sorted_triple, group_index[group], index_remap);
                IDCG = GetIDCG(sorted_triple);

                lambda.resize(pairs.size());
                for (size_t i = 0; i < pairs.size(); i++){
                    lambda[i] = GetLambdaNDCG(sorted_triple, 
                        index_remap[pairs[i].first],index_remap[pairs[i].second],IDCG);
                }
            }
        };

        class LambdaRankObj_MAP : public LambdaRankObj{
            class Quadruple{
            public:
                /* \brief the accumulated precision */
                float ap_acc_;
                /* \brief the accumulated precision assuming a positive instance is missing*/
                float ap_acc_miss_;
                /* \brief the accumulated precision assuming that one more positive instance is inserted ahead*/
                float ap_acc_add_;
                /* \brief the accumulated positive instance count */
                float hits_;

                Quadruple(){}

                Quadruple(const Quadruple& q){
                    ap_acc_ = q.ap_acc_;
                    ap_acc_miss_ = q.ap_acc_miss_;
                    ap_acc_add_ = q.ap_acc_add_;
                    hits_ = q.hits_;
                }

                Quadruple(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits
                    ) :ap_acc_(ap_acc), ap_acc_miss_(ap_acc_miss), ap_acc_add_(ap_acc_add), hits_(hits){

                }

            };

            /*
            * \brief Obtain the delta MAP if trying to switch the positions of instances in index1 or index2
            *        in sorted triples
            * \param sorted_triple the fields are predition,label,original index
            * \param index1,index2 the instances switched
            * \param map_acc a vector containing the accumulated precisions for each position in a list
            */
            inline float GetLambdaMAP(const std::vector< Triple > sorted_triple,
            int index1, int index2,
            std::vector< Quadruple > &map_acc){
                if (index1 == index2 || sorted_triple[index1].label_ == sorted_triple[index2].label_) return 0.0;
                if (index1 > index2) std::swap(index1, index2);
                float original = map_acc[index2].ap_acc_; // The accumulated precision in the interval [index1,index2]
                if (index1 != 0) original -= map_acc[index1 - 1].ap_acc_;
                float changed = 0;
                if (sorted_triple[index1].label_ < sorted_triple[index2].label_){
                    changed += map_acc[index2 - 1].ap_acc_add_ - map_acc[index1].ap_acc_add_;
                    changed += (map_acc[index1].hits_ + 1.0f) / (index1 + 1);
                }
                else{
                    changed += map_acc[index2 - 1].ap_acc_miss_ - map_acc[index1].ap_acc_miss_;
                    changed += map_acc[index2].hits_ / (index2 + 1);
                }
                float ans = (changed - original) / (map_acc[map_acc.size() - 1].hits_);
                if (ans < 0) ans = -ans;
                return ans;
            }


            /*
            * \brief preprocessing results for calculating delta MAP
            * \return The first field is the accumulated precision, the second field is the
            *         accumulated precision assuming a positive instance is missing,
            *         the third field is the accumulated precision assuming that one more positive
            *         instance is inserted, the fourth field is the accumulated positive instance count
            */
            inline void GetMAPAcc(const std::vector< Triple > sorted_triple, 
                std::vector< Quadruple > &map_acc){
                map_acc.resize(sorted_triple.size());
                float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                for (size_t i = 1; i <= sorted_triple.size(); i++){
                    if ((int)sorted_triple[i - 1].label_ == 1) {
                        hit++;
                        acc1 += hit / i;
                        acc2 += (hit - 1) / i;
                        acc3 += (hit + 1) / i;
                    }
                    map_acc[i-1] = Quadruple(acc1, acc2, acc3, hit);
                }
            }

            inline void GetLambda(const std::vector<float> &preds,
            const std::vector<float> &labels,
            const std::vector<unsigned> &group_index,
            const std::vector< std::pair<int, int> > &pairs, std::vector<float> &lambda, int group){
                std::vector< Triple > sorted_triple;
                std::vector<int> index_remap;
                std::vector< Quadruple > map_acc;
           
                GetSortedTuple(preds, labels, group_index, group, sorted_triple);
                GetIndexMap(sorted_triple, group_index[group], index_remap);
                GetMAPAcc(sorted_triple, map_acc);

                lambda.resize(pairs.size());
                for (size_t i = 0; i < pairs.size(); i++){
                    lambda[i] = GetLambdaMAP(sorted_triple,
                        index_remap[pairs[i].first], index_remap[pairs[i].second], map_acc);
                }
            }
        };

        
    };
};
#endif
