// some backup code

        class LambdaRankObj_NDCG : public LambdaRankObj{

            static inline float CalcDCG(const std::vector< float > &rec) {
                double sumdcg = 0.0;
                for (size_t i = 0; i < rec.size(); i++){
                    const unsigned rel = static_cast<unsigned>(rec[i]);
                    if (rel != 0){
                        sumdcg += logf(2.0f) *((1 << rel) - 1) / logf(i + 2);
                    }
                }
                return static_cast<float>(sumdcg);
            }

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
                return CalcDCG(labels);
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

