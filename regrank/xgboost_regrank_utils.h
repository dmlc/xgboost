#ifndef XGBOOST_REGRANK_UTILS_H
#define XGBOOST_REGRANK_UTILS_H
/*!
 * \file xgboost_regrank_utils.h
 * \brief useful helper functions
 * \author Tianqi Chen, Kailong Chen
 */
namespace xgboost{
    namespace regrank{
        // simple helper function to do softmax
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
                rec[i] /= static_cast<float>(wsum);
            }                
        }        
        // simple helper function to do softmax
        inline static int FindMaxIndex( std::vector<float>& rec ){
            size_t mxid = 0;
            for( size_t i = 1; i < rec.size(); ++ i ){
                if( rec[i] > rec[mxid]+1e-6f ){
                    mxid = i;
                }
            }
            return (int)mxid;
        }        
        inline static bool CmpFirst(const std::pair<float, unsigned> &a, const std::pair<float, unsigned> &b){
            return a.first > b.first;
        }
        inline static bool CmpSecond(const std::pair<float, unsigned> &a, const std::pair<float, unsigned> &b){
            return a.second > b.second;
        }
    };
};

#endif

