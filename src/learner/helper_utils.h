#ifndef XGBOOST_LEARNER_HELPER_UTILS_H_
#define XGBOOST_LEARNER_HELPER_UTILS_H_
/*!
 * \file helper_utils.h
 * \brief useful helper functions
 * \author Tianqi Chen, Kailong Chen
 */
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
namespace xgboost {
namespace learner {
// simple helper function to do softmax
inline static void Softmax(std::vector<float>* p_rec) {
  std::vector<float> &rec = *p_rec;
  float wmax = rec[0];
  for (size_t i = 1; i < rec.size(); ++i) {
    wmax = std::max(rec[i], wmax);
  }
  double wsum = 0.0f;
  for (size_t i = 0; i < rec.size(); ++i) {
    rec[i] = std::exp(rec[i]-wmax);
    wsum += rec[i];
  }
  for (size_t i = 0; i < rec.size(); ++i) {
    rec[i] /= static_cast<float>(wsum);
  }
}
// simple helper function to do softmax
inline static int FindMaxIndex(const std::vector<float>& rec) {
  size_t mxid = 0;
  for (size_t i = 1; i < rec.size(); ++i) {
    if (rec[i] > rec[mxid] + 1e-6f) {
      mxid = i;
    }
  }
  return static_cast<int>(mxid);
}

inline static bool CmpFirst(const std::pair<float, unsigned> &a,
                            const std::pair<float, unsigned> &b) {
  return a.first > b.first;
}
inline static bool CmpSecond(const std::pair<float, unsigned> &a,
                             const std::pair<float, unsigned> &b) {
  return a.second > b.second;
}
}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_HELPER_UTILS_H_
