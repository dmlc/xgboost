#ifndef XGBOOST_COMMON_FEATURE_BUNDLING_H_
#define XGBOOST_COMMON_FEATURE_BUNDLING_H_

#include <vector>
#include <string>

#include "hist_util.h"
#include "column_matrix.h"
#include "random.h"
#include "common.h"

namespace xgboost {

// Exclusive Feature Bundling
class FeatureBundler {
  using BitSet = std::vector<bool>;

  void MarkUsed(BitSet* p_mark, const common::Column& column) {
    // One bit for each element in column
    BitSet& mark = *p_mark;
    switch (column.GetType()) {
      case common::kDenseColumn: {
        for (size_t i = 0; i < column.Size(); ++i) {
          if (column.GetFeatureBinIdx(i) != std::numeric_limits<uint32_t>::max()) {
            mark[i] = true;
          }
        }
      } break;
      case common::kSparseColumn: {
        for (size_t i = 0; i < column.Size(); ++i) {
          mark[column.GetRowIdx(i)] = true;
        }
      } break;
    }
  }

  size_t GetConflictCount(const BitSet& mark,
                          const common::Column& column,
                          size_t max_cnt) {
    size_t ret = 0;
    if (column.GetType() == xgboost::common::kDenseColumn) {
      for (size_t i = 0; i < column.Size(); ++i) {
        if (column.GetFeatureBinIdx(i) != std::numeric_limits<uint32_t>::max() && mark[i]) {
          ++ret;
          if (ret > max_cnt) {
            return max_cnt + 1;
          }
        }
      }
    } else {
      for (size_t i = 0; i < column.Size(); ++i) {
        if (mark[column.GetRowIdx(i)]) {
          ++ret;
          if (ret > max_cnt) {
            return max_cnt + 1;
          }
        }
      }
    }
    return ret;
  }

  std::vector<std::vector<unsigned>> FindGroups(
      const std::vector<unsigned>& feature_list,
      const std::vector<size_t>& feature_nnz,
      const common::ColumnMatrix& colmat,
      size_t nrow,
      const tree::TrainParam& param) {
    /* Goal: Bundle features together that has little or no "overlap", i.e.
       only a few data points should have nonzero values for
       member features.
       Note that one-hot encoded features will be grouped together. */

    std::vector<std::vector<unsigned>> groups;
    std::vector<BitSet> conflict_marks;
    std::vector<size_t> group_nnz;
    std::vector<size_t> group_conflict_cnt;
    const auto max_conflict_cnt
        = static_cast<size_t>(param.max_conflict_rate * nrow);

    for (auto fid : feature_list) {
      const common::Column& column = colmat.GetColumn(fid);

      const size_t cur_fid_nnz = feature_nnz[fid];
      bool need_new_group = true;

      // randomly choose some of existing groups as candidates
      std::vector<size_t> search_groups;
      for (size_t gid = 0; gid < groups.size(); ++gid) {
        if (group_nnz[gid] + cur_fid_nnz <= nrow + max_conflict_cnt) {
          search_groups.push_back(gid);
        }
      }
      std::shuffle(search_groups.begin(), search_groups.end(), common::GlobalRandom());
      if (param.max_search_group > 0 && search_groups.size() > param.max_search_group) {
        search_groups.resize(param.max_search_group);
      }

      // examine each candidate group: is it okay to insert fid?
      for (auto gid : search_groups) {
        const size_t rest_max_cnt = max_conflict_cnt - group_conflict_cnt[gid];
        const size_t cnt = GetConflictCount(conflict_marks[gid], column, rest_max_cnt);
        if (cnt <= rest_max_cnt) {
          need_new_group = false;
          groups[gid].push_back(fid);
          group_conflict_cnt[gid] += cnt;
          group_nnz[gid] += cur_fid_nnz - cnt;
          MarkUsed(&conflict_marks[gid], column);
          break;
        }
      }

      // create new group if necessary
      if (need_new_group) {
        groups.emplace_back();
        groups.back().push_back(fid);
        group_conflict_cnt.push_back(0);
        conflict_marks.emplace_back(nrow, false);
        MarkUsed(&conflict_marks.back(), column);
        group_nnz.emplace_back(cur_fid_nnz);
      }
    }
    return groups;
  }

  std::vector<std::vector<unsigned>>
  FastFeatureGrouping(const common::GHistIndexMatrix& gmat,
                      const common::ColumnMatrix& colmat,
                      const tree::TrainParam& param) {
    const size_t nrow = gmat.row_ptr.size() - 1;
    const size_t nfeature = gmat.cut.row_ptr.size() - 1;

    std::vector<unsigned> feature_list(nfeature);
    std::iota(feature_list.begin(), feature_list.end(), 0);

    // sort features by nonzero counts, descending order
    std::vector<size_t> feature_nnz(nfeature);
    std::vector<unsigned> features_by_nnz(feature_list);
    gmat.GetFeatureCounts(&feature_nnz[0]);
    std::sort(features_by_nnz.begin(), features_by_nnz.end(),
              [&feature_nnz](unsigned a, unsigned b) {
                return feature_nnz[a] > feature_nnz[b];
              });

    auto groups_alt1 = FindGroups(feature_list, feature_nnz, colmat, nrow, param);
    auto groups_alt2 = FindGroups(features_by_nnz, feature_nnz, colmat, nrow, param);

    auto& groups = (groups_alt1.size() > groups_alt2.size()) ? groups_alt2 : groups_alt1;

    // take apart small, sparse groups, as it won't help speed
    {
      std::vector<std::vector<unsigned>> ret;
      for (const auto& group : groups) {
        if (group.size() <= 1 || group.size() >= 5) {
          ret.push_back(group);  // keep singleton groups and large (5+) groups
        } else {
          size_t nnz = 0;
          for (auto fid : group) {
            nnz += feature_nnz[fid];
          }
          double nnz_rate = static_cast<double>(nnz) / nrow;
          // take apart small sparse group, due it will not gain on speed
          if (nnz_rate <= param.sparse_threshold) {
            for (auto fid : group) {
              ret.emplace_back();
              ret.back().emplace_back(fid);
            }
          } else {
            ret.emplace_back(group);
          }
        }
      }
      groups = std::move(ret);
    }

    // shuffle groups
    std::shuffle(groups.begin(), groups.end(), common::GlobalRandom());

    return groups;
  }

 public:
  std::vector<std::vector<unsigned>> GroupFeatures(
      const common::GHistIndexMatrix& gmat,
      const common::ColumnMatrix& colmat,
      const tree::TrainParam& param) {
    return FastFeatureGrouping(gmat, colmat, param);
  }
};

}  // namespace xgboost

#endif  // XGBOOST_COMMON_FEATURE_BUNDLING_H_
