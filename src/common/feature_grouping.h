/*!
 * Original work Copyright 2017 by Microsoft Corporation
 * Modified work Copyright 2017 by Contributors
 * \file feature_grouping.h
 * \brief Facilities for bundling features together
 * \author Philip Cho
 */

/*
 * NOTICE: this file (feature_grouping.h) is a modified version of code adopted from
 * the LightGBM project (https://github.com/Microsoft/LightGBM). Following the original
 * authors' terms, we reproduce below the entire license notice from LightGBM:
 *
 * The MIT License (MIT)
 *
 * Copyright (c) Microsoft Corporation 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * List of original authors:
 *     Guolin Ke (@guolinke)
 *     Qiwei Ye (@chivee)
 *     xuehui (@xuehui1991)
 *     wxchan (@wxchan)
 *     kimi (@zhangyafeikimi)
 *     Huan Zhang (@huanzhang12)
 *     cbecker (@cbecker)
 *     Allard van Mossel (@Allardvm)
 */

namespace xgboost {
namespace common {

template <typename T>
static unsigned GetConflictCount(const std::vector<bool>& mark,
                                 const Column<T>& column,
                                 unsigned max_cnt) {
  unsigned ret = 0;
  if (column.type == xgboost::common::kDenseColumn) {
    for (size_t i = 0; i < column.len; ++i) {
      if (column.index[i] != std::numeric_limits<T>::max() && mark[i]) {
        ++ret;
        if (ret > max_cnt) {
          return max_cnt + 1;
        }
      }
    }
  } else {
    for (size_t i = 0; i < column.len; ++i) {
      if (mark[column.row_ind[i]]) {
        ++ret;
        if (ret > max_cnt) {
          return max_cnt + 1;
        }
      }
    }
  }
  return ret;
}

template <typename T>
inline void
MarkUsed(std::vector<bool>* p_mark, const Column<T>& column) {
  std::vector<bool>& mark = *p_mark;
  if (column.type == xgboost::common::kDenseColumn) {
    for (size_t i = 0; i < column.len; ++i) {
      if (column.index[i] != std::numeric_limits<T>::max()) {
        mark[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < column.len; ++i) {
      mark[column.row_ind[i]] = true;
    }
  }
}

template <typename T>
inline std::vector<std::vector<unsigned>>
FindGroups_(const std::vector<unsigned>& feature_list,
            const std::vector<bst_uint>& feature_nnz,
            const ColumnMatrix& colmat,
            unsigned nrow,
            const FastHistParam& param) {
  /* Goal: Bundle features together that has little or no "overlap", i.e.
           only a few data points should have nonzero values for
           member features.
           Note that one-hot encoded features will be grouped together. */

  std::vector<std::vector<unsigned>> groups;
  std::vector<std::vector<bool>> conflict_marks;
  std::vector<unsigned> group_nnz;
  std::vector<unsigned> group_conflict_cnt;
  const unsigned max_conflict_cnt
    = static_cast<unsigned>(param.max_conflict_rate * nrow);

  for (auto fid : feature_list) {
    const Column<T>& column = colmat.GetColumn<T>(fid);

    const size_t cur_fid_nnz = feature_nnz[fid];
    bool need_new_group = true;

    // randomly choose some of existing groups as candidates
    std::vector<unsigned> search_groups;
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
      const unsigned rest_max_cnt = max_conflict_cnt - group_conflict_cnt[gid];
      const unsigned cnt = GetConflictCount(conflict_marks[gid], column, rest_max_cnt);
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

inline std::vector<std::vector<unsigned>>
FindGroups(const std::vector<unsigned>& feature_list,
           const std::vector<bst_uint>& feature_nnz,
           const ColumnMatrix& colmat,
           unsigned nrow,
           const FastHistParam& param) {
  XGBOOST_TYPE_SWITCH(colmat.dtype, {
    return FindGroups_<DType>(feature_list, feature_nnz, colmat, nrow, param);
  });
  return std::vector<std::vector<unsigned>>();  // to avoid warning message
}

inline std::vector<std::vector<unsigned>>
FastFeatureGrouping(const GHistIndexMatrix& gmat,
                    const ColumnMatrix& colmat,
                    const FastHistParam& param) {
  const size_t nrow = gmat.row_ptr.size() - 1;
  const size_t nfeature = gmat.cut->row_ptr.size() - 1;

  std::vector<unsigned> feature_list(nfeature);
  std::iota(feature_list.begin(), feature_list.end(), 0);

  // sort features by nonzero counts, descending order
  std::vector<bst_uint> feature_nnz(nfeature);
  std::vector<unsigned> features_by_nnz(feature_list);
  gmat.GetFeatureCounts(&feature_nnz[0]);
  std::sort(features_by_nnz.begin(), features_by_nnz.end(),
            [&feature_nnz](int a, int b) {
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
        unsigned nnz = 0;
        for (auto fid : group) {
          nnz += feature_nnz[fid];
        }
        double nnz_rate = static_cast<double>(nnz) / nrow;
        // take apart small sparse group, due it will not gain on speed
        if (nnz_rate <= param.sparse_threshold) {
          for (auto fid : group) {
            ret.emplace_back();
            ret.back().push_back(fid);
          }
        } else {
          ret.push_back(group);
        }
      }
    }
    groups = std::move(ret);
  }

  // shuffle groups
  std::shuffle(groups.begin(), groups.end(), common::GlobalRandom());

  return groups;
}

}  // namespace common
}  // namespace xgboost
