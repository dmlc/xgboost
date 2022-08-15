/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <memory>  // std::make_shared

#include "../../../src/data/iterative_dmatrix.h"
#include "../helpers.h"

namespace xgboost {
namespace data {
template <typename Page, typename Iter, typename Cuts>
void TestRefDMatrix(Cuts&& get_cuts) {
  int n_bins = 256;
  Iter iter(0.3, 2048);
  auto m = std::make_shared<IterativeDMatrix>(&iter, iter.Proxy(), nullptr, Reset, Next,
                                              std::numeric_limits<float>::quiet_NaN(), 0, n_bins);

  Iter iter_1(0.8, 32, Iter::Cols(), 13);
  auto m_1 = std::make_shared<IterativeDMatrix>(&iter_1, iter_1.Proxy(), m, Reset, Next,
                                                std::numeric_limits<float>::quiet_NaN(), 0, n_bins);

  for (auto const& page_0 : m->template GetBatches<Page>({})) {
    for (auto const& page_1 : m_1->template GetBatches<Page>({})) {
      auto const& cuts_0 = get_cuts(page_0);
      auto const& cuts_1 = get_cuts(page_1);
      ASSERT_EQ(cuts_0.Values(), cuts_1.Values());
      ASSERT_EQ(cuts_0.Ptrs(), cuts_1.Ptrs());
      ASSERT_EQ(cuts_0.MinValues(), cuts_1.MinValues());
    }
  }

  m_1 = std::make_shared<IterativeDMatrix>(&iter_1, iter_1.Proxy(), nullptr, Reset, Next,
                                           std::numeric_limits<float>::quiet_NaN(), 0, n_bins);
  for (auto const& page_0 : m->template GetBatches<Page>({})) {
    for (auto const& page_1 : m_1->template GetBatches<Page>({})) {
      auto const& cuts_0 = get_cuts(page_0);
      auto const& cuts_1 = get_cuts(page_1);
      ASSERT_NE(cuts_0.Values(), cuts_1.Values());
      ASSERT_NE(cuts_0.Ptrs(), cuts_1.Ptrs());
    }
  }

  // Use DMatrix as ref
  auto dm = RandomDataGenerator(2048, Iter::Cols(), 0.5).GenerateDMatrix(true);
  auto dqm = std::make_shared<IterativeDMatrix>(&iter_1, iter_1.Proxy(), dm, Reset, Next,
                                                std::numeric_limits<float>::quiet_NaN(), 0, n_bins);
  for (auto const& page_0 : dm->template GetBatches<Page>({})) {
    for (auto const& page_1 : dqm->template GetBatches<Page>({})) {
      auto const& cuts_0 = get_cuts(page_0);
      auto const& cuts_1 = get_cuts(page_1);
      ASSERT_EQ(cuts_0.Values(), cuts_1.Values());
      ASSERT_EQ(cuts_0.Ptrs(), cuts_1.Ptrs());
      ASSERT_EQ(cuts_0.MinValues(), cuts_1.MinValues());
    }
  }
}
}  // namespace data
}  // namespace xgboost
