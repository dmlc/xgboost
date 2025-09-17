/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>  // for int8_t, int32_t
#include <vector>   // for vector

namespace enc {
template <typename Encoder, typename DfTest>
void TestOrdinalEncoderStrs() {
  Encoder encoder;
  auto sol = std::vector<std::int32_t>{0, 3, 1};

  {
    auto df = DfTest::Make(DfTest::MakeStrs("c", "b", "d", "a"));
    auto orig_dict = df.View();
    ASSERT_EQ(orig_dict.Size(), 1);

    auto new_df = DfTest::Make(DfTest::MakeStrs("c", "a", "b"));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    ASSERT_EQ(new_df.Mapping().size(), 3);

    ASSERT_EQ(new_df.Mapping(), sol);
  }
  {
    // longer strings
    auto df = DfTest::Make(DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeStrs("cbd", "ab", "bbd"));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    ASSERT_EQ(new_df.Mapping().size(), 3);
    ASSERT_EQ(new_df.Mapping(), sol);
  }
  {
    // Test error message.
    auto df = DfTest::Make(DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeStrs("oops", "ab", "bbd"));
    auto new_dict = new_df.View();
    ASSERT_THAT([&] { encoder.Recode(orig_dict, new_dict, new_df.MappingView()); },
                ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr("`oops`")));
  }
  {
    // Multi-columns
    auto df = DfTest::Make(DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"),
                           DfTest::MakeStrs("b", "c", "a", "d"));
    auto orig_dict = df.View();

    auto new_df =
        DfTest::Make(DfTest::MakeStrs("cbd", "ab", "bbd"), DfTest::MakeStrs("d", "a", "b"));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    auto segs = new_df.Segment();
    auto beg = segs[0];
    auto end = segs[1];

    auto sol0 = sol;
    for (auto i = beg, k = 0; i < end; ++i, ++k) {
      ASSERT_EQ(sol0[k], new_df.Mapping()[i]);
    }

    beg = end;
    end = segs[2];
    auto sol1 = std::vector{3, 2, 0};
    for (auto i = beg, k = 0; i < end; ++i, ++k) {
      ASSERT_EQ(sol1[k], new_df.Mapping()[i]);
    }
  }
}

template <typename Encoder, typename DfTest>
void TestOrdinalEncoderInts() {
  Encoder encoder;
  auto sol = std::vector<std::int32_t>{0, 3, 1};

  {
    auto df = DfTest::Make(DfTest::MakeInts(2, 1, 3, 0));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeInts(2, 0, 1));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    ASSERT_EQ(new_df.Mapping(), sol);
  }
  {
    // Test error message.
    auto df = DfTest::Make(DfTest::MakeInts(2, 1, 3, 0));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeInts(2, 0, 5));
    auto new_dict = new_df.View();
    ASSERT_THAT([&] { encoder.Recode(orig_dict, new_dict, new_df.MappingView()); },
                ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr("`5`")));
  }
  {
    auto df = DfTest::Make(DfTest::MakeInts(0), DfTest::MakeInts(0, 1));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeInts(0), DfTest::MakeInts(0, 1));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    auto mapping = new_df.Mapping();
    std::vector<std::int32_t> sol{0, 0, 1};
    ASSERT_EQ(mapping, sol);
  }
}

template <typename Encoder, typename DfTest>
void TestOrdinalEncoderMixed() {
  Encoder encoder;
  auto sol = std::vector<std::int32_t>{0, 3, 1};

  {
    auto df =
        DfTest::Make(DfTest::MakeInts(2, 1, 3, 0), DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeInts(2, 0, 1), DfTest::MakeStrs("cbd", "ab", "bbd"));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    ASSERT_EQ(new_df.Mapping().size(), 6);
    for (std::size_t i = 0; i < new_df.Mapping().size(); ++i) {
      ASSERT_EQ(new_df.Mapping()[i], sol[i % sol.size()]);
    }
  }
  {
    auto df =
        DfTest::Make(DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"), DfTest::MakeInts(2, 1, 3, 0));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeStrs("cbd", "ab", "bbd"), DfTest::MakeInts(2, 0, 1));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    ASSERT_EQ(new_df.Mapping().size(), 6);
    for (std::size_t i = 0; i < new_df.Mapping().size(); ++i) {
      ASSERT_EQ(new_df.Mapping()[i], sol[i % sol.size()]);
    }
  }
  {
    auto df =
        DfTest::Make(DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"), DfTest::MakeInts(2, 1, 3, 0),
                     DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"));
    auto orig_dict = df.View();

    auto new_df = DfTest::Make(DfTest::MakeStrs("cbd", "ab", "bbd"), DfTest::MakeInts(2, 0),
                               DfTest::MakeStrs("cbd", "ab", "bbd"));
    auto new_dict = new_df.View();

    encoder.Recode(orig_dict, new_dict, new_df.MappingView());
    ASSERT_EQ(new_df.Mapping().size(), 8);
    for (std::size_t i = 0; i < 3; ++i) {
      ASSERT_EQ(new_df.Mapping()[i], sol[i]);
    }
    for (std::size_t i = 3, k = 0; i < 5; ++i, ++k) {
      ASSERT_EQ(new_df.Mapping()[i], sol[k]);
    }
    for (std::size_t i = 5, k = 0; i < 8; ++i, ++k) {
      ASSERT_EQ(new_df.Mapping()[i], sol[k]);
    }
  }
}

template <typename Encoder, typename DfTest>
void TestOrdinalEncoderEmpty() {
  auto sol = std::vector<std::int32_t>{0, 3, 1};
  Encoder encoder;
  auto df = DfTest::Make(DfTest::MakeInts(), DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"),
                         DfTest::MakeInts());
  auto orig_dict = df.View();

  auto new_df =
      DfTest::Make(DfTest::MakeInts(), DfTest::MakeStrs("cbd", "ab", "bbd"), DfTest::MakeInts());
  auto new_dict = new_df.View();
  encoder.Recode(orig_dict, new_dict, new_df.MappingView());
  ASSERT_EQ(new_df.Mapping().size(), 3);
  ASSERT_EQ(new_df.Mapping(), sol);
}
}  // namespace enc
