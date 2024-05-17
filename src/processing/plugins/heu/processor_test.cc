/**
 * Copyright 2014-2024 by XGBoost Contributors
 */

#include "processor.h"

#include "gtest/gtest.h"

namespace processing::test {

class ProcessorTest : public testing::Test {
 public:
  void SetUp() override {
    std::map<std::string, std::string> params = {
        {kLibraryPath, "processing/plugins"}};
    auto loader = processing::ProcessorLoader(params);

    active_processor_ = loader.load("heu");
    active_processor_->Initialize(true,
                                  {{"PUBLIC_KEY_PATH", "processing/plugins"},
                                   {"SECRET_KEY_PATH", "processing/plugins"},
                                   {"scale", "1000000"}});

    passive_processor_ = loader.load("heu");
    passive_processor_->Initialize(false,
                                   {{"PUBLIC_KEY_PATH", "processing/plugins"}});
  }

  void TearDown() override {
    active_processor_->Shutdown();
    active_processor_ = nullptr;  // TODO: free?

    passive_processor_->Shutdown();
    passive_processor_ = nullptr;  // TODO: free?
  }

 protected:
  processing::Processor *active_processor_ = nullptr;
  processing::Processor *passive_processor_ = nullptr;

  // clang-format off
  // Test data, 4 Rows, 2 Features
  std::vector<double> gh_pairs_ = {
          1.1, 2.1,
          3.1, 4.1,
          5.1, 6.1,
          7.1, 8.1
  };  // 4 Rows, 8 GH Pairs
  std::vector<uint32_t> cuts_ = {0, 4, 10};  // 2 features, one has 4 bins, another 6
  std::vector<int> slots_ = {
          0, 4,
          1, 9,
          3, 7,
          0, 4
  };

  std::vector<int> node0_ = {0, 2};
  std::vector<int> node1_ = {1, 3};

  std::map<int, std::vector<int>> nodes_ = {{0, node0_},
                                            {1, node1_}};
  // clang-format on
};

TEST_F(ProcessorTest, TestAggregation) {
  size_t buf_size;
  auto *buffer = active_processor_->ProcessGHPairs(&buf_size, gh_pairs_);
  passive_processor_->HandleGHPairs(&buf_size, buffer, buf_size);
  active_processor_->FreeBuffer(buffer);

  passive_processor_->InitAggregationContext(cuts_, slots_);
  buffer = passive_processor_->ProcessAggregation(&buf_size, nodes_);
  auto histograms = active_processor_->HandleAggregation(buffer, buf_size);
  passive_processor_->FreeBuffer(buffer);

  std::vector<double> expected_result = {
      1.1, 2.1, 0, 0, 0, 0, 5.1, 6.1, 1.1, 2.1, 0,   0,  0, 0,
      5.1, 6.1, 0, 0, 0, 0, 7.1, 8.1, 3.1, 4.1, 0,   0,  0, 0,
      7.1, 8.1, 0, 0, 0, 0, 0,   0,   0,   0,   3.1, 4.1};

  ASSERT_EQ(histograms.size(), expected_result.size())
      << "Histograms have different sizes";

  for (size_t i = 0; i < histograms.size(); ++i) {
    ASSERT_NEAR(histograms[i], expected_result[i], 1e-6);
  }
}

}  // namespace processing::test
