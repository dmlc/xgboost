/*!
 * Copyright 2024 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/processing/processor.h"

const double kError = 1E-10;

class ProcessorTest : public testing::Test {
 public:
    void SetUp() override {
        auto loader = processing::ProcessorLoader();
        processor_ = loader.load(processing::kMockProcessor);
        processor_->Initialize(true, {});
        loader.unload();
    }

    void TearDown() override {
        processor_->Shutdown();
        delete processor_;
        processor_ = nullptr;
    }

 protected:
    processing::Processor *processor_ = nullptr;

    // Test data, 4 Rows, 2 Features
    std::vector<double> gh_pairs_ = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1};  // 4 Rows, 8 GH Pairs
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

    std::vector<double> histo1_ = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> histo2_ = {5.0, 6.0, 7.0, 8.0};
};

TEST_F(ProcessorTest, TestLoading) {
    auto base_class = dynamic_cast<processing::Processor *>(processor_);
    ASSERT_NE(base_class, nullptr);
}

TEST_F(ProcessorTest, TestGHEncoding) {
    size_t buf_size;
    auto buffer = processor_->ProcessGHPairs(&buf_size, gh_pairs_);
    size_t expected_size = 24;  // DAM header size
    expected_size += gh_pairs_.size() * 10 * 8;  // Dummy plugin duplicate each number 10x to simulate encryption
    ASSERT_EQ(buf_size, expected_size);

    size_t new_size;
    auto new_buffer = processor_->HandleGHPairs(&new_size, buffer, buf_size);
    // Dummy plugin doesn't change buffer
    ASSERT_EQ(new_size, buf_size);
    ASSERT_EQ(0, memcmp(buffer, new_buffer, buf_size));

    // Clean up
    processor_->FreeBuffer(buffer);
    processor_->FreeBuffer(new_buffer);
}

TEST_F(ProcessorTest, TestAggregation) {
    size_t buf_size;
    auto gh_buffer = processor_->ProcessGHPairs(&buf_size, gh_pairs_);  // Pass the GH pairs to the plugin

    processor_->InitAggregationContext(cuts_, slots_);
    auto buffer = processor_->ProcessAggregation(&buf_size, nodes_);
    auto histos = processor_->HandleAggregation(buffer, buf_size);
    double expected_result[] = {
            1.1, 2.1, 0, 0, 0, 0, 5.1, 6.1, 1.1, 2.1,
            0, 0, 0, 0, 5.1, 6.1, 0, 0, 0, 0,
            7.1, 8.1, 3.1, 4.1, 0, 0, 0, 0, 7.1, 8.1,
            0, 0, 0, 0, 0, 0, 0, 0, 3.1, 4.1
    };

    auto expected_size = sizeof(expected_result)/sizeof(expected_result[0]);

    ASSERT_EQ(expected_size, histos.size()) << "Histograms have different sizes";

    for (size_t i = 0; i < histos.size(); ++i) {
        EXPECT_NEAR(expected_result[i], histos[i], kError) << "Histogram differs at index " << i;
    }

    processor_->FreeBuffer(buffer);
    processor_->FreeBuffer(gh_buffer);
}
