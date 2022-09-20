/*!
 * Copyright (c) by XGBoost Contributors 2019
 */
#include <gtest/gtest.h>

#include <fstream>

#include "../../../src/common/io.h"
#include "../helpers.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory

namespace xgboost {
namespace common {
TEST(MemoryFixSizeBuffer, Seek) {
  size_t constexpr kSize { 64 };
  std::vector<int32_t> memory( kSize );
  rabit::utils::MemoryFixSizeBuffer buf(memory.data(), memory.size());
  buf.Seek(rabit::utils::MemoryFixSizeBuffer::kSeekEnd);
  size_t end = buf.Tell();
  ASSERT_EQ(end, kSize);
}

TEST(IO, FileExtension) {
  std::string filename {u8"model.json"};
  auto ext = FileExtension(filename);
  ASSERT_EQ(ext, u8"json");
}

TEST(IO, FixedSizeStream) {
  std::string buffer {"This is the content of stream"};
  {
    MemoryFixSizeBuffer stream(static_cast<void *>(&buffer[0]), buffer.size());
    PeekableInStream peekable(&stream);
    FixedSizeStream fixed(&peekable);

    std::string out_buffer;
    fixed.Take(&out_buffer);
    ASSERT_EQ(buffer, out_buffer);
  }

  {
    std::string huge_buffer;
    for (size_t i = 0; i < 512; i++) {
      huge_buffer += buffer;
    }

    MemoryFixSizeBuffer stream(static_cast<void*>(&huge_buffer[0]), huge_buffer.size());
    PeekableInStream peekable(&stream);
    FixedSizeStream fixed(&peekable);

    std::string out_buffer;
    fixed.Take(&out_buffer);
    ASSERT_EQ(huge_buffer, out_buffer);
  }
}

TEST(IO, LoadSequentialFile) {
  EXPECT_THROW(LoadSequentialFile("non-exist"), dmlc::Error);

  dmlc::TemporaryDirectory tempdir;
  std::ofstream fout(tempdir.path + "test_file");
  std::string content;

  // Generate a JSON file.
  size_t constexpr kRows = 1000, kCols = 100;
  std::shared_ptr<DMatrix> p_dmat{
    RandomDataGenerator{kRows, kCols, 0}.GenerateDMatrix(true)};
  std::unique_ptr<Learner> learner { Learner::Create({p_dmat}) };
  learner->SetParam("tree_method", "hist");
  learner->Configure();

  for (int32_t iter = 0; iter < 10; ++iter) {
    learner->UpdateOneIter(iter, p_dmat);
  }
  Json out { Object() };
  learner->SaveModel(&out);
  std::string str;
  Json::Dump(out, &str);

  std::string tmpfile = tempdir.path + "/model.json";
  {
    std::unique_ptr<dmlc::Stream> fo(
        dmlc::Stream::Create(tmpfile.c_str(), "w"));
    fo->Write(str.c_str(), str.size());
  }

  auto loaded = LoadSequentialFile(tmpfile, true);
  ASSERT_EQ(loaded, str);

  ASSERT_THROW(LoadSequentialFile("non-exist", true), dmlc::Error);
}
}  // namespace common
}  // namespace xgboost
