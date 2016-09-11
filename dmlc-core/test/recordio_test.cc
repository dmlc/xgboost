#include <string>
#include <cstdlib>
#include <dmlc/io.h>
#include <dmlc/recordio.h>

int main(int argc, char *argv[]) {  
  if (argc < 4) {
    printf("Usage: <filename> ndata dlen [nsplit]\n");
    return 0;
  }
  using namespace dmlc;
  int nsplit = 4;
  if (argc > 4) nsplit = atoi(argv[4]);
  LOG(INFO) << "generate the test-cases into";
  int ndata = atoi(argv[2]);
  int dlen = atoi(argv[3]);
  std::vector<std::string> data;  
  const unsigned kMagic = dmlc::RecordIOWriter::kMagic;
  for (int i = 0; i < ndata; ++i) {
    std::string s;
    s.resize(rand() % dlen);
    // generate random string
    for (size_t j = 0; j < s.length(); ++j) {
      s[j] = static_cast<char>(rand() & 255);
    }
    int rnd = rand() % 4;
    if (rnd == 4) {
      size_t n = s.length();
      s.resize(s.length() + 4);
      std::memcpy(BeginPtr(s) + n, &kMagic, sizeof(kMagic));
    } else if (rnd == 3) {
      s.resize(std::max(s.length(), 4UL));
      std::memcpy(BeginPtr(s), &kMagic, sizeof(kMagic));      
    } else if (rnd == 2) {
      for (size_t k = 0; k + 4 <= s.length(); k += 4) {
        if (rand() % 2) {
          std::memcpy(BeginPtr(s) + 4, &kMagic, sizeof(kMagic));
        }
      }
    } else if (rnd == 1) {
      for (size_t k = 0; k + 4 <= s.length(); k += 4) {
        if (rand() % 10) {
          std::memcpy(BeginPtr(s) + 4, &kMagic, sizeof(kMagic));
        }
      }
    }
    data.push_back(s);    
  }
  LOG(INFO) << "generate the test-cases into" << argv[1];  
  {// output
    dmlc::Stream *fs = dmlc::Stream::Create(argv[1], "wb");
    dmlc::RecordIOWriter writer(fs);
    for (size_t i = 0; i < data.size(); ++i) {
      writer.WriteRecord(data[i]);
    }
    delete fs;
    printf("finish writing with %lu exceptions\n", writer.except_counter());
  }
  {// input
    LOG(INFO) << "Test RecordIOReader..";
    dmlc::Stream *fi = dmlc::Stream::Create(argv[1], "r");
    dmlc::RecordIOReader reader(fi);
    std::string temp;
    size_t lcnt = 0;
    while (reader.NextRecord(&temp)) {
      CHECK(lcnt < data.size());
      CHECK(temp.length() == data[lcnt].length());
      if (temp.length() != 0) {
        CHECK(!memcmp(BeginPtr(temp), BeginPtr(data[lcnt]), temp.length()));
      }
      ++lcnt;
    }
    delete fi;
    LOG(INFO) << "Test RecordIOReader.. Pass";
  }
  {// InputSplit::RecordiO
    LOG(INFO) << "Test InputSplit for RecordIO..";
    size_t lcnt = 0;
    for (int i = 0; i < nsplit; ++i) {
      InputSplit::Blob rec;
      dmlc::InputSplit *split = InputSplit::Create(argv[1], i, nsplit, "recordio");
      while (split->NextRecord(&rec)) {
        CHECK(lcnt < data.size());
        CHECK(rec.size == data[lcnt].length());
        if (rec.size != 0) {
          CHECK(!memcmp(rec.dptr, BeginPtr(data[lcnt]), rec.size));
        }
        ++lcnt;
      }
      delete split;
    }
    LOG(INFO) << "Test InputSplit for RecordIO.. Pass";
  }
  {// InputSplit::RecordIO Chunk Read
    LOG(INFO) << "Test InputSplit for RecordIO.. ChunkReader";
    size_t lcnt = 0;
    InputSplit::Blob chunk;
    dmlc::InputSplit *split = InputSplit::Create(argv[1], 0, 1, "recordio");
    while (split->NextChunk(&chunk)) {
      for (int i = 0; i < nsplit; ++i) {
        InputSplit::Blob rec;
        dmlc::RecordIOChunkReader reader(chunk, i, nsplit);
        while (reader.NextRecord(&rec)) {
          CHECK(lcnt < data.size());
          CHECK(rec.size == data[lcnt].length());
          if (rec.size != 0) {
            CHECK(!memcmp(rec.dptr, BeginPtr(data[lcnt]), rec.size));
          }
          ++lcnt;
        }
      }
    }
    delete split;
    LOG(INFO) << "Test InputSplit for RecordIO.. ChunkReader Pass";
  }
  return 0;
}
