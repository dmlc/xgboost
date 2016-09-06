#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <dmlc/io.h>
#include <dmlc/recordio.h>

int main(int argc, char *argv[]) {  
  if (argc < 5) {
    printf("Usage: <filename> partid npart nmax\n");
    return 0;
  }
  using namespace dmlc;
  dmlc::InputSplit *in = dmlc::InputSplit::
      Create(argv[1],
             atoi(argv[2]),
             atoi(argv[3]),
             "text");
  size_t nmax = static_cast<size_t>(atol(argv[4]));
  size_t lcnt = 0;
  InputSplit::Blob rec;
  std::vector<std::string> data;
  while (in->NextRecord(&rec)) {
    data.push_back(std::string((char*)rec.dptr, rec.size));
    ++lcnt;
    if (lcnt == nmax) {
      LOG(INFO) << "finish loading " << lcnt << " lines";
      break;
    }
  }
  LOG(INFO) << "Call BeforeFirst when lcnt="
            << lcnt << " nmax=" << nmax;
  in->BeforeFirst();
  lcnt = 0;
  while (in->NextRecord(&rec)) {
    std::string dat = std::string((char*)rec.dptr, rec.size);
    if (lcnt < nmax) {
      CHECK(rec.size == data[lcnt].length());
      CHECK(!memcmp(rec.dptr, BeginPtr(data[lcnt]), rec.size));
    } else {
      data.push_back(dat);
    }
    ++lcnt;
  }
  LOG(INFO) << "Call BeforeFirst again";
  in->BeforeFirst();
  lcnt = 0;
  while (in->NextRecord(&rec)) {
    std::string dat = std::string((char*)rec.dptr, rec.size);
    CHECK(lcnt < data.size());
    CHECK(rec.size == data[lcnt].length());
    CHECK(!memcmp(rec.dptr, BeginPtr(data[lcnt]), rec.size));
    ++lcnt;
  }
  delete in;
  LOG(INFO) << "All tests passed";
  return 0;
}
