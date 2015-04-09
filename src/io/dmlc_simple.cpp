#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include "../utils/io.h"

// implements a single no split version of DMLC
// in case we want to avoid dependency on dmlc-core

namespace xgboost {
namespace utils {
class SingleFileSplit : public dmlc::InputSplit {
 public:
  explicit SingleFileSplit(const char *fname) 
      : use_stdin_(false) {
    if (!std::strcmp(fname, "stdin")) {
#ifndef XGBOOST_STRICT_CXX98_
      use_stdin_ = true; fp_ = stdin;
#endif
    }
    if (!use_stdin_) {
      fp_ = utils::FopenCheck(fname, "r");
    }
    end_of_file_ = false;
  }
  virtual ~SingleFileSplit(void) {
    if (!use_stdin_) std::fclose(fp_);
  }
  virtual bool ReadLine(std::string *out_data) {
    if (end_of_file_) return false;
    out_data->clear();
    while (true) {
      char c = std::fgetc(fp_);
      if (c == EOF) {
        end_of_file_ = true;
      }
      if (c != '\r' && c != '\n' && c != EOF) {
        *out_data += c;
      } else {
        if (out_data->length() != 0) return true;
        if (end_of_file_) return false;
      }
    }
    return false;
  }  
    
 private:
  std::FILE *fp_;
  bool use_stdin_;
  bool end_of_file_;
};

class StdFile : public dmlc::Stream {
 public:
  explicit StdFile(const char *fname, const char *mode)
      : use_stdio(false) {
    using namespace std;
#ifndef XGBOOST_STRICT_CXX98_
    if (!strcmp(fname, "stdin")) {
      use_stdio = true; fp = stdin;
    }
    if (!strcmp(fname, "stdout")) {
      use_stdio = true; fp = stdout;
    }
#endif
    if (!strncmp(fname, "file://", 7)) fname += 7;
    if (!use_stdio) {
      std::string flag = mode;
      if (flag == "w") flag = "wb";
      if (flag == "r") flag = "rb";
      fp = utils::FopenCheck(fname, flag.c_str());
      
    }
  }  
  virtual ~StdFile(void) {
    this->Close();
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, 1, size, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  virtual void Seek(size_t pos) {
    std::fseek(fp, static_cast<long>(pos), SEEK_SET);
  }
  virtual size_t Tell(void) {
    return std::ftell(fp);
  }
  virtual bool AtEnd(void) const {
    return std::feof(fp) != 0;
  }
  inline void Close(void) {
    if (fp != NULL && !use_stdio) {
      std::fclose(fp); fp = NULL;
    }
  }

 private:
  std::FILE *fp;
  bool use_stdio;
};
}  // namespace utils
}  // namespace xgboost

namespace dmlc {
InputSplit* InputSplit::Create(const char *uri,
                               unsigned part,
                               unsigned nsplit) {
  using namespace xgboost;
  const char *msg = "xgboost is compiled in local mode\n"\
      "to use hdfs, s3 or distributed version, compile with make dmlc=1";
  utils::Check(strncmp(uri, "s3://", 5) != 0, msg);
  utils::Check(strncmp(uri, "hdfs://", 7) != 0, msg);
  utils::Check(nsplit == 1, msg);
  return new utils::SingleFileSplit(uri);
}

Stream *Stream::Create(const char *uri, const char * const flag) {
  using namespace xgboost;
  const char *msg = "xgboost is compiled in local mode\n"\
      "to use hdfs, s3 or distributed version, compile with make dmlc=1";
  utils::Check(strncmp(uri, "s3://", 5) != 0, msg);
  utils::Check(strncmp(uri, "hdfs://", 7) != 0, msg);
  return new utils::StdFile(uri, flag);
}
}  // namespace dmlc

