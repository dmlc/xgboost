#ifndef RABIT_LEARN_UTILS_IO_H_
#define RABIT_LEARN_UTILS_IO_H_
/*!
 * \file io.h
 * \brief additional stream interface
 * \author Tianqi Chen
 */
namespace rabit {
namespace utils {
/*! \brief implementation of file i/o stream */
class FileStream : public ISeekStream {
 public:
  explicit FileStream(FILE *fp) : fp(fp) {}
  explicit FileStream(void) {
    this->fp = NULL;
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, size, 1, fp);
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
  inline void Close(void) {
    if (fp != NULL){
      std::fclose(fp); fp = NULL;
    }
  }

 private:
  FILE *fp;
};
}  // namespace utils
}  // namespace rabit
#endif  // RABIT_LEARN_UTILS_IO_H_
