#ifndef RABIT_LEARN_IO_LINE_SPLIT_INL_H_
#define RABIT_LEARN_IO_LINE_SPLIT_INL_H_
/*!
 * \std::FILE line_split-inl.h
 * \brief base implementation of line-spliter
 * \author Tianqi Chen
 */
#include <vector>
#include <utility>
#include <cstring>
#include <string>
#include "../../include/rabit.h"
#include "./io.h"
#include "./buffer_reader-inl.h"

namespace rabit {
namespace io {
class LineSplitBase : public InputSplit {
 public:
  virtual ~LineSplitBase() {
    this->Destroy();
  }
  virtual bool NextLine(std::string *out_data) {
    if (file_ptr_ >= file_ptr_end_ &&
        offset_curr_ >= offset_end_) return false;
    out_data->clear();
    while (true) {
      char c = reader_.GetChar();
      if (reader_.AtEnd()) {
        if (out_data->length() != 0) return true;
        file_ptr_ += 1;
        if (offset_curr_ != file_offset_[file_ptr_]) {
          utils::Error("warning:std::FILE size not calculated correctly\n");
          offset_curr_ = file_offset_[file_ptr_];
        }
        if (offset_curr_ >= offset_end_) return false;
        utils::Assert(file_ptr_ + 1 < file_offset_.size(),
                      "boundary check");
        delete fs_;
        fs_ = this->GetFile(file_ptr_);
        reader_.set_stream(fs_);
      } else {
        ++offset_curr_;
        if (c != '\r' && c != '\n' && c != EOF) {
          *out_data += c;
        } else {
          if (out_data->length() != 0) return true;
          if (file_ptr_ >= file_ptr_end_ &&
              offset_curr_ >= offset_end_) return false;
        }
      }
    }
  }

 protected:
  // constructor
  LineSplitBase(void)
      : fs_(NULL), reader_(kBufferSize) {
  }
  /*!
   * \brief destroy all the filesystem resources owned
   * can be called by child destructor
   */
  inline void Destroy(void) {
    if (fs_ != NULL) {
      delete fs_; fs_ = NULL;
    }
  }
  /*!
   * \brief initialize the line spliter,
   * \param file_size, size of each std::FILEs
   * \param rank the current rank of the data
   * \param nsplit number of split we will divide the data into
   */
  inline void Init(const std::vector<size_t> &file_size,
                   unsigned rank, unsigned nsplit) {
    file_offset_.resize(file_size.size() + 1);
    file_offset_[0] = 0;
    for (size_t i = 0; i < file_size.size(); ++i) {
      file_offset_[i + 1] = file_offset_[i] + file_size[i];
    }
    size_t ntotal = file_offset_.back();
    size_t nstep = (ntotal + nsplit - 1) / nsplit;
    offset_begin_ = std::min(nstep * rank, ntotal);
    offset_end_ = std::min(nstep * (rank + 1), ntotal);    
    offset_curr_ = offset_begin_;
    if (offset_begin_ == offset_end_) return;
    file_ptr_ = std::upper_bound(file_offset_.begin(),
                                 file_offset_.end(),
                                 offset_begin_) - file_offset_.begin() - 1;
    file_ptr_end_ = std::upper_bound(file_offset_.begin(),
                                     file_offset_.end(),
                                     offset_end_) - file_offset_.begin() - 1;
    fs_ = GetFile(file_ptr_);
    reader_.set_stream(fs_);
    // try to set the starting position correctly
    if (file_offset_[file_ptr_] != offset_begin_) {
      fs_->Seek(offset_begin_ - file_offset_[file_ptr_]);
      while (true) {
        char c = reader_.GetChar(); 
        if (!reader_.AtEnd()) ++offset_curr_;
        if (c == '\n' || c == '\r' || c == EOF) return;
      }
    }
  }
  /*!
   * \brief get the seek stream of given file_index
   * \return the corresponding seek stream at head of std::FILE
   */
  virtual utils::ISeekStream *GetFile(size_t file_index) = 0;
  /*!
   * \brief split names given 
   * \param out_fname output std::FILE names
   * \param uri_ the iput uri std::FILE
   * \param dlm deliminetr
   */
  inline static void SplitNames(std::vector<std::string> *out_fname,
                                const char *uri_,
                                const char *dlm) {
    std::string uri = uri_;
    char *p = std::strtok(BeginPtr(uri), dlm);
    while (p != NULL) {
      out_fname->push_back(std::string(p));
      p = std::strtok(NULL, dlm);
    }
  }
 private:
  /*! \brief current input stream */
  utils::ISeekStream *fs_;
  /*! \brief std::FILE pointer of which std::FILE to read on */
  size_t file_ptr_;
  /*! \brief std::FILE pointer where the end of std::FILE lies */
  size_t file_ptr_end_;
  /*! \brief get the current offset */
  size_t offset_curr_;
  /*! \brief beginning of offset */
  size_t offset_begin_;
  /*! \brief end of the offset */
  size_t offset_end_;
  /*! \brief byte-offset of each std::FILE */
  std::vector<size_t> file_offset_;
  /*! \brief buffer reader */
  StreamBufferReader reader_;
  /*! \brief buffer size */
  const static size_t kBufferSize = 256;  
};

/*! \brief line split from single std::FILE */
class SingleFileSplit : public InputSplit {
 public:
  explicit SingleFileSplit(const char *fname) {
    if (!std::strcmp(fname, "stdin")) {
#ifndef RABIT_STRICT_CXX98_
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
  virtual bool NextLine(std::string *out_data) {
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
}  // namespace io
}  // namespace rabit
#endif  // RABIT_LEARN_IO_LINE_SPLIT_INL_H_
