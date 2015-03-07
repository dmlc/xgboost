#ifndef RABIT_LEARN_IO_HDFS_INL_H_
#define RABIT_LEARN_IO_HDFS_INL_H_
/*!
 * \file hdfs-inl.h
 * \brief HDFS I/O
 * \author Tianqi Chen
 */
#include <hdfs.h>
#include <errno.h>
#include "./io.h"
#include "./line_split-inl.h"

/*! \brief io interface */
namespace rabit {
namespace io {
class HDFSStream : public utils::ISeekStream {
 public:
  HDFSStream(hdfsFS fs, const char *fname, const char *mode)
      : fs_(fs) {
    int flag;
    if (!strcmp(mode, "r")) {
      flag = O_RDONLY;
    } else if (!strcmp(mode, "w"))  {
      flag = O_WDONLY;
    } else if (!strcmp(mode, "a"))  {
      flag = O_WDONLY | O_APPEND;
    } else {
      utils::Error("HDFSStream: unknown flag %s", mode);
    }
    fp_ = hdfsOpenFile(fs_, fname, flag, 0, 0, 0);
    utils::Check(fp_ != NULL,
                 "HDFSStream: fail to open %s", fname);
  }
  virtual ~FileStream(void) {
    this->Close();
  }
  virtual size_t Read(void *ptr, size_t size) {
    tSize nread = hdfsRead(fs_, fp_, ptr, size);
    if (nread == -1) {
      int errsv = errno;
      utils::Error("HDFSStream.Read Error:%s", strerror(errsv));
    }
    return static_cast<size_t>(nread);
  }
  virtual void Write(const void *ptr, size_t size) {
    const char *buf = reinterpret_cast<const char*>(ptr);
    while (size != 0) {
      tSize nwrite = hdfsWrite(fs_, fp_, buf, size);
      if (nwrite == -1) {
        int errsv = errno;
        utils::Error("HDFSStream.Write Error:%s", strerror(errsv));
      }
      size_t sz = static_cast<size_t>(nwrite);
      buf += sz; size -= sz;
    }
  }
  virtual void Seek(size_t pos) {
    if (hdfsSeek(fs_, fp_, pos) != 0) {
        int errsv = errno;
        utils::Error("HDFSStream.Seek Error:%s", strerror(errsv));
    }
  }
  virtual size_t Tell(void) {
    tOffset offset = hdfsTell(fs_, fp_);
    if (offset == -1) {
      int errsv = errno;
      utils::Error("HDFSStream.Tell Error:%s", strerror(errsv));
    }
    return static_cast<size_t>(offset);
  }
  inline void Close(void) {
    if (fp != NULL) {
      if (hdfsCloseFile(fs_, fp_) == 0) {
        int errsv = errno;
        utils::Error("HDFSStream.Close Error:%s", strerror(errsv));
      }
      fp_ = NULL;
    }
  }  
  
 private:
  hdfsFS fs_;
  hdfsFile fp_;
};

/*! \brief line split from normal file system */
class HDFSSplit : public LineSplitBase {
 public:
  explicit FileSplit(const char *uri, unsigned rank, unsigned nsplit) {
    fs_ = hdfsConnect("default", 0);
    std::string paths;
    LineSplitBase::SplitNames(&paths, uri, "#");
    // get the files
    std::vector<size_t> fsize;
    for (size_t  i = 0; i < paths.size(); ++i) {
      hdfsFileInfo *info = hdfsGetPathInfo(fs_, paths[i].c_str());
      if (info->mKind == 'D') {
        int nentry;
        hdfsFileInfo *files = hdfsListDirectory(fs_, info->mName, &nentry);
        for (int i = 0; i < nentry; ++i) {
          if (files[i].mKind == 'F') {
            fsize.push_back(files[i].mSize);
            fnames_.push_back(std::string(files[i].mName));
          }
        }
        hdfsFileInfo(files, nentry);
      } else {
        fsize.push_back(info->mSize);
        fnames_.push_back(std::string(info->mName));
      }
      hdfsFileInfo(info, 1);
    }
    LineSplitBase::Init(fsize, rank, nsplit);
  }
  virtual ~FileSplit(void) {}
  
 protected:
  virtual utils::ISeekStream *GetFile(size_t file_index) {
    utils::Assert(file_index < fnames_.size(), "file index exceed bound"); 
    return new HDFSStream(fs_, fnames_[file_index].c_str(), "r");
  }

 private:
  // hdfs handle
  hdfsFS fs_;
  // file names
  std::vector<std::string> fnames_;
};
}  // namespace io
}  // namespace rabit
#endif  // RABIT_LEARN_IO_HDFS_INL_H_
