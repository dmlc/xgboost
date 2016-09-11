// Copyright by Contributors
#include <dmlc/logging.h>
#include <limits>
#include "./hdfs_filesys.h"

namespace dmlc {
namespace io {
// implementation of HDFS stream
class HDFSStream : public SeekStream {
 public:
  HDFSStream(hdfsFS fs,
             int *ref_counter,
             hdfsFile fp)
      : fs_(fs), ref_counter_(ref_counter),
        fp_(fp) {
  }

  virtual ~HDFSStream(void) {
    this->Close();
    ref_counter_[0] -= 1;
    if (ref_counter_[0] == 0) {
      delete ref_counter_;
      if (hdfsDisconnect(fs_) != 0) {
        int errsv = errno;
        LOG(FATAL) << "HDFSStream.hdfsDisconnect Error:" << strerror(errsv);
      }
    }
  }

  virtual size_t Read(void *ptr, size_t size) {
    char *buf = static_cast<char*>(ptr);
    size_t nleft = size;
    while (nleft != 0) {
      size_t nmax = static_cast<size_t>(std::numeric_limits<tSize>::max());
      tSize ret = hdfsRead(fs_, fp_, buf, std::min(nleft, nmax));
      if (ret > 0) {
        size_t n = static_cast<size_t>(ret);
        nleft -= n; buf += n;
      } else if (ret == 0) {
        break;
      } else {
        int errsv = errno;
        if (errno == EINTR) continue;
        LOG(FATAL) << "HDFSStream.hdfsRead Error:" << strerror(errsv);
      }
    }
    return size - nleft;
  }

  virtual void Write(const void *ptr, size_t size) {
    const char *buf = reinterpret_cast<const char*>(ptr);
    while (size != 0) {
      tSize nwrite = hdfsWrite(fs_, fp_, buf, size);
      if (nwrite == -1) {
        int errsv = errno;
        LOG(FATAL) << "HDFSStream.hdfsWrite Error:" << strerror(errsv);
      }
      size_t sz = static_cast<size_t>(nwrite);
      buf += sz; size -= sz;
    }
  }
  virtual void Seek(size_t pos) {
    if (hdfsSeek(fs_, fp_, pos) != 0) {
      int errsv = errno;
      LOG(FATAL) << "HDFSStream.hdfsSeek Error:" << strerror(errsv);
    }
  }
  virtual size_t Tell(void) {
    tOffset offset = hdfsTell(fs_, fp_);
    if (offset == -1) {
      int errsv = errno;
      LOG(FATAL) << "HDFSStream.hdfsTell Error:" << strerror(errsv);
    }
    return static_cast<size_t>(offset);
  }
  inline void Close(void) {
    if (fp_ != NULL) {
      if (hdfsCloseFile(fs_, fp_) == -1) {
        int errsv = errno;
        LOG(FATAL) << "HDFSStream.hdfsClose Error:" << strerror(errsv);
      }
      fp_ = NULL;
    }
  }

 private:
  hdfsFS fs_;
  int *ref_counter_;
  hdfsFile fp_;
};

HDFSFileSystem::HDFSFileSystem(const std::string &namenode): namenode_(namenode) {
  fs_ = hdfsConnect(namenode_.c_str(), 0);
  if (fs_ == NULL) {
    LOG(FATAL) << "Failed to load HDFS-configuration:";
  }
  ref_counter_ = new int();
  ref_counter_[0] = 1;
}

HDFSFileSystem::~HDFSFileSystem(void) {
  ref_counter_[0] -= 1;
  if (ref_counter_[0] == 0) {
    delete ref_counter_;
    if (hdfsDisconnect(fs_) != 0) {
      int errsv = errno;
      LOG(FATAL) << "HDFSStream.hdfsDisconnect Error:" << strerror(errsv);
    }
  }
}

void HDFSFileSystem::ResetNamenode(const std::string &namenode) {
  if (hdfsDisconnect(fs_) != 0) {
    int errsv = errno;
    LOG(FATAL) << "HDFSStream.hdfsDisconnect Error: " << strerror(errsv);
  }

  namenode_ = namenode;
  fs_ = hdfsConnect(namenode_.c_str(), 0);
  if (fs_ == NULL) {
    LOG(FATAL) << "Failed to load HDFS-configuration: " << namenode_.c_str();
  }
  ref_counter_[0] = 1;
}

inline FileInfo ConvertPathInfo(const URI &path, const hdfsFileInfo &info) {
  FileInfo ret;
  ret.size = info.mSize;
  switch (info.mKind) {
    case 'D': ret.type = kDirectory; break;
    case 'F': ret.type = kFile; break;
    default: LOG(FATAL) << "unknown file type" << info.mKind;
  }
  URI hpath(info.mName);
  if (hpath.protocol == "hdfs://") {
    ret.path = hpath;
  } else {
    ret.path = path;
    ret.path.name = info.mName;
  }
  return ret;
}

FileInfo HDFSFileSystem::GetPathInfo(const URI &path) {
  CHECK(path.protocol == "hdfs://") << "HDFSFileSystem only works with hdfs";
  hdfsFileInfo *info = hdfsGetPathInfo(fs_, path.str().c_str());
  CHECK(info != NULL) << "Path do not exist:" << path.str();
  FileInfo ret = ConvertPathInfo(path, *info);
  hdfsFreeFileInfo(info, 1);
  return ret;
}

void HDFSFileSystem::ListDirectory(const URI &path, std::vector<FileInfo> *out_list) {
  int nentry;
  hdfsFileInfo *files = hdfsListDirectory(fs_, path.str().c_str(), &nentry);
  CHECK(files != NULL) << "Error when ListDirectory " << path.str();
  out_list->clear();
  for (int i = 0; i < nentry; ++i) {
    out_list->push_back(ConvertPathInfo(path, files[i]));
  }
  hdfsFreeFileInfo(files, nentry);
}

SeekStream *HDFSFileSystem::Open(const URI &path,
                                 const char* const mode,
                                 bool allow_null) {
  using namespace std;
  int flag = 0;
  if (!strcmp(mode, "r")) {
    flag = O_RDONLY;
  } else if (!strcmp(mode, "w"))  {
    flag = O_WRONLY;
  } else if (!strcmp(mode, "a"))  {
    flag = O_WRONLY | O_APPEND;
  } else {
    LOG(FATAL) << "HDFSStream: unknown flag %s" << mode;
  }
  hdfsFile fp_ = hdfsOpenFile(fs_, path.str().c_str(), flag, 0, 0, 0);
  if (fp_ != NULL) {
    ref_counter_[0] += 1;
    return new HDFSStream(fs_, ref_counter_, fp_);
  }
  CHECK(allow_null) << " HDFSFileSystem: fail to open \"" << path.str() << '\"';
  return NULL;
}

SeekStream *HDFSFileSystem::OpenForRead(const URI &path, bool allow_null) {
  return Open(path, "r", allow_null);
}
}  // namespace io
}  // namespace dmlc
