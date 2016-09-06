/*!
 *  Copyright (c) 2015 by Contributors
 * \file filesystem.h
 * \brief general file system io interface
 * \author Tianqi Chen
 */
#ifndef DMLC_IO_FILESYS_H_
#define DMLC_IO_FILESYS_H_

#include <dmlc/io.h>
#include <cstring>
#include <string>
#include <vector>

namespace dmlc {
namespace io {
/*! \brief common data structure for URI */
struct URI {
  /*! \brief protocol */
  std::string protocol;
  /*!
   * \brief host name, namenode for HDFS, bucket name for s3
   */
  std::string host;
  /*! \brief name of the path */
  std::string name;
  /*! \brief enable default constructor */
  URI(void) {}
  /*!
   * \brief construct from URI string
   */
  explicit URI(const char *uri) {
    const char *p = std::strstr(uri, "://");
    if (p == NULL) {
      name = uri;
    } else {
      protocol = std::string(uri, p - uri + 3);
      uri = p + 3;
      p = std::strchr(uri, '/');
      if (p == NULL) {
        host = uri; name = '/';
      } else {
        host = std::string(uri, p - uri);
        name = p;
      }
    }
  }
  /*! \brief string representation */
  inline std::string str(void) const {
    return protocol + host + name;
  }
};

/*! \brief type of file */
enum FileType {
  /*! \brief the file is file */
  kFile,
  /*! \brief the file is directory */
  kDirectory
};

/*! \brief use to store file information */
struct FileInfo {
  /*! \brief full path to the file */
  URI path;
  /*! \brief the size of the file */
  size_t size;
  /*! \brief the type of the file */
  FileType type;
  /*! \brief default constructor */
  FileInfo() : size(0), type(kFile) {}
};

/*! \brief file system system interface */
class FileSystem {
 public:
  /*!
   * \brief get singleton of filesystem instance according to URI
   * \param path can be s3://..., hdfs://..., file://...,
   *            empty string(will return local)
   * \return a corresponding filesystem, report error if
   *         we cannot find a matching system
   */
  static FileSystem *GetInstance(const URI &path);
  /*! \brief virtual destructor */
  virtual ~FileSystem() {}
  /*!
   * \brief get information about a path
   * \param path the path to the file
   * \return the information about the file
   */
  virtual FileInfo GetPathInfo(const URI &path) = 0;
  /*!
   * \brief list files in a directory
   * \param path to the file
   * \param out_list the output information about the files
   */
  virtual void ListDirectory(const URI &path, std::vector<FileInfo> *out_list) = 0;
  /*!
   * \brief open a stream
   * \param path path to file
   * \param uri the uri of the input, can contain hdfs prefix
   * \param flag can be "w", "r", "a
   * \param allow_null whether NULL can be returned, or directly report error
   * \return the created stream, can be NULL when allow_null == true and file do not exist
   */
  virtual Stream *Open(const URI &path,
                       const char* const flag,
                       bool allow_null = false) = 0;
  /*!
   * \brief open a seekable stream for read
   * \param path the path to the file
   * \param allow_null whether NULL can be returned, or directly report error
   * \return the created stream, can be NULL when allow_null == true and file do not exist
   */
  virtual SeekStream *OpenForRead(const URI &path,
                                  bool allow_null = false) = 0;
};
}  // namespace io
}  // namespace dmlc
#endif  // DMLC_IO_FILESYS_H_
