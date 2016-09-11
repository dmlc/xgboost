/*!
 *  Copyright (c) 2015 by Contributors
 * \file local_filesys.h
 * \brief local access module
 * \author Tianqi Chen
 */
#ifndef DMLC_IO_LOCAL_FILESYS_H_
#define DMLC_IO_LOCAL_FILESYS_H_

#include <vector>
#include "./filesys.h"

namespace dmlc {
namespace io {
/*! \brief local file system */
class LocalFileSystem : public FileSystem {
 public:
  /*! \brief destructor */
  virtual ~LocalFileSystem() {}
  /*!
   * \brief get information about a path
   * \param path the path to the file
   * \return the information about the file
   */
  virtual FileInfo GetPathInfo(const URI &path);
  /*!
   * \brief list files in a directory
   * \param path to the file
   * \param out_list the output information about the files
   */
  virtual void ListDirectory(const URI &path, std::vector<FileInfo> *out_list);
  /*!
   * \brief open a stream, will report error and exit if bad thing happens
   * NOTE: the IStream can continue to work even when filesystem was destructed
   * \param path path to file
   * \param uri the uri of the input
   * \param allow_null whether NULL can be returned, or directly report error
   * \return the created stream, can be NULL when allow_null == true and file do not exist
   */
  virtual SeekStream *Open(const URI &path,
                           const char* const flag,
                           bool allow_null);
  /*!
   * \brief open a seekable stream for read
   * \param path the path to the file
   * \param allow_null whether NULL can be returned, or directly report error
   * \return the created stream, can be NULL when allow_null == true and file do not exist
   */
  virtual SeekStream *OpenForRead(const URI &path, bool allow_null);
  /*!
   * \brief get a singleton of LocalFileSystem when needed
   * \return a singleton instance
   */
  inline static LocalFileSystem *GetInstance(void) {
    static LocalFileSystem instance;
    return &instance;
  }

 private:
  LocalFileSystem() {}
};
}  // namespace io
}  // namespace dmlc
#endif  // DMLC_IO_LOCAL_FILESYS_H_
