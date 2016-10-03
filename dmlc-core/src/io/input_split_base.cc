// Copyright by Contributors
#include <dmlc/logging.h>
#include <dmlc/common.h>
#include <algorithm>
#include "./line_split.h"

#if DMLC_USE_REGEX
#include <regex>
#endif

namespace dmlc {
namespace io {
void InputSplitBase::Init(FileSystem *filesys,
                          const char *uri,
                          size_t align_bytes) {
  this->filesys_ = filesys;
  // initialize the path
  this->InitInputFileInfo(uri);
  file_offset_.resize(files_.size() + 1);
  file_offset_[0] = 0;
  for (size_t i = 0; i < files_.size(); ++i) {
    file_offset_[i + 1] = file_offset_[i] + files_[i].size;
    CHECK(files_[i].size % align_bytes == 0)
        << "file do not align by " << align_bytes << " bytes";
  }
  this->align_bytes_ = align_bytes;
}

void InputSplitBase::ResetPartition(unsigned rank,
                                    unsigned nsplit) {
  size_t ntotal = file_offset_.back();
  size_t nstep = (ntotal + nsplit - 1) / nsplit;
  // align the nstep to 4 bytes
  nstep = ((nstep + align_bytes_ - 1) / align_bytes_) * align_bytes_;
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
  if (fs_ != NULL) {
    delete fs_; fs_ = NULL;
  }
  // find the exact ending position
  if (offset_end_ != file_offset_[file_ptr_end_]) {
    CHECK(offset_end_ >file_offset_[file_ptr_end_]);
    CHECK(file_ptr_end_ < files_.size());
    fs_ = filesys_->OpenForRead(files_[file_ptr_end_].path);
    fs_->Seek(offset_end_ - file_offset_[file_ptr_end_]);
    offset_end_ += SeekRecordBegin(fs_);
    delete fs_;
  }
  fs_ = filesys_->OpenForRead(files_[file_ptr_].path);
  if (offset_begin_ != file_offset_[file_ptr_]) {
    fs_->Seek(offset_begin_ - file_offset_[file_ptr_]);
    offset_begin_ += SeekRecordBegin(fs_);
  }
  this->BeforeFirst();
}

void InputSplitBase::BeforeFirst(void) {
  if (offset_begin_ >= offset_end_) return;
  size_t fp = std::upper_bound(file_offset_.begin(),
                               file_offset_.end(),
                               offset_begin_) - file_offset_.begin() - 1;
  if (file_ptr_ != fp) {
    delete fs_;
    file_ptr_ = fp;
    fs_ = filesys_->OpenForRead(files_[file_ptr_].path);
  }
  // seek to beginning of stream
  fs_->Seek(offset_begin_ - file_offset_[file_ptr_]);
  offset_curr_ = offset_begin_;
  tmp_chunk_.begin = tmp_chunk_.end = NULL;
  // clear overflow buffer
  overflow_.clear();
}

InputSplitBase::~InputSplitBase(void) {
  delete fs_;
  // no need to delete filesystem, it was singleton
}

std::string InputSplitBase::StripEnd(std::string str, char ch) {
  while (str.length() != 0 && str[str.length() - 1] == ch) {
    str.resize(str.length() - 1);
  }
  return str;
}

void InputSplitBase::InitInputFileInfo(const std::string& uri) {
  // split by :
  const char dlm = ';';
  std::vector<std::string> file_list = Split(uri, dlm);
  std::vector<URI> expanded_list;

  // expand by match regex pattern.
  for (size_t i = 0; i < file_list.size(); ++i) {
    URI path(file_list[i].c_str());
    size_t pos = path.name.rfind('/');
    if (pos == std::string::npos || pos + 1 == path.name.length()) {
      expanded_list.push_back(path);
    } else {
      URI dir = path;
      dir.name = path.name.substr(0, pos);
      std::vector<FileInfo> dfiles;
      filesys_->ListDirectory(dir, &dfiles);
      bool exact_match = false;
      for (size_t i = 0; i < dfiles.size(); ++i) {
        if (StripEnd(dfiles[i].path.name, '/') == StripEnd(path.name, '/')) {
          expanded_list.push_back(dfiles[i].path);
          exact_match = true;
          break;
        }
      }
#if DMLC_USE_REGEX
      if (!exact_match) {
        std::string spattern = path.name;
        try {
          std::regex pattern(spattern);
          for (size_t i = 0; i < dfiles.size(); ++i) {
            if (dfiles[i].type != kFile || dfiles[i].size == 0) continue;
            std::string stripped = StripEnd(dfiles[i].path.name, '/');
            std::smatch base_match;
            if (std::regex_match(stripped, base_match, pattern)) {
              for (size_t j = 0; j < base_match.size(); ++j) {
                if (base_match[j].str() == stripped) {
                  expanded_list.push_back(dfiles[i].path); break;
                }
              }
            }
          }
        } catch (std::regex_error& e) {
          LOG(FATAL) << e.what() << " bad regex " << spattern
                     << "This could due to compiler version, g++-4.9 is needed";
        }
      }
#endif  // DMLC_USE_REGEX
    }
  }

  for (size_t i = 0; i < expanded_list.size(); ++i) {
    const URI& path = expanded_list[i];
    FileInfo info = filesys_->GetPathInfo(path);
    if (info.type == kDirectory) {
      std::vector<FileInfo> dfiles;
      filesys_->ListDirectory(info.path, &dfiles);
      for (size_t i = 0; i < dfiles.size(); ++i) {
        if (dfiles[i].size != 0 && dfiles[i].type == kFile) {
          files_.push_back(dfiles[i]);
        }
      }
    } else {
      if (info.size != 0) {
        files_.push_back(info);
      }
    }
  }
  CHECK_NE(files_.size(), 0)
      << "Cannot find any files that matches the URI patternz " << uri;
}

size_t InputSplitBase::Read(void *ptr, size_t size) {
  if (offset_begin_ >= offset_end_) return 0;
  if (offset_curr_ +  size > offset_end_) {
    size = offset_end_ - offset_curr_;
  }
  if (size == 0) return 0;
  size_t nleft = size;
  char *buf = reinterpret_cast<char*>(ptr);
  while (true) {
    size_t n = fs_->Read(buf, nleft);
    nleft -= n; buf += n;
    offset_curr_ += n;
    if (nleft == 0) break;
    if (n == 0) {
      if (offset_curr_ != file_offset_[file_ptr_ + 1]) {
        LOG(ERROR) << "curr=" << offset_curr_
                   << ",begin=" << offset_begin_
                   << ",end=" << offset_end_
                   << ",fileptr=" << file_ptr_
                   << ",fileoffset=" << file_offset_[file_ptr_ + 1];
        for (size_t i = 0; i < file_ptr_; ++i) {
          LOG(ERROR) << "offset[" << i << "]=" << file_offset_[i];
        }
        LOG(FATAL) << "file offset not calculated correctly";
      }
      if (file_ptr_ + 1 >= files_.size()) break;
      file_ptr_ += 1;
      delete fs_;
      fs_ = filesys_->OpenForRead(files_[file_ptr_].path);
    }
  }
  return size - nleft;
}

bool InputSplitBase::ReadChunk(void *buf, size_t *size) {
  size_t max_size = *size;
  if (max_size <= overflow_.length()) {
    *size = 0; return true;
  }
  if (overflow_.length() != 0) {
    std::memcpy(buf, BeginPtr(overflow_), overflow_.length());
  }
  size_t olen = overflow_.length();
  overflow_.resize(0);
  size_t nread = this->Read(reinterpret_cast<char*>(buf) + olen,
                            max_size - olen);
  nread += olen;
  if (nread == 0) return false;
  if (nread != max_size) {
    *size = nread;
    return true;
  } else {
    const char *bptr = reinterpret_cast<const char*>(buf);
    // return the last position where a record starts
    const char *bend = this->FindLastRecordBegin(bptr, bptr + max_size);
    *size = bend - bptr;
    overflow_.resize(max_size - *size);
    if (overflow_.length() != 0) {
      std::memcpy(BeginPtr(overflow_), bend, overflow_.length());
    }
    return true;
  }
}

bool InputSplitBase::Chunk::Load(InputSplitBase *split, size_t buffer_size) {
  if (buffer_size + 1 > data.size()) {
    data.resize(buffer_size + 1);
  }
  while (true) {
    // leave one tail chunk
    size_t size = (data.size() - 1) * sizeof(size_t);
    // set back to 0 for string safety
    data.back() = 0;
    if (!split->ReadChunk(BeginPtr(data), &size)) return false;
    if (size == 0) {
      data.resize(data.size() * 2);
    } else {
      begin = reinterpret_cast<char *>(BeginPtr(data));
      end = begin + size;
      break;
    }
  }
  return true;
}

bool InputSplitBase::ExtractNextChunk(Blob *out_chunk, Chunk *chunk) {
  if (chunk->begin == chunk->end) return false;
  out_chunk->dptr = chunk->begin;
  out_chunk->size = chunk->end - chunk->begin;
  chunk->begin = chunk->end;
  return true;
}
}  // namespace io
}  // namespace dmlc
