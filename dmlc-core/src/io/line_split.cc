// Copyright by Contributors
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <algorithm>
#include "./line_split.h"

namespace dmlc {
namespace io {
size_t LineSplitter::SeekRecordBegin(Stream *fi) {
  char c = '\0';
  size_t nstep = 0;
  // search till fist end-of-line
  while (true) {
    if (fi->Read(&c, sizeof(c)) == 0) return nstep;
    nstep += 1;
    if (c == '\n' || c == '\r') break;
  }
  // search until first non-endofline
  while (true) {
    if (fi->Read(&c, sizeof(c)) == 0) return nstep;
    if (c != '\n' && c != '\r') break;
    // non-end-of-line should not count
    nstep += 1;
  }
  return nstep;
}
const char* LineSplitter::FindLastRecordBegin(const char *begin,
                                              const char *end) {
  CHECK(begin != end);
  for (const char *p = end - 1; p != begin; --p) {
    if (*p == '\n' || *p == '\r') return p + 1;
  }
  return begin;
}

bool LineSplitter::ExtractNextRecord(Blob *out_rec, Chunk *chunk) {
  if (chunk->begin == chunk->end) return false;
  char *p;
  for (p = chunk->begin; p != chunk->end; ++p) {
    if (*p == '\n' || *p == '\r') break;
  }
  for (; p != chunk->end; ++p) {
    if (*p != '\n' && *p != '\r') break;
  }
  // set the string end sign for safety
  if (p == chunk->end) {
    *p = '\0';
  } else {
    *(p - 1) = '\0';
  }
  out_rec->dptr = chunk->begin;
  out_rec->size = p - chunk->begin;
  chunk->begin = p;
  return true;
}

}  // namespace io
}  // namespace dmlc
