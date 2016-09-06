// Copyright by Contributors
#include <dmlc/recordio.h>
#include <dmlc/logging.h>
#include <algorithm>
#include "./recordio_split.h"

namespace dmlc {
namespace io {
size_t RecordIOSplitter::SeekRecordBegin(Stream *fi) {
  size_t nstep = 0;
  uint32_t v, lrec;
  while (true) {
    if (fi->Read(&v, sizeof(v)) == 0) return nstep;
    nstep += sizeof(v);
    if (v == RecordIOWriter::kMagic) {
      CHECK(fi->Read(&lrec, sizeof(lrec)) != 0)
            << "invalid record io format";
      nstep += sizeof(lrec);
      uint32_t cflag = RecordIOWriter::DecodeFlag(lrec);
      if (cflag == 0 || cflag == 1) break;
    }
  }
  // should point at head of record
  return nstep - 2 * sizeof(uint32_t);
}
const char* RecordIOSplitter::FindLastRecordBegin(const char *begin,
                                                  const char *end) {
  CHECK_EQ((reinterpret_cast<size_t>(begin) & 3UL), 0);
  CHECK_EQ((reinterpret_cast<size_t>(end) & 3UL), 0);
  const uint32_t *pbegin = reinterpret_cast<const uint32_t *>(begin);
  const uint32_t *p = reinterpret_cast<const uint32_t *>(end);
  CHECK(p >= pbegin + 2);
  for (p = p - 2; p != pbegin; --p) {
    if (p[0] == RecordIOWriter::kMagic) {
      uint32_t cflag = RecordIOWriter::DecodeFlag(p[1]);
      if (cflag == 0 || cflag == 1) {
        return reinterpret_cast<const char*>(p);
      }
    }
  }
  return begin;
}

bool RecordIOSplitter::ExtractNextRecord(Blob *out_rec, Chunk *chunk) {
  if (chunk->begin == chunk->end) return false;
  CHECK(chunk->begin + 2 * sizeof(uint32_t) <= chunk->end)
      << "Invalid RecordIO Format";
  CHECK_EQ((reinterpret_cast<size_t>(chunk->begin) & 3UL), 0);
  CHECK_EQ((reinterpret_cast<size_t>(chunk->end) & 3UL), 0);
  uint32_t *p = reinterpret_cast<uint32_t *>(chunk->begin);
  uint32_t cflag = RecordIOWriter::DecodeFlag(p[1]);
  uint32_t clen = RecordIOWriter::DecodeLength(p[1]);
  // skip header
  out_rec->dptr = chunk->begin + 2 * sizeof(uint32_t);
  // move pbegin
  chunk->begin += 2 * sizeof(uint32_t) + (((clen + 3U) >> 2U) << 2U);
  CHECK(chunk->begin <= chunk->end) << "Invalid RecordIO Format";
  out_rec->size = clen;
  if (cflag == 0) return true;
  const uint32_t kMagic = RecordIOWriter::kMagic;
  // abnormal path, move data around to make a full part
  CHECK(cflag == 1U) << "Invalid RecordIO Format";
  while (cflag != 3U) {
    CHECK(chunk->begin + 2 * sizeof(uint32_t) <= chunk->end);
    p = reinterpret_cast<uint32_t *>(chunk->begin);
    CHECK(p[0] == RecordIOWriter::kMagic);
    cflag = RecordIOWriter::DecodeFlag(p[1]);
    clen = RecordIOWriter::DecodeLength(p[1]);
    // pad kmagic in between
    std::memcpy(reinterpret_cast<char*>(out_rec->dptr) + out_rec->size,
                &kMagic, sizeof(kMagic));
    out_rec->size += sizeof(kMagic);
    // move the rest of the blobs
    if (clen != 0) {
      std::memmove(reinterpret_cast<char*>(out_rec->dptr) + out_rec->size,
                   chunk->begin + 2 * sizeof(uint32_t), clen);
      out_rec->size += clen;
    }
    chunk->begin += 2 * sizeof(uint32_t) + (((clen + 3U) >> 2U) << 2U);
  }
  return true;
}
}  // namespace io
}  // namespace dmlc
