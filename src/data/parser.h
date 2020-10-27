/*!
 * Copyright 2020 by XGBoost Contributors
 * \file parser.h
 */

#ifndef XGBOOST_DATA_PARSER_H_
#define XGBOOST_DATA_PARSER_H_
#include <dmlc/io.h>
#include "../../dmlc-core/src/io/uri_spec.h"
#include "../../dmlc-core/src/io/line_split.h"
#include "../../dmlc-core/src/data/csv_parser.h"
#include "../../dmlc-core/src/data/libfm_parser.h"
#include "../../dmlc-core/src/data/libsvm_parser.h"

#include "xgboost/logging.h"

namespace xgboost {
namespace data {

/*
 * Overriding text data processing infrastructures in dmlc core for external memory
 * support.  There are a few issues we are trying to address/workaround here:
 *
 * - Avoid threaded iterator in dmlc, which is not thread safe and relies on C++ memory
 *   model with heavy use of C++ threading primitives.
 * - Avoid the threaded input split, which uses threaded iterator internally.
 * - Override text input parser, which returns data blocks depending on number of system
 *   threads.
 * - Batch size is not respected in dmlc-core, instead it has a buffer size.
 *
 * In general, the infrastructure in dmlc-core is more concerned with
 * performance/parallelism, but here we need consistency for data partitioning and memory
 * usage.
 */

class TextInputSplit : public dmlc::InputSplit {
  dmlc::io::InputSplitBase::Chunk *tmp_chunk_;
  std::unique_ptr<dmlc::io::InputSplitBase> base_;
  size_t batch_size_;

 public:
  TextInputSplit(std::unique_ptr<dmlc::io::InputSplitBase> &&split,
                 size_t batch_size)
      : base_{std::forward<std::unique_ptr<dmlc::io::InputSplitBase>>(split)},
        batch_size_{batch_size} {}

  size_t GetTotalSize() override {
    LOG(FATAL) << "Not implemented";
    return 0;
  }
  bool NextRecord(Blob *out_rec) override {
    LOG(FATAL) << "Not implemented";
    return false;
  }
  void ResetPartition(unsigned part_index, unsigned num_parts) override {
    LOG(FATAL) << "Not implemented";
  }

  bool NextChunkEx(dmlc::io::InputSplitBase::Chunk *chunk) {
    if (!chunk->Load(base_.get(), batch_size_)) return false;
    return true;
  }
  void BeforeFirst() override {
    base_->BeforeFirst();
    if (tmp_chunk_ != nullptr) {
      delete tmp_chunk_;
      tmp_chunk_ = nullptr;
    }
  }
  bool NextChunk(Blob *out_chunk) override {
    if (tmp_chunk_ == nullptr) {
      tmp_chunk_ = new dmlc::io::InputSplitBase::Chunk(batch_size_);
    }

    out_chunk->dptr = tmp_chunk_->begin;
    out_chunk->size = tmp_chunk_->end - tmp_chunk_->begin;

    while (!base_->ExtractNextRecord(out_chunk, tmp_chunk_)) {
      if (!NextChunkEx(tmp_chunk_)) {
        return false;
      }
    }

    return true;
  }
};

inline dmlc::InputSplit *CreateInputSplit(std::string const& uri, unsigned part,
                                          unsigned nsplit, const size_t batch_size) {
  namespace io = dmlc::io;
  io::URISpec spec(uri.c_str(), part, nsplit);
  CHECK(part < nsplit) << "Invalid input parameter for input split.";
  io::URI path(spec.uri.c_str());
  std::unique_ptr<io::InputSplitBase> split{nullptr};
  split.reset(new io::LineSplitter(io::FileSystem::GetInstance(path),
                                   spec.uri.c_str(), part, nsplit));
  CHECK_EQ(spec.cache_file.length(), 0);
  return new TextInputSplit(std::move(split), batch_size);
}

// Due the the parsing implementation in dmlc core, number of blocks equals to number of
// available threads.  This violates external memory so we concatenate all the blocks
// here.
template <typename IndexType, typename DType = float>
void ConcatBlocks(
    std::vector<dmlc::data::RowBlockContainer<IndexType, DType>> *data) {
  if (data->empty()) {
    return;
  }
  auto &block = data->front();
  for (size_t i = 1; i < data->size(); ++i) {
    block.Push(data->at(i).GetBlock());
  }

  data->resize(1);
  data->shrink_to_fit();
}

template <typename IndexType, typename DType = float>
class CSVParser : public dmlc::data::CSVParser<IndexType, DType> {
  using dmlc::data::CSVParser<IndexType, DType>::CSVParser;
  using TextParserBase = dmlc::data::TextParserBase<IndexType, DType>;

  bool ParseNext(std::vector<dmlc::data::RowBlockContainer<IndexType, DType> > *data) override {
    auto ret = TextParserBase::FillData(data);
    ConcatBlocks(data);
    return ret;
  }
};

template <typename IndexType, typename DType = float>
class LibFMParser : public dmlc::data::LibFMParser<IndexType, DType> {
  using dmlc::data::LibFMParser<IndexType, DType>::LibFMParser;
  using TextParserBase = dmlc::data::TextParserBase<IndexType, DType>;

  bool ParseNext(std::vector<dmlc::data::RowBlockContainer<IndexType, DType> > *data) override {
    auto ret = TextParserBase::FillData(data);
    ConcatBlocks(data);
    return ret;
  }
};

template <typename IndexType, typename DType = float>
class LibSVMParser : public dmlc::data::LibSVMParser<IndexType, DType> {
  using dmlc::data::LibSVMParser<IndexType, DType>::LibSVMParser;
  using TextParserBase = dmlc::data::TextParserBase<IndexType, DType>;

  bool ParseNext(std::vector<dmlc::data::RowBlockContainer<IndexType, DType> > *data) override {
    auto ret = TextParserBase::FillData(data);
    ConcatBlocks(data);
    return ret;
  }
};

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType, DType> *
CreateCSVParser(const std::string& path,
                const std::map<std::string, std::string>& args,
                unsigned part_index,
                unsigned num_parts, size_t batch_size) {
  dmlc::InputSplit *source = CreateInputSplit(path, part_index, num_parts, batch_size);
  return new CSVParser<IndexType, DType>(source, args, 2);
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType> *
CreateLibSVMParser(const std::string& path,
                   const std::map<std::string, std::string>& args,
                   unsigned part_index,
                   unsigned num_parts, size_t batch_size) {
  dmlc::InputSplit *source = CreateInputSplit(path, part_index, num_parts, batch_size);
  dmlc::data::ParserImpl<IndexType> *parser =
      new LibSVMParser<IndexType>(source, args, 2);
  return parser;
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType> *
CreateLibFMParser(const std::string& path,
                  const std::map<std::string, std::string>& args,
                  unsigned part_index,
                  unsigned num_parts, size_t batch_size) {
  dmlc::InputSplit *source = CreateInputSplit(path, part_index, num_parts, batch_size);
  dmlc::data::ParserImpl<IndexType> *parser =
      new LibFMParser<IndexType>(source, args, 2);
  return parser;
}

template <typename IndexType, typename DType = float>
inline dmlc::Parser<IndexType, DType> *
CreateParser(std::string uri, unsigned part_index, unsigned num_parts,
             std::string type) {
  dmlc::io::URISpec spec(uri.c_str(), part_index, num_parts);
  size_t batch_size = 256;
  if (type == "auto") {
    if (spec.args.count("format") != 0) {
      type = spec.args.at("format");
    } else {
      type = "libsvm";
    }
    if (spec.args.count("batch_size")) {
      try {
        batch_size = std::stoull(spec.args.at("batch_size"));
      } catch (std::invalid_argument const& e) {
        LOG(FATAL) << e.what();
      } catch (std::out_of_range const& e) {
        LOG(FATAL) << e.what();
      }
    }
  }
  CHECK_GT(batch_size, 0);

  // create parser
  if (type == "csv") {
    return CreateCSVParser<IndexType, DType>(spec.uri, spec.args, part_index,
                                              num_parts, batch_size);
  } else if (type == "libsvm") {
    return CreateLibSVMParser<IndexType, DType>(spec.uri, spec.args, part_index,
                                                num_parts, batch_size);
  } else if (type == "libfm") {
    return CreateLibFMParser<IndexType, DType>(spec.uri, spec.args, part_index,
                                               num_parts, batch_size);
  } else {
    LOG(FATAL) << "Unknown file format: " << type;
    return nullptr;
  }
}
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_PARSER_H_
