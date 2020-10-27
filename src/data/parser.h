#ifndef XGBOOST_DATA_PARSER_H_
#define XGBOOST_DATA_PARSER_H_
#include <dmlc/io.h>
#include "../../dmlc-core/src/io/single_threaded_input_split.h"
#include "../../dmlc-core/src/io/single_file_split.h"
#include "../../dmlc-core/src/io/uri_spec.h"
#include "../../dmlc-core/src/io/line_split.h"
#include "../../dmlc-core/src/io/indexed_recordio_split.h"
#include "../../dmlc-core/src/io/recordio_split.h"
#include "../../dmlc-core/src/io/cached_input_split.h"
#include "../../dmlc-core/src/data/csv_parser.h"
#include "../../dmlc-core/src/data/libfm_parser.h"
#include "../../dmlc-core/src/data/libsvm_parser.h"

#include "xgboost/logging.h"

namespace xgboost {
namespace data {
inline dmlc::InputSplit *
CreateInputSplit(const char *uri_, const char *index_uri_, unsigned part,
                 unsigned nsplit, const char *type, const bool shuffle = false,
                 const int seed = 0, const size_t batch_size = 256,
                 const bool recurse_directories = false) {
  namespace io = dmlc::io;
  io::URISpec spec(uri_, part, nsplit);
  if (!strcmp(spec.uri.c_str(), "stdin")) {
    return new io::SingleFileSplit(spec.uri.c_str());
  }
  CHECK(part < nsplit) << "invalid input parameter for InputSplit::Create";
  io::URI path(spec.uri.c_str());
  io::InputSplitBase *split = nullptr;
  if (!strcmp(type, "text")) {
    split = new io::LineSplitter(io::FileSystem::GetInstance(path),
                                 spec.uri.c_str(), part, nsplit);
  } else if (!strcmp(type, "indexed_recordio")) {
    if (index_uri_ != nullptr) {
      io::URISpec index_spec(index_uri_, part, nsplit);
      split = new io::IndexedRecordIOSplitter(
          io::FileSystem::GetInstance(path), spec.uri.c_str(),
          index_spec.uri.c_str(), part, nsplit, batch_size, shuffle, seed);
    } else {
      LOG(FATAL) << "need to pass index file to use IndexedRecordIO";
    }
  } else if (!strcmp(type, "recordio")) {
    split = new io::RecordIOSplitter(io::FileSystem::GetInstance(path),
                                     spec.uri.c_str(), part, nsplit,
                                     recurse_directories);
  } else {
    LOG(FATAL) << "unknown input split type " << type;
  }
  if (spec.cache_file.length() == 0) {
    return new io::SingleThreadedInputSplit(split, batch_size);
  } else {
    return new io::CachedInputSplit(split, spec.cache_file.c_str());
  }
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType, DType> *
CreateCSVParser(const std::string& path,
                const std::map<std::string, std::string>& args,
                unsigned part_index,
                unsigned num_parts) {
  dmlc::InputSplit *source =
      CreateInputSplit(path.c_str(), nullptr, part_index, num_parts, "text");
  return new dmlc::data::CSVParser<IndexType, DType>(source, args, 2);
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType> *
CreateLibSVMParser(const std::string& path,
                   const std::map<std::string, std::string>& args,
                   unsigned part_index,
                   unsigned num_parts) {
  dmlc::InputSplit *source =
      CreateInputSplit(path.c_str(), nullptr, part_index, num_parts, "text");
  dmlc::data::ParserImpl<IndexType> *parser =
      new dmlc::data::LibSVMParser<IndexType>(source, args, 2);
  return parser;
}

template<typename IndexType, typename DType = float>
dmlc::Parser<IndexType> *
CreateLibFMParser(const std::string& path,
                  const std::map<std::string, std::string>& args,
                  unsigned part_index,
                  unsigned num_parts) {
  dmlc::InputSplit *source =
      CreateInputSplit(path.c_str(), nullptr, part_index, num_parts, "text");
  dmlc::data::ParserImpl<IndexType> *parser =
      new dmlc::data::LibFMParser<IndexType>(source, args, 2);
  return parser;
}

template <typename IndexType, typename DType = float>
inline dmlc::Parser<IndexType, DType> *
CreateParser(const char *uri_, unsigned part_index, unsigned num_parts,
             const char *type) {
  std::string ptype = type;
  dmlc::io::URISpec spec(uri_, part_index, num_parts);
  if (ptype == "auto") {
    if (spec.args.count("format") != 0) {
      ptype = spec.args.at("format");
    } else {
      ptype = "libsvm";
    }
  }

  const dmlc::ParserFactoryReg<IndexType, DType> *e =
      dmlc::Registry<dmlc::ParserFactoryReg<IndexType, DType>>::Get()->Find(
          ptype);
  if (e == NULL) {
    LOG(FATAL) << "Unknown data type " << ptype;
  }
  // create parser
  return (*e->body)(spec.uri, spec.args, part_index, num_parts);
}
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_PARSER_H_
