/*!
 * Copyright 2018 by Contributors
 * \file json.h
 * \brief JSON serialization of nested key-value store.
 */
#ifndef XGBOOST_COMMON_JSON_H_
#define XGBOOST_COMMON_JSON_H_

#include "./nested_kvstore.h"

namespace xgboost {
namespace serializer {

NestedKVStore LoadKVStoreFromJSON(std::istream* stream);
void SaveKVStoreToJSON(const NestedKVStore& kvstore, std::ostream* stream);

}   // namespace serializer
}   // namespace xgboost

#endif  // XGBOOST_COMMON_JSON_H_
