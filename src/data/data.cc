#include <xgboost/data.h>

namespace xgboost {
// implementation of inline functions
void MetaInfo::Clear() {
  num_row = num_col = 0;
  labels.clear();
  root_index.clear();
  group_ptr.clear();
  weights.clear();
  base_margin.clear();
}

void MetaInfo::SaveBinary(dmlc::Stream *fo) const {
  int version = kVersion;
  fo->Write(&version, sizeof(version));
  fo->Write(&num_row, sizeof(num_row));
  fo->Write(&num_col, sizeof(num_col));
  fo->Write(labels);
  fo->Write(group_ptr);
  fo->Write(weights);
  fo->Write(root_index);
  fo->Write(base_margin);
}

void MetaInfo::LoadBinary(dmlc::Stream *fi) {
  int version;
  CHECK(fi->Read(&version, sizeof(version)) == sizeof(version)) << "MetaInfo: invalid format";
  CHECK_EQ(version, kVersion) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_row, sizeof(num_row)) == sizeof(num_row)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&num_col, sizeof(num_col)) == sizeof(num_col)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&labels)) <<  "MetaInfo: invalid format";
  CHECK(fi->Read(&group_ptr)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&weights)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&root_index)) << "MetaInfo: invalid format";
  CHECK(fi->Read(&base_margin)) << "MetaInfo: invalid format";
}
}  // namespace xgboost
