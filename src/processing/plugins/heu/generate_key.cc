/**
 * Copyright 2014-2024 by XGBoost Contributors
 */

#include <fcntl.h>

#include "gflags/gflags.h"
#include "heu/library/phe/phe.h"

int GenerateFile(std::string_view file_name, std::string_view buf) {
  int fd = open(file_name.data(), O_CREAT | O_TRUNC | O_WRONLY, 0664);
  YACL_ENFORCE(fd != -1, "errno {}, {}", errno, strerror(errno));

  auto ret = write(fd, buf.data(), buf.size());
  YACL_ENFORCE(ret != -1, "errno {}, {}", errno, strerror(errno));
  close(fd);
  return 0;
}

DEFINE_string(schema, "ou", "Schema");
DEFINE_int32(key_size, 2048, "Key size of phe schema.");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  fmt::print("schema: {}, key_size: {}\n", FLAGS_schema, FLAGS_key_size);
  auto schema_type = heu::lib::phe::ParseSchemaType(FLAGS_schema);
  auto he_kit =
      std::make_unique<heu::lib::phe::HeKit>(schema_type, FLAGS_key_size);
  auto pk = he_kit->GetPublicKey()->Serialize();
  auto sk = he_kit->GetSecretKey()->Serialize();
  GenerateFile("public-key", pk);
  GenerateFile("secret-key", sk);
  fmt::print("generate key files done\n");
  return 0;
}
